from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import copy
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations, get_scan_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_model, get_criterion
from utils.evaluate_utils import scanmix_test, scanmix_big_test
from utils.train_utils import scanmix_train, scanmix_eval_train, scanmix_warmup, scanmix_scan, scanmix_big_train, scanmix_big_eval_train, scanmix_big_warmup

parser = argparse.ArgumentParser(description='DivideMix')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--r', default=0, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--inference', default=None, type=str)
parser.add_argument('--load_state_dict', default=None, type=str)
parser.add_argument('--cudaid', default=0)
parser.add_argument('--dividemix_only', action='store_true')
parser.add_argument('--nopretrain', action='store_true')
parser.add_argument('--big', action='store_true')
parser.add_argument('--lr_sl', type=float, default=None)

parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

device = device = torch.device('cuda:{}'.format(args.cudaid))

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#meta_info
meta_info = copy.deepcopy(args.__dict__)
p = create_config(args.config_env, args.config_exp, meta_info)
meta_info['dataset'] = p['dataset']
meta_info['noise_file'] = '{}/{:.2f}'.format(p['noise_dir'], args.r)
meta_info['noise_rate'] = args.r
if args.noise_mode == 'asym':
    meta_info['noise_file'] += '_asym'
elif 'semantic' in args.noise_mode:
    meta_info['noise_file'] += '_{}'.format(args.noise_mode)
meta_info['noise_file'] += '.json'
meta_info['probability'] = None
meta_info['pred'] = None

Path(os.path.join(p['scanmix_dir'], 'savedDicts')).mkdir(parents=True, exist_ok=True)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    #model = get_model(p, p['scan_model'])
    #pretrain_path = 'byol/essential-byol/17wed0of/checkpoints/epoch=15-step=2938.ckpt'
    pretrain_path = 'byol/2ibxjln7/checkpoints/epoch=49-step=9749.ckpt'
    if args.nopretrain:
        model = get_model(p)
    else:
        model = get_model(p, pretrain_path)
    model = model.to(device)
    return model

test_log = open(os.path.join(p['scanmix_dir'], 'acc.txt'), 'w')
stats_log = open(os.path.join(p['scanmix_dir'], 'stats.txt'), 'w')

def get_loader(p, mode, meta_info):
    if mode == 'test':
        meta_info['mode'] = 'test'
        val_transformations = get_val_transformations(p)
        val_dataset = get_val_dataset(p, val_transformations, meta_info=meta_info)
        val_dataloader = get_val_dataloader(p, val_dataset)
        return val_dataloader
    
    elif mode == 'train':
        meta_info['mode'] = 'labeled'
        train_transformations = get_train_transformations(p)
        labeled_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        labeled_dataloader = get_train_dataloader(p, labeled_dataset)
        meta_info['mode'] = 'unlabeled'
        unlabeled_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        unlabeled_dataloader = get_train_dataloader(p, unlabeled_dataset)
        return labeled_dataloader, unlabeled_dataloader

    elif mode == 'eval_train':
        meta_info['mode'] = 'all'
        eval_transformations = get_val_transformations(p)
        eval_dataset = get_train_dataset(p, eval_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        eval_dataloader = get_val_dataloader(p, eval_dataset)
        return eval_dataloader
    
    elif mode == 'warmup':
        meta_info['mode'] = 'all'
        warmup_transformations = get_train_transformations(p)
        warmup_dataset = get_train_dataset(p, warmup_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        warmup_dataloader = get_train_dataloader(p, warmup_dataset, explicit_batch_size=p['batch_size']*2)
        return warmup_dataloader

    elif mode == 'neighbors':
        meta_info['mode'] = 'neighbor'
        train_transformations = get_train_transformations(p)
        neighbor_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_neighbors_dataset=True, to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        neighbor_dataloader = get_train_dataloader(p, neighbor_dataset, explicit_batch_size=p['batch_size_scan'])
        return neighbor_dataloader
    
    else:
        raise NotImplementedError

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
criterion_dm, criterion_sl = get_criterion(p)
conf_penalty = NegEntropy()

def main():
    print('| Building net')
    net1 = create_model()
    print('created first model')
    net2 = create_model()
    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr=p['lr'], momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=p['lr'], momentum=0.9, weight_decay=5e-4)

    if args.load_state_dict is not None:
        print('Loading saved state dict from {}'.format(args.load_state_dict))
        checkpoint = torch.load(args.load_state_dict)
        net1.load_state_dict(checkpoint['net1_state_dict'])
        net2.load_state_dict(checkpoint['net2_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        start_epoch = checkpoint['epoch']+1
        # test current state
        test_loader = get_loader(p, 'test', meta_info)
        acc = scanmix_test(start_epoch-1,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.flush()
    else:
        start_epoch = 0

    all_loss = [[],[]] # save the history of losses from two networks

    
    for epoch in range(start_epoch, p['num_epochs']+1):   
        lr=p['lr']
        if epoch >= 150:
            lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr        
        test_loader = get_loader(p, 'test', meta_info)
        eval_loader = get_loader(p, 'eval_train', meta_info)  
        
        if epoch<p['warmup']:       
            warmup_trainloader = get_loader(p, 'warmup', meta_info)
            print('Warmup Net1')
            if not args.big:
                scanmix_warmup(epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)    
            else:
                scanmix_big_warmup(p, epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)    
            print('\nWarmup Net2')
            if not args.big:
                scanmix_warmup(epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)
            else:
                scanmix_big_warmup(p, epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)

            if epoch == p['warmup']-1:

                if not args.big:
                    prob1,_,_=scanmix_eval_train(args,net1,[], epoch, eval_loader, CE, device=device)   
                    prob2,_,_=scanmix_eval_train(args,net2,[], epoch, eval_loader, CE, device=device)
                else:
                    prob1,_,_=scanmix_big_eval_train(p, args,net1,[], epoch, eval_loader, CE, device=device)   
                    prob2,_,_=scanmix_big_eval_train(p, args,net2,[], epoch, eval_loader, CE, device=device)
                pred1 = (prob1 > p['p_threshold'])      
                pred2 = (prob2 > p['p_threshold'])
                noise1 = len((1-pred1).nonzero()[0])/len(pred1)
                noise2 = len((1-pred2).nonzero()[0])/len(pred2)
                predicted_noise = (noise1 + noise2) / 2
                print('\nPREDICTED NOISE RATE: {}'.format(predicted_noise))
                if predicted_noise <= 0.6:
                    args.lr_sl = 0.00001
                    p['augmentation_strategy'] = 'dividemix'
                else:
                    args.lr_sl = 0.001
                    p['augmentation_strategy'] = 'ours'
    
        else:  
            if not args.big:       
                prob1,all_loss[0],pl_1=scanmix_eval_train(args,net1,all_loss[0], epoch, eval_loader, CE, device=device)   
                prob2,all_loss[1],pl_2=scanmix_eval_train(args,net2,all_loss[1], epoch, eval_loader, CE, device=device)  
            else:   
                prob1,all_loss[0],pl_1=scanmix_big_eval_train(p, args,net1,all_loss[0], epoch, eval_loader, CE, device=device)   
                prob2,all_loss[1],pl_2=scanmix_big_eval_train(p, args,net2,all_loss[1], epoch, eval_loader, CE, device=device)       
                
            pred1 = (prob1 > p['p_threshold'])      
            pred2 = (prob2 > p['p_threshold'])      

            print('[DM] Train Net1')
            meta_info['probability'] = prob2
            meta_info['pred'] = pred2
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            if not args.big:
                scanmix_train(p, epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net1  
            else:
                scanmix_big_train(p, epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net1  
            
            print('\n[DM] Train Net2')
            meta_info['probability'] = prob1
            meta_info['pred'] = pred1
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            if not args.big:
                scanmix_train(p, epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net2       
            else:
                scanmix_big_train(p, epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net2       
            
            if not args.dividemix_only:
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = args.lr_sl    
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = args.lr_sl  
                meta_info['predicted_labels'] = pl_2   
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                print('\n[SL] Train Net1')
                scanmix_scan(neighbor_dataloader, net1, criterion_sl, optimizer1, epoch, device)
                meta_info['predicted_labels'] = pl_1  
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                print('\n[SL] Train Net2')
                scanmix_scan(neighbor_dataloader, net2, criterion_sl, optimizer2, epoch, device)

        if not args.big:
            acc = scanmix_test(epoch,net1,net2,test_loader, device=device)
        else:
            acc = scanmix_big_test(epoch,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush() 
        torch.save({
                    'net1_state_dict': net1.state_dict(),
                    'net2_state_dict': net2.state_dict(),
                    'epoch': epoch,
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    }, os.path.join(p['scanmix_dir'], 'savedDicts/checkpoint.json'))

if __name__ == "__main__":
    main()