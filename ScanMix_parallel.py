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
from utils.evaluate_utils import scanmix_big_test, scanmix_test
from utils.train_utils import scanmix_big_train, scanmix_big_eval_train, scanmix_big_warmup, scanmix_scan
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='DivideMix')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--r', default=0, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--load_state_dict', default=None, type=str)
parser.add_argument('--cudaids', nargs=2, type=int)
parser.add_argument('--lr_sl', type=float, required=True)

parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()
p = create_config(args.config_env, args.config_exp)

os.environ["CUDA_VISIBLE_DEVICES"] = '%s,%s'%(args.cudaids[0],args.cudaids[1])

device_1 = torch.device('cuda:{}'.format(0))
device_2 = torch.device('cuda:{}'.format(1))

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#meta_info
meta_info = copy.deepcopy(args.__dict__)
meta_info['dataset'] = p['dataset']
meta_info['probability'] = None
meta_info['pred'] = None
meta_info['noise_rate'] = args.r

checkpoint_dir = 'results/{}/scanmix/'.format(p['train_db_name'])
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.join(checkpoint_dir, 'savedDicts')).mkdir(parents=True, exist_ok=True)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(device):
    model = get_model(p, p['scan_model'])
    model = model.to(device)
    return model

test_log = open(os.path.join(checkpoint_dir, 'acc_{}.txt'.format(args.lr_sl)), 'w')
stats_log = open(os.path.join(checkpoint_dir, 'stats_{}.txt'.format(args.lr_sl)), 'w') 

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
        train_transformations = get_scan_transformations(p)
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
    net1 = create_model(device_1)
    net2 = create_model(device_2)

    net1_clone = create_model(device_2)
    net2_clone = create_model(device_1)

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
        if epoch >= (p['num_epochs']/2):
            lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr        
        
        if epoch<p['warmup']:       
            warmup_trainloader1 = get_loader(p, 'warmup', meta_info)
            warmup_trainloader2 = get_loader(p, 'warmup', meta_info)
            p1 = mp.Process(target=scanmix_big_warmup, args=(p,epoch,net1,optimizer1,warmup_trainloader1, CEloss, conf_penalty, args.noise_mode, device_1))    
            p2 = mp.Process(target=scanmix_big_warmup, args=(p,epoch,net2,optimizer2,warmup_trainloader2, CEloss, conf_penalty, args.noise_mode, device_2))
            p1.start()
            p2.start() 
    
        else:                  
                
            pred1 = (prob1 > p['p_threshold'])      
            pred2 = (prob2 > p['p_threshold'])      

            print('[DM] Train')
            meta_info['probability'] = prob2
            meta_info['pred'] = pred2
            labeled_trainloader1, unlabeled_trainloader1 = get_loader(p, 'train', meta_info)
            meta_info['probability'] = prob1
            meta_info['pred'] = pred1
            labeled_trainloader2, unlabeled_trainloader2 = get_loader(p, 'train', meta_info)
            
            p1 = mp.Process(target=scanmix_big_train, args=(p, epoch,net1,net2_clone,optimizer1,labeled_trainloader1, unlabeled_trainloader1, criterion_dm, args.lambda_u, device_1)) 
            p2 = mp.Process(target=scanmix_big_train, args=(p, epoch,net2,net1_clone,optimizer2,labeled_trainloader2, unlabeled_trainloader2, criterion_dm, args.lambda_u, device_2))
            p1.start()
            p2.start()

            p1.join()
            p2.join()

            for param_group in optimizer1.param_groups:
                param_group['lr'] = args.lr_sl    
            for param_group in optimizer2.param_groups:
                param_group['lr'] = args.lr_sl  

            print('\n[SL] Train')
            meta_info['predicted_labels'] = pl_2   
            neighbor_dataloader1 = get_loader(p, 'neighbors', meta_info)
            meta_info['predicted_labels'] = pl_1  
            neighbor_dataloader2 = get_loader(p, 'neighbors', meta_info)
            p1 = mp.Process(target=scanmix_scan,args=(neighbor_dataloader1, net1, criterion_sl, optimizer1, epoch, device_1))
            p2 = mp.Process(target=scanmix_scan,args=(neighbor_dataloader2, net2, criterion_sl, optimizer2, epoch, device_2))

            p1.start()
            p2.start()

        p1.join()
        p2.join()

        net1_clone.load_state_dict(net1.state_dict())
        net2_clone.load_state_dict(net2.state_dict())

        test_loader = get_loader(p, 'test', meta_info)

        eval_loader1 = get_loader(p, 'eval_train', meta_info)  
        eval_loader2 = get_loader(p, 'eval_train', meta_info)  
        
        manager = mp.Manager()
        output1 = manager.dict()
        output2 = manager.dict()

        p1 = mp.Process(target=scanmix_big_eval_train, args=(p,args,net1,epoch, eval_loader1, CE, device_1, output1))
        p2 = mp.Process(target=scanmix_big_eval_train, args=(p,args,net2,epoch, eval_loader2, CE, device_2, output2))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        prob1, pl_1 = output1['prob'], output1['pl']
        prob2, pl_2 = output2['prob'], output2['pl']

        q1 = mp.Queue()
        p1 = mp.Process(target=scanmix_big_test, args=(epoch,net1,net2_clone,test_loader,device_1, q1))

        p1.start()
        acc = q1.get()
        p1.join()

        print('\nEpoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush() 

        if (epoch+1) % 5 == 0:
            torch.save({
                        'net1_state_dict': net1.state_dict(),
                        'net2_state_dict': net2.state_dict(),
                        'epoch': epoch,
                        'optimizer1': optimizer1.state_dict(),
                        'optimizer2': optimizer2.state_dict(),
                        }, os.path.join(checkpoint_dir, 'savedDicts/checkpoint.json'))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()