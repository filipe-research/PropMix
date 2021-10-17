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
from sklearn.mixture import GaussianMixture
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations, get_scan_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_model, get_criterion
from utils.evaluate_utils import scanmix_test, test
from utils.train_utils import  scanmix_warmup,  scanmix_eval_train_classes, scanmix_train_proportional
from utils.plot_utils import compute_histogram_bins, plot_gmm_remove_noisy, plot_histogram_loss, plot_modelview_histogram_loss
import pickle

parser = argparse.ArgumentParser(description='PropMix_plots')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--r', default=0, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
# parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--inference', default=None, type=str)
parser.add_argument('--load_state_dict', default=None, type=str)
parser.add_argument('--cudaid', default=0)
# parser.add_argument('--dividemix_only', action='store_true')
parser.add_argument('--strong_aug', action='store_true')
parser.add_argument('--single_pred', action='store_true')
# parser.add_argument('--big', action='store_true')
# parser.add_argument('--lr_sl', type=float, default=None)

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
elif args.noise_mode == 'sym':
    meta_info['noise_file'] += '_sym'
# elif 'semantic' in args.noise_mode:
#     meta_info['noise_file'] += '_{}'.format(args.noise_mode)
# meta_info['noise_file'] = '{}/{}'.format(p['noise_dir'],args.noise_mode)

meta_info['noise_file'] += '.json'
meta_info['probability'] = None
meta_info['pred'] = None

if args.strong_aug:
    p['augmentation_strategy'] = 'ours'

# if args.single_pred:
#     p['propmix_dir'] = p['propmix_dir']+"_single"

Path(os.path.join(p['propmix_dir']+'_plots', 'savedDicts')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(p['propmix_dir']+'_plots', 'plots')).mkdir(parents=True, exist_ok=True)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    
    model = get_model(p, p['scan_model'])
    model = model.to(device)
    return model

test_log = open(os.path.join(p['propmix_dir']+'_plots', 'acc.txt'), 'w')
stats_log = open(os.path.join(p['propmix_dir']+'_plots', 'stats.txt'), 'w')

def get_loader(p, mode, meta_info):
    if mode == 'test':
        meta_info['mode'] = 'test'
        val_transformations = get_val_transformations(p)
        val_dataset = get_val_dataset(p, val_transformations, meta_info=meta_info)
        val_dataloader = get_val_dataloader(p, val_dataset)
        return val_dataloader
    
    elif mode == 'train':
        # meta_info['mode'] = 'labeled'
        meta_info['mode'] = 'proportional'
        train_transformations = get_train_transformations(p)
        train_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        # labeled_dataloader = get_train_dataloader(p, labeled_dataset)
        # meta_info['mode'] = 'unlabeled'
        # unlabeled_dataset = get_train_dataset(p, train_transformations, 
        #                                 split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        # unlabeled_dataloader = get_train_dataloader(p, unlabeled_dataset)
        train_dataloader = get_train_dataloader(p, train_dataset)
        # return labeled_dataloader, unlabeled_dataloader
        return train_dataloader

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
        acc_hist=checkpoint['acc_hist']
        # test current state
        test_loader = get_loader(p, 'test', meta_info)
        acc = scanmix_test(start_epoch-1,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.flush()
    else:
        start_epoch = 0
        acc_hist=[]

    all_loss = [[],[]] # save the history of losses from two networks
    train_hist = {}
    

    for epoch in range(start_epoch, p['num_epochs']+1):   
        lr=p['lr']
        if args.noise_mode in ['semantic_desnenet','semantic_resnet','semantic_vgg']:
            if p['train_db_name']=='cifar-10':
                if epoch >= 40:
                    lr /= 10      
                    if epoch >= 80:    
                        lr /= 10      
            elif p['train_db_name']=='cifar-100':
                if epoch >= 80:
                    lr /= 10      
                    if epoch >= 120:    
                        lr /= 10   
                        if epoch >= 160:
                            lr /= 10   
        else:
            if epoch >= 150:
                lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr        
        test_loader = get_loader(p, 'test', meta_info)
        eval_loader = get_loader(p, 'eval_train', meta_info)  

        
        
        clean_labels = eval_loader.dataset.clean_label
        noisy_labels = eval_loader.dataset.noise_label
        

        inds_noisy = np.asarray([ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]])
        inds_clean = np.delete(np.arange(len(noisy_labels)), inds_noisy)

        

        train_hist['clean_labels'] = clean_labels
        train_hist['noisy_labels'] = noisy_labels
        train_hist['inds_clean'] = inds_clean
        train_hist['inds_noisy'] = inds_noisy

         
        if epoch<p['warmup']:       
            warmup_trainloader = get_loader(p, 'warmup', meta_info)
            print('Warmup Net1')
            scanmix_warmup(epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)    
            scanmix_warmup(epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)
            # if not args.big:
            #     scanmix_warmup(epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)    
            # else:
            #     scanmix_big_warmup(p, epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)    
            # print('\nWarmup Net2')
            # if not args.big:
            #     scanmix_warmup(epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)
            # else:
            #     scanmix_big_warmup(p, epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)

    
        else:  

            prob1,all_loss[0],pl_1, pl1_classes=scanmix_eval_train_classes(args,net1,all_loss[0], epoch, eval_loader, CE, device=device, num_classes=p['num_classes'])   
            prob2,all_loss[1],pl_2, pl2_classes=scanmix_eval_train_classes(args,net2,all_loss[1], epoch, eval_loader, CE, device=device, num_classes=p['num_classes'])          
            # if not args.big:       
            #     prob1,all_loss[0],pl_1=scanmix_eval_train(args,net1,all_loss[0], epoch, eval_loader, CE, device=device)   
            #     prob2,all_loss[1],pl_2=scanmix_eval_train(args,net2,all_loss[1], epoch, eval_loader, CE, device=device)  
            # else:   
            #     prob1,all_loss[0],pl_1=scanmix_big_eval_train(p, args,net1,all_loss[0], epoch, eval_loader, CE, device=device)   
            #     prob2,all_loss[1],pl_2=scanmix_big_eval_train(p, args,net2,all_loss[1], epoch, eval_loader, CE, device=device)       
                
            pred1 = (prob1 > p['p_threshold'])      
            pred2 = (prob2 > p['p_threshold'])    

            #online training
            idx_noisy1 = (1-pred1).nonzero()[0] 
            idx_noisy2 = (1-pred2).nonzero()[0] 

            equal_view_noisy = [i for i in idx_noisy1 if i in idx_noisy2]

            preds_class_guess1 =  torch.max(pl1_classes, dim=-1)[1][equal_view_noisy]
            true_label_noisy_view1 = np.array(clean_labels)[equal_view_noisy]

            idx_guess_correct = [c for c in range(len(preds_class_guess1)) if preds_class_guess1[c]==true_label_noisy_view1[c]]
            idx_guess_wrong = [c for c in range(len(preds_class_guess1)) if preds_class_guess1[c]!=true_label_noisy_view1[c]]

            preds_noisy_samples1 = pl1_classes[equal_view_noisy]
            preds_noisy_samples2 = pl2_classes[equal_view_noisy]
            # joint_pred = pl1_classes[idx_noisy1]
            if args.single_pred:
                joint_pred = (preds_noisy_samples1+preds_noisy_samples1)/2.0
            else:
                joint_pred = (preds_noisy_samples1+preds_noisy_samples2)/2.0

            ### net 1
            sort_distances = np.sort(joint_pred, 1)[:, -2:]
            # min_margin = sort_distances[:, 1] - sort_distances[:, 0]
            min_margin = sort_distances[:, 1] 
            rank_ind = np.argsort(min_margin)
            ranked_idx_noisy = np.array(idx_noisy1)[rank_ind]
            
            
            input_pred = min_margin.reshape(-1,1)
            if len(input_pred)>1:
                gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
                gmm.fit(input_pred)
                prob = gmm.predict_proba(input_pred) 
                prob = prob[:,gmm.means_.argmin()]  
                #idx_gmm_rem = (prob>=args.tag_th).nonzero()[0]
                #idx_gmm_rem = (prob>=0.5).nonzero()[0]
                idx_gmm_rem = (prob>=p['p_threshold2']).nonzero()[0]
                balanced_idx = np.array(equal_view_noisy)[idx_gmm_rem] 
                # balanced_idx = np.array(idx_noisy1)[idx_gmm_rem] 

                #idx_keep = (prob>=0.5).nonzero()[0]  
                idx_keep = (prob>=p['p_threshold2']).nonzero()[0]  
            else:
                balanced_idx=[]
                idx_keep = []  #wrong, it should be idx_remove

            idx_clean1 = (pred1).nonzero()[0] 

            
            correct_removal = [i for i in balanced_idx if i in inds_noisy]
            wrong_removal = [i for i in balanced_idx if i in inds_clean]
            correct_keep = [i for i in np.array(equal_view_noisy)[idx_keep] if i in inds_clean]
            wrong_keep = [i for i in np.array(equal_view_noisy)[idx_keep] if i in inds_noisy]

            
            train_hist[epoch]={}
            train_hist[epoch]['loss']= all_loss[0][-1].numpy()
            train_hist[epoch]['idx_view_unlabeled']= idx_noisy1
            train_hist[epoch]['idx_view_labeled']= idx_clean1
            train_hist[epoch]['equal_view_noisy']= equal_view_noisy
            train_hist[epoch]['idx_guess_correct']= idx_guess_correct
            train_hist[epoch]['idx_guess_wrong']= idx_guess_wrong
            train_hist[epoch]['idx_remove']= balanced_idx  
            train_hist[epoch]['idx_keep']= idx_keep   #wrong
            train_hist[epoch]['correct_removal'] = correct_removal  #wrong
            train_hist[epoch]['wrong_removal'] = wrong_removal  #wrong
            train_hist[epoch]['correct_keep'] = correct_keep  #wrong
            train_hist[epoch]['wrong_keep'] = wrong_keep #wrong
            train_hist[epoch]['min_margin'] = min_margin
            







            if epoch%10==0:
                
                if len(idx_keep)>0:
                    path_plot = os.path.join(p['propmix_dir']+'_plots','plots/noisy_conf_ep%03d.png'% (epoch))
                    plot_gmm_remove_noisy(path_plot , min_margin, idx_guess_correct, idx_guess_wrong, idx_keep)
                    path_plot = os.path.join(p['propmix_dir']+'_plots','plots/sep_loss_epoch%03d.png'% (epoch))
                    plot_histogram_loss(path_plot, all_loss, inds_clean, inds_noisy)
                    path_plot = os.path.join(p['propmix_dir']+'_plots','plots/view_sep_loss_epoch%03d.png' % (epoch))
                    plot_modelview_histogram_loss(path_plot, all_loss, inds_clean, inds_noisy, idx_view_labeled=idx_clean1, idx_view_unlabeled=idx_noisy1)

            print('[DM] Train Net1')

            prob2[idx_noisy2] = 0 #set the probability of being clean to 0 (this force the label to be the prediction in the training stage)
            meta_info['probability'] = prob2
            meta_info['pred'] = pred2 #doesnt matter in this approach
            meta_info['idx_remove'] = balanced_idx
            trainloader = get_loader(p, 'train', meta_info)

            scanmix_train_proportional(p, epoch,net1,net2,optimizer1,trainloader, criterion_dm, args.lambda_u, device=device) # train net1  

            # if not args.big:
            #     scanmix_train(p, epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net1  
            # else:
            #     scanmix_big_train(p, epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net1  
            
            print('\n[DM] Train Net2')
            prob1[idx_noisy1] = 0 #set the probability of being clean to 0 (this force the label to be the prediction in the training stage)
            meta_info['probability'] = prob1
            meta_info['pred'] = pred1 #doesnt matter in this approach
            meta_info['idx_remove'] = balanced_idx
            trainloader = get_loader(p, 'train', meta_info)
            scanmix_train_proportional(p, epoch,net2,net1,optimizer2,trainloader, criterion_dm, args.lambda_u, device=device) # train net2       

            # meta_info['probability'] = prob1
            # meta_info['pred'] = pred1
            # labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            # if not args.big:
            #     scanmix_train(p, epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net2       
            # else:
            #     scanmix_big_train(p, epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device) # train net2       
            
            # if not args.dividemix_only:
            #     for param_group in optimizer1.param_groups:
            #         param_group['lr'] = args.lr_sl    
            #     for param_group in optimizer2.param_groups:
            #         param_group['lr'] = args.lr_sl  
            #     meta_info['predicted_labels'] = pl_2   
            #     neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
            #     print('\n[SL] Train Net1')
            #     scanmix_scan(neighbor_dataloader, net1, criterion_sl, optimizer1, epoch, device)
            #     meta_info['predicted_labels'] = pl_1  
            #     neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
            #     print('\n[SL] Train Net2')
            #     scanmix_scan(neighbor_dataloader, net2, criterion_sl, optimizer2, epoch, device)

        acc = test(epoch,net1,net2,test_loader, device=device)
        acc_hist.append(acc)
        # if not args.big:
        #     acc = scanmix_test(epoch,net1,net2,test_loader, device=device)
        # else:
        #     acc = scanmix_big_test(epoch,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush() 
        torch.save({
                    'net1_state_dict': net1.state_dict(),
                    'net2_state_dict': net2.state_dict(),
                    'epoch': epoch,
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'acc_hist': acc_hist,
                    }, os.path.join(p['propmix_dir']+'_plots', 'savedDicts/checkpoint.json'))

    print('\nBest:%.2f  avgLast10: %.2f\n'%(max(acc_hist),sum(acc_hist[-10:])/10.0))
    test_log.write('\nBest:%.2f  avgLast10: %.2f\n'%(max(acc_hist),sum(acc_hist[-10:])/10.0))
    test_log.flush() 

    #saving dict for plots
    with open(os.path.join(p['propmix_dir']+'_plots', 'savedDicts/train_hist.pickle'), 'wb') as fp:
        pickle.dump(train_hist, fp)


if __name__ == "__main__":
    main()