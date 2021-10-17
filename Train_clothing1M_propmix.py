from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture
from pathlib import Path
# from gcloud import download_clothing1M_unzip,upload_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
# parser.add_argument("--dst_bucket_namespace", type=str, default="aiml-carneiro-research", help="The namespace of the GCS bucket to write model weights to")
# parser.add_argument("--dst_bucket_name", type=str, default="aiml-carneiro-research-data", help="The name of the GCS bucket to write model weights to")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '%s'%(args.gpuid)
# torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch,net,net2,optimizer,trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    # unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(trainloader.dataset)//args.batch_size)+1

    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(trainloader):      
        # try:
        #     inputs_u, inputs_u2 = unlabeled_train_iter.next()
        # except:
        #     unlabeled_train_iter = iter(unlabeled_trainloader)
        #     inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        # inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            # outputs_u11 = net(inputs_u)
            # outputs_u12 = net(inputs_u2)
            # outputs_u21 = net2(inputs_u)
            # outputs_u22 = net2(inputs_u2)            
            
            # pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            # ptu = pu**(1/args.T) # temparature sharpening
            
            # targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            # targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x11 = net(inputs_x)
            outputs_x12 = net(inputs_x2)   
            outputs_x21 = net2(inputs_x)
            outputs_x22 = net2(inputs_x2)            
            
            # px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = (torch.softmax(outputs_x11, dim=1) + torch.softmax(outputs_x12, dim=1) + torch.softmax(outputs_x21, dim=1) + torch.softmax(outputs_x22, dim=1)) / 4
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        
        # all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_x, inputs_x2], dim=0)
        # all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_x, targets_x], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

        
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()
    
def val(net,val_loader,k):
    save_path = path_exp
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = '%s/net%d.pth.tar'%(save_path,k)
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)       
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc    
    
def eval_train(epoch,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    preds_classes = torch.zeros(num_samples, args.num_class) 
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            eval_preds = F.softmax(outputs, -1).cpu().data
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                preds_classes[n] =  eval_preds[b]  
                n+=1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths, preds_classes
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
def create_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048,args.num_class)
    model = nn.DataParallel(model)
    model = model.cuda()
    return model     

def save_models(epoch, net1, optimizer1, net2, optimizer2):
    save_path = path_exp
    state = ({
                    'epoch'     : epoch,
                    'state_dict1'     : net1.state_dict(),
                    'optimizer1'      : optimizer1.state_dict(),
                    # 'scheduler1': scheduler1,
                    'state_dict2'     : net2.state_dict(),
                    'optimizer2'      : optimizer2.state_dict(),
                    'prob1':    prob1,
                    'prob2':    prob2
                    
                    # 'scheduler2': scheduler2,
                })
    

    # if (epoch%80==0) :
    #     fn = os.path.join(save_path, 'model_%d.pth.tar'%epoch)
    #     torch.save(state, fn)
        # fn_log = os.path.join(save_path, 'model_%d_log.pth.tar'%epoch)
        # torch.save(state2, fn_log)

    if epoch%1==0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        # fn2_log = os.path.join(save_path, 'model_ckpt_hist.pth.tar')
        # torch.save(state2, fn2_log) 
    # upload_checkpoint(args.dst_bucket_namespace, args.dst_bucket_name, path_acc_test)
    # upload_checkpoint(args.dst_bucket_namespace, args.dst_bucket_name, path_acc_val)
    # upload_checkpoint(args.dst_bucket_namespace, args.dst_bucket_name, '%s/net1.pth.tar'%path_exp)  
    # upload_checkpoint(args.dst_bucket_namespace, args.dst_bucket_name, '%s/net2.pth.tar'%path_exp)  

name_method = 'propmix'
exp_str = 'clothing1Mrnd_yespretrained_%s_lu_%d'%(name_method, int(args.lambda_u))
path_exp='results/clothing1M/' + exp_str

Path(path_exp).mkdir(parents=True, exist_ok=True)
# try:
#     os.stat(path_exp)
# except:
#     os.mkdir(resultados)
#     os.mkdir(path_exp)

log=open('%s/val_acc.txt'%path_exp,'w')     
log.flush()

log_test=open('%s/test_acc.txt'%path_exp,'w')     
log_test.flush()

path_acc_val = '%s/val_acc.txt'%path_exp
path_acc_test = '%s/test_acc.txt'%path_exp

# download_clothing1M_unzip(data_dir=args.data_path,bucket_namespace=args.dst_bucket_namespace,bucket_name=args.dst_bucket_name)

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

test_loader = loader.run('test')

best_acc = [0,0]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    if epoch<1:     # warm up  
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)  

        #online training
        idx_noisy1 = (1-pred1).nonzero()[0] 
        idx_noisy2 = (1-pred2).nonzero()[0] 

        equal_view_noisy = [i for i in idx_noisy1 if i in idx_noisy2]

        preds_class_guess1 =  torch.max(pl1_classes, dim=-1)[1][equal_view_noisy]

        preds_noisy_samples1 = pl1_classes[equal_view_noisy]
        preds_noisy_samples2 = pl2_classes[equal_view_noisy]
        # joint_pred = pl1_classes[idx_noisy1]
        
        joint_pred = (preds_noisy_samples1+preds_noisy_samples2)/2.0

        ### net 1
        sort_distances = np.sort(joint_pred, 1)[:, -2:]
        # min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        min_margin = sort_distances[:, 1] 
        rank_ind = np.argsort(min_margin)
        ranked_idx_noisy = np.array(idx_noisy1)[rank_ind]
        
        input_pred = min_margin.reshape(-1,1)
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(input_pred)
        prob = gmm.predict_proba(input_pred) 
        prob = prob[:,gmm.means_.argmin()]  
        #idx_gmm_rem = (prob>=args.tag_th).nonzero()[0]
        idx_gmm_rem = (prob>=0.5).nonzero()[0]
        balanced_idx = np.array(equal_view_noisy)[idx_gmm_rem] 
        # balanced_idx = np.array(idx_noisy1)[idx_gmm_rem] 

        idx_keep = (prob>=0.5).nonzero()[0]  


        prob2[idx_noisy2] = 0 #set the probability of being clean to 0 (this force the label to be the prediction in the training stage)
        prob1[idx_noisy1] = 0
        # meta_info['probability'] = prob2
        # meta_info['pred'] = pred2 #doesnt matter in this approach
        # meta_info['idx_remove'] = balanced_idx    
        
        print('\n\nTrain Net1')
        trainloader = loader.run('train',pred2,prob2,paths=paths2, idx_remove=balanced_idx) # co-divide
        train(epoch,net1,net2,optimizer1,trainloader)              # train net1
        print('\nTrain Net2')
        trainloader = loader.run('train',pred1,prob1,paths=paths1, idx_remove=balanced_idx) # co-divide
        train(epoch,net2,net1,optimizer2,trainloader)              # train net2
    
    val_loader = loader.run('val') # validation
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1, pl1_classes = eval_train(epoch,net1) 
    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  
    prob2,paths2, pl2_classes = eval_train(epoch,net2) 

    acc = test(net1,net2,test_loader)     
    log_test.write('Epoch:%d\tTest Accuracy:%.2f\n'%(epoch, acc)) 
    log_test.flush() 

    save_models(epoch, net1, optimizer1, net2, optimizer2)


net1.load_state_dict(torch.load('%s/net1.pth.tar'%path_exp))
net2.load_state_dict(torch.load('%s/net2.pth.tar'%path_exp))
acc = test(net1,net2,test_loader)      

log.write('Test Accuracy:%.2f\n'%(acc))
log.flush() 

# upload_checkpoint(args.dst_bucket_namespace, args.dst_bucket_name, path_acc_test)
# upload_checkpoint(args.dst_bucket_namespace, args.dst_bucket_name, path_acc_val)
