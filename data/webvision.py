from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os

class Webvision(Dataset): 
    def __init__(self, root_dir, transform, meta_info, num_classes): 
        self.root = root_dir
        self.transform = transform
        self.mode = meta_info['mode']
        pred = meta_info['pred']
        num_class = num_classes
        self.probability = meta_info['probability']  
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            if self.mode == 'all' or self.mode == 'neighbor':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [self.probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                  
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))  
                elif self.mode == 'proportional':
                    temp_indices = list(range(len(pred)))
                    pred_idx = [i for i in temp_indices if i not in meta_info['idx_remove']] 
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [self.probability[i] for i in pred_idx]     
                
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)   
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            img = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            if self.transform is not None:
                img = self.transform(img)
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out
        elif self.mode=='neighbor':
            img_path = self.train_imgs[index]
            img = Image.open(self.root+img_path).convert('RGB')
            target = self.train_labels[img_path]
            if self.transform is not None:
                img = self.transform(img)
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out
        elif self.mode=='proportional':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob  
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    