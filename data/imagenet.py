"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob



class ImageNetSubset(data.Dataset):
    
    def __init__(self, root_dir, split='val', transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = root_dir
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        classes100 = []
        classes1000 = []
        all_50k_images = [ x for x in os.listdir(self.root) if os.path.isfile(self.root+x)]
        all_50k_images.sort()
        self.val_imgs = []    #filtered images containing the first 100 classes
        self.val_labels = {}

        with open(self.root+'data/ILSVRC2012_validation_to_webvision_labels_ground_truth.txt') as f:
            lines=f.readlines()
            for li in lines:
                classes1000.append(int(li))

        assert(len(classes1000)==len(all_50k_images))

        for id, img_name in enumerate(all_50k_images):
            if int(classes1000[id])<50:  #does not have class 0
                self.val_imgs.append(img_name)
                self.val_labels[img_name] = int(classes1000[id])

    def __len__(self):
        return len(self.val_imgs)

    def __getitem__(self, index):
        img_path = self.val_imgs[index]
        target = self.val_labels[img_path]     
        img = Image.open(self.root+img_path).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
        out = {'image': img, 'target': target, 'meta': {'index': index}}

        return out
