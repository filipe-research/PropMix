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


# class ImageNet(datasets.ImageFolder):
#     def __init__(self, root=MyPath.db_root_dir('imagenet'), split='train', transform=None):
#         super(ImageNet, self).__init__(root=os.path.join(root, 'ILSVRC2012_img_%s' %(split)),
#                                          transform=None)
#         self.transform = transform 
#         self.split = split
#         self.resize = tf.Resize(256)

    
#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, index):
#         path, target = self.imgs[index]
#         with open(path, 'rb') as f:
#             img = Image.open(f).convert('RGB')
#         im_size = img.size
#         img = self.resize(img)
        
#         if self.transform is not None:
#             img = self.transform(img)

#         out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

#         return out

#     def get_image(self, index):
#         path, target = self.imgs[index]
#         with open(path, 'rb') as f:
#             img = Image.open(f).convert('RGB')
#         img = self.resize(img) 
#         return img


class ImageNetSubset(data.Dataset):
    #def __init__(self, subset_file, root=MyPath.db_root_dir('imagenet'), split='train', 
    def __init__(self, root_dir, split='val', transform=None):
        super(ImageNetSubset, self).__init__()

        #self.root = os.path.join(root, 'ILSVRC2012_img_%s' %(split))
        self.root = root_dir
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        # with open(subset_file, 'r') as f:
        #     result = f.read().splitlines()
        classes100 = []
        classes1000 = []
        all_50k_images = [ x for x in os.listdir(self.root) if os.path.isfile(self.root+x)]
        all_50k_images.sort()
        self.val_imgs = []    #filtered images containing the first 100 classes
        self.val_labels = {}

        
        

            

        #with open(self.root+'data/ILSVRC2012_validation_ground_truth.txt') as f:
        with open(self.root+'data/ILSVRC2012_validation_to_webvision_labels_ground_truth.txt') as f:
            lines=f.readlines()
            for li in lines:
                classes1000.append(int(li))
                # print(li.split()[0].rstrip())d
                # classes.append(li.split()[0].rstrip())   

        assert(len(classes1000)==len(all_50k_images))

        for id, img_name in enumerate(all_50k_images):
            if int(classes1000[id])<50:  #does not have class 0
                self.val_imgs.append(img_name)
                #self.val_labels.append(classes1000[id])
                self.val_labels[img_name] = int(classes1000[id])
        # import pdb; pdb.set_trace()

        
        
        # import pdb; pdb.set_trace()
                
        

    # def get_image(self, index):
    #     path, target = self.imgs[index]
    #     with open(path, 'rb') as f:
    #         img = Image.open(f).convert('RGB')
    #     img = self.resize(img) 
    #     return img

    def __len__(self):
        return len(self.val_imgs)

    def __getitem__(self, index):
        img_path = self.val_imgs[index]
        target = self.val_labels[img_path]     
        img = Image.open(self.root+img_path).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
        out = {'image': img, 'target': target, 'meta': {'index': index}}
        # out = {'image': img, 'target': target, 'meta': {'im_size': size, 'index': index, 'class_name': class_name}}

        return out
