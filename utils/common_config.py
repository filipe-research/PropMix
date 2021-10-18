
# from byol_essential.byol_models.model import BYOL
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
import pdb
import copy

 
def get_criterion(p):
    if p['criterion'] == 'simclr':
        from losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'scan':
        from losses.losses import SCANLoss
        criterion = SCANLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'confidence-cross-entropy':
        from losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])
    
    elif p['criterion'] in  ['propmix']:
        from losses.losses import SemiLoss
        return SemiLoss()

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    elif p['backbone'] == 'InceptionResNetV2':
        return 1536

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-20', 'cifar-100', 'mini_imagenet_blue', 'mini_imagenet_red', 'mini_imagenet32_red','stanford_cars_blue']:
            from models.resnet_cifar import resnet18
            backbone = resnet18()

        # elif p['train_db_name'] == 'stl-10':
        #     from models.resnet_stl import resnet18
        #     backbone = resnet18()
        
        else:
            raise NotImplementedError
    elif p['backbone'] == 'PRN18':
        from models.preact_resnet_cifar import PreActResNet18
        backbone = PreActResNet18()
    elif p['backbone'] == 'vgg19':
        raise NotImplementedError

    elif p['backbone'] == 'densenet':
        from models.densenet_rog import densenet
        
        backbone = densenet(num_classes=p['num_classes'])
        

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()  

        else:
            raise NotImplementedError 

    elif p['backbone'] == 'InceptionResNetV2':
        if  p['train_db_name'] in ['webvision', 'imagenet_100']:
        # if  p['train_db_name'] in ["imagenet_100",]:
            from models.InceptionResNetV2 import network
            backbone = network()  

        else:
            
            raise NotImplementedError 
    elif p['backbone'] == 'resnet34':
        if p['train_db_name'] in ['cifar-10-plc', 'cifar-100-plc']:
            from models.resnet import resnet34
            backbone = resnet34()  

    else:
        import pdb; pdb.set_trace()
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['simclr', 'moco']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] in ['scan']:
        from models.models import ClusteringModel
        # if p['setup'] == 'selflabel':
        #     assert(p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'], p['setup'])
    
    elif p['setup'] in ['propmix']:
        from models.models import Model
        model = Model(backbone, p['num_classes'], p['num_heads'], p['setup'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    import pdb;pdb.set_trace()
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')
        
        if p['setup'] == 'scan': # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)

        # elif p['setup'] == 'selflabel': # Weights are supposed to be transfered from scan 
        #     # We only continue with the best head (pop all heads first, then copy back the best head)
        #     model_state = state['model']
        #     all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
        #     best_head_weight = model_state['cluster_head.%d.weight' %(state['head'])]
        #     best_head_bias = model_state['cluster_head.%d.bias' %(state['head'])]
        #     for k in all_heads:
        #         model_state.pop(k)

        #     model_state['cluster_head.0.weight'] = best_head_weight
        #     model_state['cluster_head.0.bias'] = best_head_bias
        #     missing = model.load_state_dict(model_state, strict=True)
        
        elif p['setup'] in ['propmix']: # Weights are supposed to be transfered from scan 
            # We only continue with the best head (pop all heads first, then copy back the best head)
            
            
            model_state = state['model']
            
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.weight' %(state['head'])]
            best_head_bias = model_state['cluster_head.%d.bias' %(state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['head.weight'] = best_head_weight
            model_state['head.bias'] = best_head_bias
            # model_state['sl_head.weight'] = best_head_weight
            # model_state['sl_head.bias'] = best_head_bias
            missing = model.load_state_dict(copy.deepcopy(model_state), strict=True)
            

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


def get_train_dataset(p, transform, to_augmented_dataset=False,
                        to_neighbors_dataset=False, to_noisy_dataset=False, split=None, meta_info={}):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        if p['setup'] in ['propmix']:
            from data.cifar_propmix import cifar_dataset
            dataset = cifar_dataset(dataset=p['dataset'],root_dir=p['data_path'],transform=transform, meta_info=meta_info)
            to_noisy_dataset = False
        else:
            from data.cifar import CIFAR10
            dataset = CIFAR10(root=p['data_path'] , train=True, transform=transform, download=True)
    
    elif p['train_db_name'] == 'cifar-100':
        if p['setup'] in [ 'propmix']:
            from data.cifar_propmix import cifar_dataset
            dataset = cifar_dataset(dataset=p['dataset'],root_dir=p['data_path'],transform=transform, meta_info=meta_info)
            to_noisy_dataset = False
        else:
            from data.cifar import CIFAR100
            dataset = CIFAR100(root=p['data_path'], train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='train', transform=transform)

    elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=transform)
    
    elif p['train_db_name'] == 'webvision':
        from data.webvision import Webvision
        dataset = Webvision(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'])
    elif p['train_db_name'] == 'mini_imagenet_blue':
        from data.redblue import MiniImagenet
        dataset = MiniImagenet(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'], color='blue')
    elif p['train_db_name'] in  ['mini_imagenet_red', 'mini_imagenet32_red']:
        from data.redblue import MiniImagenet
        dataset = MiniImagenet(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'], color='red')
    elif p['train_db_name'] == 'stanford_cars_blue':
        from data.redblue import StanfordCars
        dataset = StanfordCars(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'], color='blue')
    elif p['train_db_name'] == 'cifar-10-plc':
        from data.cifar_plc import CIFAR10
        dataset = CIFAR10(root=p['data_path'], split='train', train_ratio=0.9, trust_ratio=0, download=True, transform=transform)

    elif p['train_db_name'] == 'cifar-100-plc':
        from data.cifar_plc import CIFAR100
        dataset = CIFAR100(root=p['data_path'], split='train', train_ratio=0.9, trust_ratio=0, download=True, transform=transform)
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)

    if to_noisy_dataset:
        from data.custom_dataset import NoisyDataset
        dataset = NoisyDataset(dataset, meta_info)
    
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'], meta_info.get('predicted_labels'))
    
    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False, meta_info=None):
    # Base dataset
    
    if p['val_db_name'] == 'cifar-10':
        if p['setup'] in [ 'propmix']:
            from data.cifar_propmix import cifar_dataset
            dataset = cifar_dataset(dataset=p['dataset'],root_dir=p['data_path'],transform=transform, meta_info=meta_info)
        else:
            from data.cifar import CIFAR10
            dataset = CIFAR10(root=p['data_path'], train=False, transform=transform, download=True)
    
    elif p['val_db_name'] == 'cifar-100':
        if p['setup'] in ['propmix']:
            from data.cifar_propmix import cifar_dataset
            dataset = cifar_dataset(dataset=p['dataset'],root_dir=p['data_path'],transform=transform, meta_info=meta_info)
            
        else:
            from data.cifar import CIFAR100
            dataset = CIFAR100(root=p['data_path'], train=True, transform=transform, download=True)
            

    # elif p['val_db_name'] == 'stl-10':
    #     from data.stl import STL10
    #     dataset = STL10(split='test', transform=transform, download=True)
    
    # elif p['val_db_name'] == 'imagenet':
    #     from data.imagenet import ImageNet
    #     dataset = ImageNet(split='val', transform=transform)
    
    #elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
    elif str(p['val_db_name']) in ['imagenet_100']:
        from data.imagenet import ImageNetSubset
        
        # subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        #dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)
        dataset = ImageNetSubset(root_dir=p['data_path'], split='val', transform=transform)

    elif p['val_db_name'] == 'webvision':
        from data.webvision import Webvision
        dataset = Webvision(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'])
    elif p['val_db_name'] == 'mini_imagenet_blue':
        from data.redblue import MiniImagenet
        dataset = MiniImagenet(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'], color='blue')
    elif p['val_db_name'] in ['mini_imagenet_red','mini_imagenet32_red']:
        from data.redblue import MiniImagenet
        dataset = MiniImagenet(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'], color='red')
    elif p['val_db_name'] == 'stanford_cars_blue':
        from data.redblue import StanfordCars
        dataset = StanfordCars(root_dir=p['data_path'], transform=transform, meta_info=meta_info, num_classes=p['num_classes'], color='blue')
    elif p['val_db_name'] == 'cifar-10-plc':
        from data.cifar_plc import CIFAR10
        dataset = CIFAR10(root=p['data_path'], split='test', train_ratio=0.9, trust_ratio=0, download=True, transform=transform)

    elif p['val_db_name'] == 'cifar-100-plc':
        from data.cifar_plc import CIFAR100
        dataset = CIFAR100(root=p['data_path'], split='test', train_ratio=0.9, trust_ratio=0, download=True, transform=transform)
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5

    return dataset


def get_train_dataloader(p, dataset, shuffle=True, explicit_batch_size=None):
    batch_size = p['batch_size']
    if explicit_batch_size is not None:
        batch_size = explicit_batch_size
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=batch_size, pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=shuffle)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])

    #elif p['augmentation_strategy'] in  ['dividemix','dividemix_webv']:
    elif p['augmentation_strategy'] in  ['dividemix','dividemix_webv']:
        trnfs = []
        if 'resize' in p['augmentation_kwargs'].keys():
            trnfs = [transforms.Resize(p['augmentation_kwargs']['resize'])]
        try:
            padding = p['augmentation_kwargs']['crop_padding']
        except:
            padding = None
        trnfs.append(transforms.RandomCrop(p['augmentation_kwargs']['crop_size'], padding=padding))
        trnfs.append(transforms.RandomHorizontalFlip())
        trnfs.append(transforms.ToTensor())
        trnfs.append(transforms.Normalize(**p['augmentation_kwargs']['normalize']))
        return transforms.Compose(trnfs)
    elif p['augmentation_strategy'] == 'dividemix_webvision':
        trnfs = []
        if 'resize' in p['augmentation_kwargs'].keys():
            trnfs = [transforms.Resize(p['augmentation_kwargs']['resize'])]
        # try:
        #     padding = p['augmentation_kwargs']['crop_padding']
        # except:
        #     padding = None
        trnfs = [transforms.Resize(p['augmentation_kwargs']['resize'])]
        trnfs.append(transforms.RandomResizedCrop(p['augmentation_kwargs']['crop_size']))
        # trnfs.append(transforms.RandomCrop(p['augmentation_kwargs']['crop_size'], padding=padding))
        trnfs.append(transforms.RandomHorizontalFlip())
        trnfs.append(transforms.ToTensor())
        trnfs.append(transforms.Normalize(**p['augmentation_kwargs']['normalize']))
        return transforms.Compose(trnfs)
    elif p['augmentation_strategy'] == 'dividemix_red_mini_imagenet':
        trnfs = []
        
        # trnfs = [transforms.Resize(p['augmentation_kwargs']['random_resized_crop'])]
        
        trnfs.append(transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']))
        # trnfs.append(transforms.RandomCrop(p['augmentation_kwargs']['crop_size'], padding=padding))
        trnfs.append(transforms.RandomHorizontalFlip())
        trnfs.append(transforms.ToTensor())
        trnfs.append(transforms.Normalize(**p['augmentation_kwargs']['normalize']))
        return transforms.Compose(trnfs)
    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    trnfs = []
    if 'resize' in p['augmentation_kwargs'].keys():
        trnfs = [transforms.Resize(p['augmentation_kwargs']['resize'])]
    # if p['augmentation_strategy'] != 'dividemix_red_mini_imagenet':
    #     trnfs.append(transforms.CenterCrop(p['augmentation_kwargs']['crop_size']))
    #elif p['augmentation_strategy'] == 'dividemix_webv':
    if p['augmentation_strategy'] == 'dividemix_webv':
        trnfs = [transforms.Resize(256)]
        trnfs.append(transforms.CenterCrop(p['augmentation_kwargs']['crop_size']))
    trnfs.append(transforms.ToTensor())
    trnfs.append(transforms.Normalize(**p['augmentation_kwargs']['normalize']))
    return transforms.Compose(trnfs)

def get_scan_transformations(p):
    if 'scan_kwargs' not in p.keys():
        return get_train_transformations(p)
    trnfs = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(p['scan_kwargs']['crop_size']),
        Augment(p['scan_kwargs']['num_strong_augs']),
        transforms.ToTensor(),
        transforms.Normalize(**p['scan_kwargs']['normalize']),
        Cutout(
            n_holes = p['scan_kwargs']['cutout_kwargs']['n_holes'],
            length = p['scan_kwargs']['cutout_kwargs']['length'],
            random = p['scan_kwargs']['cutout_kwargs']['random'])]
    if 'resize' in p['scan_kwargs'].keys():
        trnfs.insert(0, transforms.Resize(p['scan_kwargs']['resize']))
    return transforms.Compose(trnfs)


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()
                
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
