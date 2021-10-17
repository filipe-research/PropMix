import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1, setup=None):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        self.setup = setup

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]
            if self.setup == 'dividemix':
                assert(self.nheads == 1)
                out = out[0]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]
            if self.setup == 'dividemix':
                assert(self.nheads == 1)
                out = out[0]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
            if self.setup == 'dividemix':
                assert(self.nheads == 1)
                out['output'] = output[0]
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out

class ScanMixModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=2, setup=None):
        super(ScanMixModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        self.setup = setup
        assert(isinstance(self.nheads, int))
        self.dm_head = nn.Linear(self.backbone_dim, nclusters)
        self.sl_head = nn.Linear(self.backbone_dim, nclusters)

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out_dm = self.dm_head(features)
            out_sl = self.sl_head(features)
            return out_dm, out_sl

        elif forward_pass == 'backbone':
            out = self.backbone(x)
            return out

        elif forward_pass == 'head':
            out_dm = self.dm_head(x)
            out_sl = self.sl_head(x)
            return out_dm, out_sl

        elif forward_pass == 'dm':
            features = self.backbone(x)
            out = self.dm_head(features)
            return out

        elif forward_pass == 'sl':
            features = self.backbone(x)
            out = self.sl_head(features)
            return out

        elif forward_pass == 'dm_head':
            out = self.dm_head(x)
            return out

        elif forward_pass == 'sl_head':
            out = self.sl_head(x)
            return out
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))      

class DivideMixModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1, setup=None):
        super(DivideMixModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        self.setup = setup
        assert(isinstance(self.nheads, int))
        self.dm_head = nn.Linear(self.backbone_dim, nclusters)
        # self.sl_head = nn.Linear(self.backbone_dim, nclusters)
        

    def forward(self, x, forward_pass='default'):
        # if forward_pass == 'default':
        #     features = self.backbone(x)
        #     out_dm = self.dm_head(features)
        #     # out_sl = self.sl_head(features)
        #     return out_dm, out_sl

        if forward_pass == 'backbone':
            out = self.backbone(x)
            return out

        # elif forward_pass == 'head':
        #     out_dm = self.dm_head(x)
        #     # out_sl = self.sl_head(x)
        #     return out_dm

        elif forward_pass == 'dm':
            features = self.backbone(x)
            out = self.dm_head(features)
            return out

        # elif forward_pass == 'sl':
        #     features = self.backbone(x)
        #     out = self.sl_head(features)
        #     return out

        elif forward_pass == 'dm_head':
            out = self.dm_head(x)
            return out

        # elif forward_pass == 'sl_head':
        #     out = self.sl_head(x)
        #     return out
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))   