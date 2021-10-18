import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

def create_config(config_file_env, config_file_exp, meta_info=None):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    # if cfg['setup'] in ['pretext', 'scan','simclr']:
    #     noise_specific = 'r={}_{}'.format(meta_info['r'], meta_info['noise_mode'])
    # else:
    if cfg['setup'] in ['propmix']:
        noise_specific = 'r={}_{}'.format(meta_info['r'], meta_info['noise_mode'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    
    if 'pretext_path' not in cfg:   #if this key does not exist
        # pretext_dir = os.path.join(pretext_dir, noise_specific)
        if cfg['setup']=='simclr':
            #pretext_folder = 'r={}_{}_{}ep_bs{}_{}'.format(meta_info['r'], meta_info['noise_mode'],  cfg['epochs'],cfg['batch_size'], cfg['backbone'])
            pretext_folder = '{}_{}ep_bs{}'.format(cfg['backbone'],cfg['epochs'],cfg['batch_size'])
            pretext_dir = os.path.join(pretext_dir, pretext_folder)
        else:
            pretext_dir = os.path.join(pretext_dir, noise_specific)
    else:
        pretext_dir = os.path.join(pretext_dir, cfg['pretext_path'])
    
    if cfg['setup'] in ['pretext','simclr']:
        mkdir_if_missing(base_dir)
        mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')


    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel',  'propmix']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        
        if 'scan_path' not in cfg:
            if 'pretext_path' in cfg:
                scan_dir = os.path.join(scan_dir, cfg['pretext_path'])
            else:
                scan_dir = os.path.join(scan_dir, noise_specific)
        else:
            scan_dir = os.path.join(scan_dir, cfg['scan_path'])
            
        # selflabel_dir = os.path.join(base_dir, 'selflabel') 
        
        
        # if cfg['setup']=='dividemix':
        #     if meta_info['nopretrain']==True:
        #         dividemix_dir = os.path.join(base_dir, 'nopretrain_dividemix')
        #         dividemix_dir = os.path.join(dividemix_dir, cfg['backbone'])
        #     else:
        #         dividemix_dir = os.path.join(base_dir, 'scan+dividemix')
        #         dividemix_dir = os.path.join(dividemix_dir, cfg['scan_path'])
                
            
        #     if meta_info['strong_aug']==True:
        #         dividemix_dir = os.path.join(dividemix_dir, noise_specific+'_sa')
        #     else:
        #         dividemix_dir = os.path.join(dividemix_dir, noise_specific)
        #     mkdir_if_missing(base_dir)
        #     mkdir_if_missing(dividemix_dir)
        #     cfg['dividemix_dir'] = dividemix_dir
        if cfg['setup']=='propmix':
            propmix_dir = os.path.join(base_dir, 'propmix')
            if 'scan_path' in cfg:
                propmix_dir = os.path.join(propmix_dir, cfg['scan_path'])
            
            noise_specific = noise_specific + '_t1_%.1f_t2_%.1f'%(cfg['p_threshold'], cfg['p_threshold2'])
            # setup_name = 'r={}_{}_{}ep_bs{}_{}'.format(meta_info['r'], meta_info['noise_mode'], cfg['epochs'],cfg['batch_size'], cfg['backbone'])
            if meta_info['strong_aug']==True:
                propmix_dir = os.path.join(propmix_dir, noise_specific+'_sa')
            else:
                propmix_dir = os.path.join(propmix_dir, noise_specific)
           
            mkdir_if_missing(base_dir)
            # mkdir_if_missing(propmix_dir)
            cfg['propmix_dir'] = propmix_dir
            
        
        # scanmix_dir = os.path.join(base_dir, 'scanmix')
        # if cfg['setup']=='scanmix':
        #     scanmix_dir = os.path.join(scanmix_dir, noise_specific+'_lu%d'%(meta_info['lambda_u']))
        # else:
        #     scanmix_dir = os.path.join(scanmix_dir, noise_specific)
        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        # mkdir_if_missing(selflabel_dir)
        cfg['base_dir'] = os.path.join(base_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        # cfg['selflabel_dir'] = selflabel_dir
        # cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        # cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')
        # cfg['scanmix_dir'] = os.path.join(scanmix_dir)
        # cfg['scanmix_checkpoint'] = os.path.join(scanmix_dir, 'checkpoint.pth.tar')
        # cfg['scanmix_model'] = os.path.join(scanmix_dir, 'model.pth.tar')
        


    return cfg 
