import os
os.environ["PYTHONWARNINGS"] = "ignore"  

import warnings
def _no_warning(*args, **kwargs):
    pass
warnings.showwarning = _no_warning  



import argparse

import yaml

import torch
import torch.utils.data
import numpy as np
import random
from models.ddm import DenoisingDiffusion 

import torch.multiprocessing as mp

import dataset 

import setproctitle 
setproctitle.setproctitle("wode")  




def parse_args_and_config(): 
    parser = argparse.ArgumentParser(description='Training Latent Denoising Diffusion Models')
    
    parser.add_argument("--config", default='configs/config.yml',  type=str, required=False,
                        help="Path to the config file")
    
    
    
    parser.add_argument('--resume', default=None, type=dict,
                        help='Path for checkpoint to load and resume')
    
    parser.add_argument("--sampling_timesteps", type=int, default=50,
                        help="Number of implicit sampling steps for validation image")
    
    parser.add_argument("--image_folder", default='validation/images/', type=str,
                        help="Location to save restored validation image")
    
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training')
    
    parser.add_argument('--f', '-f', default=None, help='jupyter notebook')
    
    args = parser.parse_args()
    
    config_path = args.config if os.path.isfile(args.config) else os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):  
    namespace = argparse.Namespace()  
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value) 
    return namespace


def main(rank, world_size): 
    setup(rank, world_size)
    args, config = parse_args_and_config()
    
    device = torch.device(f'cuda:{rank}')
    print("Using device: {}".format(device))
    config.device = device  

    
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True 
    
    
    
    
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = dataset.DATASET(config) 
    print("=> dataset length:{}".format(len(DATASET.train_dataset)))
    train_sampler, train_loader,val_loader = DATASET.get_loaders() 
    print("len(train_dataset) =", len(train_loader.dataset))
    print("len(train_loader)  =", len(train_loader))
    print("len(val_dataset)  =", len(val_loader.dataset))
    print("len(val_loader)  =", len(val_loader))

    
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config, rank)
    diffusion.train(train_sampler, train_loader,val_loader)
    print('=> training done')


def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '13156')   
    torch.distributed.init_process_group(
        backend="nccl",  
        rank=rank,
        world_size=world_size
    ) 
    torch.cuda.set_device(rank)
    
    



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True) 
    
    



