import torch
import torch.nn as nn
import utils
import torchvision
import os



class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        self.diffusion.load_ddm_ckpt(ema=False, **args.resume)
        self.diffusion.model.eval()
        self.diffusion.ae.eval()
        print('load model success!')



