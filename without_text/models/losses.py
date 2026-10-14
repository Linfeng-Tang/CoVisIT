import torch.nn as nn
import kornia.filters as KF
import torch
import torch.nn.functional as F
import numpy as np


def Re_loss(img1, img2, weights=[5, 5], mask=None, ir_flag=False):
    
    grad1 =  KF.spatial_gradient(img1, order=1).abs().sum(dim=[1,2])
    grad2 = KF.spatial_gradient(img2, order=1).abs().sum(dim=[1,2])
    loss_intensity = F.l1_loss(img1, img2)
    loss_grad = F.l1_loss(grad1, grad2)
    loss_total = weights[0] * loss_intensity + weights[1] * loss_grad
    return loss_total, loss_intensity, loss_grad