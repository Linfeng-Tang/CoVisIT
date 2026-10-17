import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
from tqdm import tqdm
import torchvision.transforms.functional as F


def compute_alpha(beta):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, betas, eta=0.0, guide=None, text_context=None): 
    
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        alpha_cumprod_pre = compute_alpha(betas)
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = alpha_cumprod_pre.index_select(0, t.long()+1)
            at_next = alpha_cumprod_pre.index_select(0, next_t.long()+1)
            xt = xs[-1].to(x.device)  
            et= model(x=xt, t=t, cond=x_cond, guide=guide, text_context=text_context) 
            
            
            
            
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  
            x0_preds.append(x0_t) 

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            
            xs.append(xt_next)
    return xs, x0_preds


def sample_sr3(x, x_cond, seq, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_height=None, p_width=None, manual_batching=False, guide=None, text_context=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_height, wi:wi + p_width] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et_output = torch.zeros_like(x_cond, device=x.device)
            
            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_height, p_width) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_height, p_width)) for (hi, wi) in corners], dim=0)
                for i in range(0, len(corners), manual_batching_size):
                    outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                               xt_patch[i:i+manual_batching_size]], dim=1), t, guide=guide, text_context=text_context)
                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                        et_output[0, :, hi:hi + p_height, wi:wi + p_width] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_height, p_width)
                    x_cond_patch = crop(x_cond, hi, wi, p_height, p_width)
                    x_cond_patch = data_transform(x_cond_patch)
                    et_output[:, :, hi:hi + p_height, wi:wi + p_width] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t, guide=guide, text_context=text_context)

            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds
