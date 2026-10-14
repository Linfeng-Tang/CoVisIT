import os
import numpy as np
import torch
import torch.nn as nn
from pretrained import ae as pretrained
from PIL import Image
from torchvision import transforms
import utils
import sys
import models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from models.unet_module import DiffusionUNet
from utils.config import Config
import torch.nn.functional as F
from models.losses_timeweight import  Re_loss_t
from models.Sobel import sobel_magnitude_condition
from models.Class_Clip import FrozenLongCLIPEmbedder
from utils.metrics_gray import calculate_psnr, calculate_ssim
from typing import Optional
from tqdm import tqdm





def soft_histogram_scaled(x, bins=64, scale=256.0, sigma=5.0):
    B, C, H, W = x.shape
    assert C == 1, "soft_histogram 需要输入灰度图 [B,1,H,W]"

    
    x = x * scale                      
    x = x.view(B, -1, 1)               
    
    device, dtype = x.device, x.dtype

    
    centers = torch.linspace(
        0.0, scale, bins, device=device, dtype=dtype
    ).view(1, 1, bins)  

    
    weights = torch.exp(-0.5 * ((x - centers) / sigma) ** 2)  

    
    hist = weights.sum(dim=1)  

    
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)

    return hist




def histogram_loss2(pred_img, gt_img, bins=64, scale=256.0, sigma=5.0):
    pred_hist = soft_histogram_scaled(pred_img, bins=bins, scale=scale, sigma=sigma)
    gt_hist   = soft_histogram_scaled(gt_img,   bins=bins, scale=scale, sigma=sigma)

    
    loss_per_sample = torch.abs(pred_hist - gt_hist).mean(dim=1)
    return loss_per_sample.mean()


def histogram_loss(pred_img, gt_img, bins=256):
    
    pred_img = pred_img.float()
    gt_img = gt_img.float()

    batch_size = pred_img.size(0)
    losses = []

    for i in range(batch_size):
        pred_single = pred_img[i]
        gt_single = gt_img[i]
        
        
        pred_pixels = pred_single.numel()
        gt_pixels = gt_single.numel()
        
        
        
        pred_hist = torch.histc(pred_single * (bins - 1), bins=bins, min=0, max=bins - 1)
        gt_hist = torch.histc(gt_single * (bins - 1), bins=bins, min=0, max=bins - 1)

        
        pred_hist_norm = pred_hist / pred_pixels
        gt_hist_norm = gt_hist / gt_pixels
        
        
        loss = torch.abs(pred_hist_norm - gt_hist_norm).mean()
        losses.append(loss)

    
    
    return torch.stack(losses).mean()

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0) 


class EMAHelper(object):  
    
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {} 

    def register(self, module): 
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):  
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):  
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)   

    def state_dict(self):  
        return self.shadow

    def load_state_dict(self, state_dict): 
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):  
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)  
    return betas


class DenoisingDiffusion:
    def __init__(self, args, config, rank=None):
        super().__init__()
        self.args = args 
        self.config = config 
        self.device = config.device
        self.inchannel = int(config.model.in_channels)

        ucfg = Config(cfg_file='configs/unet.yaml') 
        self.model = DiffusionUNet(ucfg)
        self.model.to(self.device)
        if rank is not None: 
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False) 
        self.ae = self.load_aae()
        self.text_encoder = self.load_text_encoder()

        
        if rank is not None and dist.get_rank() == 0:
            exp_name = getattr(self.args, "exp_name", "concat_super_clip")
            self.writer = SummaryWriter(log_dir=os.path.join(self.config.training.log_dir, exp_name))
        else:
            self.writer = None


        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=self.config.diffusion.beta_schedule,
            beta_start=float(self.config.diffusion.beta_start),
            beta_end=float(self.config.diffusion.beta_end),
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]
        alphas = 1. - self.betas 
        self.alphas_cumprod = torch.cumprod(alphas, dim=0) 
        self.alphas_cumprod_pre = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod], dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def load_aae(self):
        ae = nn.Module()
        ae.encoder = pretrained.Encoder()
        ae.decoder = pretrained.Decoder()

        checkpoint = torch.load(self.config.model.ae_model, weights_only=False, map_location='cpu')

        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[len('module.'):]] = v
            else:
                cleaned_state_dict[k] = v

        ae.load_state_dict(cleaned_state_dict, strict=True)

        ae.encoder.to(self.device)
        ae.decoder.to(self.device)

        for param in ae.encoder.parameters():
            param.requires_grad = False
        for param in ae.decoder.parameters():
            param.requires_grad = False

        return ae

    def load_text_encoder(self):
        text_encoder = FrozenLongCLIPEmbedder(ckpt_path=self.config.model.longclip_model, device=str(self.device))
        text_encoder.to(self.device)
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
        return text_encoder

    @torch.no_grad()
    def encode_text_batch(self, text_batch):
        if text_batch is None:
            return None
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        elif not isinstance(text_batch, (list, tuple)):
            text_batch = list(text_batch)
        text_batch = [str(t) if t is not None else '' for t in text_batch]
        text_context = self.text_encoder.encode(text_batch)
        return text_context.detach()

    def load_ddm_ckpt(self, ema=False, ckpt_path=None, epoch=None, **para): 
        checkpoint = torch.load(os.path.join(ckpt_path,f'ckpt_{epoch}.pth'), self.device,weights_only=False)
        
        optimizer_sd = torch.load(os.path.join(ckpt_path,f'optimizer_{epoch}.pth'), self.device,weights_only=False)
        
        self.start_epoch = optimizer_sd['epoch']
        self.step = optimizer_sd['step']
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(optimizer_sd['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint (epoch {}, step {})".format(optimizer_sd['epoch'], self.step))

    def train(self, train_sampler, train_loader,val_loader=None):
    
    
        if self.args.resume:
            self.load_ddm_ckpt(**self.args.resume) 
        
        
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            train_sampler.set_epoch(epoch)
            
            
            for i, (raw, gt, guide_img, text, name) in enumerate(train_loader):
                
                gt = gt.to(self.device)
                raw = raw.to(self.device)
                guide_img = guide_img.to(self.device) 

                
                gt_z = self.ae.encoder(gt)
                raw_z = self.ae.encoder(raw)
                
                
                
                
                
                guide_img_up = guide_img 
                guide_z = self.ae.encoder(guide_img_up) 

                
                
                gt_z = data_transform(gt_z)
                raw_z = data_transform(raw_z)
                guide_z = data_transform(guide_z) 

                text_context = self.encode_text_batch(text)

                self.model.train()  
                self.step += 1

                
                loss, loss1,loss2,loss3,re_int,re_grad,x0_pred_max,x0_pred_min = self.noise_estimation_loss(raw_z, gt_z, guide_z, gt, text_context=text_context) 

                
                if dist.get_rank() == 0 and self.step % 10 == 0  :
                    sys.stdout.write(f"\r[epoch: {epoch+1}/{self.config.training.n_epochs}] "
                                    f"[step: {self.step}] [loss: {loss.item():.4f}]=[loss1: {loss1.item():.4f}] + [loss2: {(20*(loss2.item())):.4f}] + [re_int:{(1000.0*(re_int.item())):.4f}]+ [re_grad:{(1000.0*(re_grad.item())):.4f}]"
                                    f"[x0_pred_max: {x0_pred_max.item():.4f}] [x0_pred_min: {x0_pred_min.item():.4f}]"
                                    f"[scale: {raw.shape[2], raw.shape[3]}]"
                    )
                    self.writer.add_scalar('loss', loss.item(), self.step)
                    self.writer.add_scalar('loss2', loss2.item(), self.step)
                    self.writer.add_scalar('loss3', loss3.item(), self.step) 
                    self.writer.add_scalar('re_int', 1000.0*re_int.item(), self.step)
                    self.writer.add_scalar('re_grad', 1000.0*re_grad.item(), self.step)
                    self.writer.add_scalar('x0_pred_max', x0_pred_max.item(), self.step)
                    self.writer.add_scalar('x0_pred_min', x0_pred_min.item(), self.step)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)

                if dist.get_rank() == 0:
                
                    if self.step % self.config.training.snapshot_freq == 0  or self.step == 1:
                        try:
                            os.makedirs(self.config.training.path, exist_ok=True) 
                            state_dict = self.model.module.state_dict()
                            for key, param in state_dict.items():
                                state_dict[key] = param.cpu()
                            torch.save({'state_dict': state_dict,
                                    'ema_helper': self.ema_helper.state_dict()
                                        }, os.path.join(self.config.training.path, f'ckpt_{epoch}.pth'))
                            torch.save({
                                'epoch': epoch + 1,
                                'step': self.step,
                                'optimizer': self.optimizer.state_dict(),
                                'params': self.args,
                                'config': self.config
                            }, os.path.join(self.config.training.path, f'optimizer_{epoch}.pth'))
                            print('==> model saved success')
                        except Exception as e:
                            print(f'==> model saving failed (Error: {str(e)})')
                
                eval_freq = getattr(self.args, "eval_freq", None)
                if eval_freq is None:
                    eval_freq = getattr(self.config.training, "eval_freq", 0)

                
                
                
                
                
                
                
                
                
                

                
                
                
                if eval_freq and (self.step % int(eval_freq) == 0) and (val_loader is not None) :

                    
                    
                    if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
                        val_loader.sampler.set_epoch(self.step)

                    m = self.run_quicktest_metrics(
                        val_loader,
                        sampling_timesteps=getattr(self.args, "sampling_timesteps", None),
                    )

                    
                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        psnr = m["psnr"]
                        ssim = m["ssim"]
                        n    = m["count"]
                        if self.writer is not None:
                            self.writer.add_scalar("val/psnr", m["psnr"], self.step)
                            self.writer.add_scalar("val/ssim", m["ssim"], self.step)
                            print(f"\n[VAL @ step {self.step}] PSNR={m['psnr']:.4f} SSIM={m['ssim']:.4f} N={m['count']}")
                        
                        if psnr > 28.5:
                            try:
                                os.makedirs(self.config.training.path, exist_ok=True)  
                                state_dict = self.model.module.state_dict() 
                                for key, param in state_dict.items():
                                    state_dict[key] = param.cpu()
                                
                                torch.save({'state_dict': state_dict,
                                            'ema_helper': self.ema_helper.state_dict()}, 
                                           os.path.join(self.config.training.path, f'ckpt_psnr_{psnr:.4f}_step{self.step}.pth'))
                                torch.save({
                                    'epoch': epoch + 1,
                                    'step': self.step,
                                    'optimizer': self.optimizer.state_dict(),
                                    'params': self.args,
                                    'config': self.config
                                }, os.path.join(self.config.training.path, f'optimizer_psnr_{psnr:.4f}_step{self.step}.pth'))
                                print(f"Model saved at PSNR {psnr:.4f}, step {self.step}")
                            except Exception as e:
                                print(f"Error saving model: {e}")
                        else:
                            print("Model not saved because PSNR is not enough.")

                    
                    if dist.is_initialized():
                        dist.barrier()
                        

    def noise_estimation_loss(self, cond, x, guide_z, gt, text_context=None): 
        b = cond.shape[0]  
        t = torch.randint(low=0, high=self.num_timesteps, size=(b // 2 + 1,)).to(self.device)
        
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:b]
        
        
        
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
        
        
        
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x).to(self.device)
        
        x_noise = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        
        output= self.model(x=x_noise, t=t.float(), cond=cond, guide=guide_z, text_context=text_context)
        

        
        loss1 = (noise - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        
        
        
        at = self.alphas_cumprod_pre.index_select(0, t.long() + 1).view(-1, 1, 1, 1)
        x0_pred = (x_noise - output * (1 - at).sqrt()) / at.sqrt()

        
        
        x0_pred_max = x0_pred.detach().max()
        x0_pred_min = x0_pred.detach().min()
        
        pred_pixel_img = self.ae.decoder(inverse_data_transform(x0_pred))
        

        
        
        pred_grayscale_img = pred_pixel_img.mean(dim=1, keepdim=True)
        gt = gt.detach().mean(dim=1, keepdim=True)

        
        hist_loss = histogram_loss2(pred_grayscale_img, gt)
        
        
        re_total, re_int, re_grad = Re_loss_t(
                    pred_grayscale_img, gt,
                    t=t, num_timesteps=self.num_timesteps,
                    weights=(1.0, 1.0),   
                    intensity='l1'
                    )

        
        loss2 = hist_loss 
        loss3 = re_total 
        
        
        
        
        lambda_hist = 10 
        lambda_l1 = 1000
        loss = loss1 + lambda_hist * hist_loss+lambda_l1*re_total

        
        return loss, loss1, loss2,loss3,re_int,re_grad,x0_pred_max,x0_pred_min


    def sample_image(self, x_cond, x_guide, x, text_context=None, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        
        
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(
                x, x_cond, seq, self.model, self.betas, eta=0.,
                corners=patch_locs, p_size=patch_size,
                guide=x_guide, text_context=text_context)
        else:
            xs = utils.sampling.generalized_steps(
                x, x_cond, seq, self.model, self.betas, eta=0.,
                guide=x_guide, text_context=text_context)
                
        if last:
            xs = xs[0][-1]
        return xs

    @torch.no_grad()
    def run_quicktest_metrics(
        self,
        val_loader,
        sampling_timesteps: Optional[int] = None,
        max_batches: Optional[int] = None,   
        use_gray: bool = True,
        save_dir: Optional[str] = None,
    ):
        if save_dir is None:
            save_dir = self.config.training.val_save_dir
        if val_loader is None:
            return {"psnr": float("nan"), "ssim": float("nan"), "count": 0}

        
        
        
        old_sampling = getattr(self.args, "sampling_timesteps", None)
        if sampling_timesteps is not None:
            self.args.sampling_timesteps = sampling_timesteps

        
        
        
        was_training = self.model.training
        self.model.eval()

        
        psnr_sum = 0.0
        ssim_sum = 0.0
        n = 0

        
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        try:
            for bidx, (raw, gt, guide_img, text, name) in enumerate(val_loader):
                if max_batches is not None and bidx >= max_batches:
                    break

                raw = raw.to(self.device, non_blocking=True)
                gt = gt.to(self.device, non_blocking=True)
                guide_img = guide_img.to(self.device, non_blocking=True)

                
                
                
                raw_z = data_transform(self.ae.encoder(raw))
                guide_z = data_transform(self.ae.encoder(guide_img))
                text_context = self.encode_text_batch(text)

                xt = torch.randn_like(raw_z)
                out_latent = self.sample_image(
                    x_cond=raw_z,
                    x_guide=guide_z,
                    x=xt,
                    text_context=text_context,
                    last=True,
                )
                out_latent = inverse_data_transform(out_latent)  
                pred = self.ae.decoder(out_latent)              

                
                
                
                
                if bidx %2== 0 and rank == 0:
                    os.makedirs(save_dir, exist_ok=True)

                    base_src = name[0] if isinstance(name, (list, tuple)) else name
                    base = os.path.splitext(os.path.basename(str(base_src)))[0]

                    
                    pred_gray = pred.mean(dim=1, keepdim=True)
                    pred0 = pred_gray[0, 0].detach().float().clamp(0, 1).cpu()

                    arr = (pred0.numpy() * 255.0 + 0.5).astype("uint8")
                    Image.fromarray(arr, mode="L").save(
                        os.path.join(save_dir, f"step{self.step}_pred_{base}.png")
                    )

                
                
                
                
                if use_gray:
                    pred_eval = pred.mean(dim=1, keepdim=True)  
                    gt_eval = gt.mean(dim=1, keepdim=True)
                else:
                    pred_eval = pred
                    gt_eval = gt

                pred_eval = pred_eval.float().clamp(0, 1)
                gt_eval = gt_eval.float().clamp(0, 1)

                
                
                bs = pred_eval.shape[0]
                for i in range(bs):
                    psnr_val = calculate_psnr(
                        pred_eval[i:i+1], gt_eval[i:i+1],
                        crop_border=0, test_y_channel=False, input_range="auto"
                    )
                    ssim_val = calculate_ssim(
                        pred_eval[i:i+1], gt_eval[i:i+1],
                        crop_border=0, test_y_channel=False, input_range="auto"
                    )
                    psnr_sum += float(psnr_val)
                    ssim_sum += float(ssim_val)
                    n += 1

            
            
            
            
            if is_dist:
                psnr_sum_t = torch.tensor(psnr_sum, device=self.device, dtype=torch.float64)
                ssim_sum_t = torch.tensor(ssim_sum, device=self.device, dtype=torch.float64)
                n_t        = torch.tensor(n,        device=self.device, dtype=torch.float64)

                dist.all_reduce(psnr_sum_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(ssim_sum_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_t,        op=dist.ReduceOp.SUM)

                psnr_sum = psnr_sum_t.item()
                ssim_sum = ssim_sum_t.item()
                n = int(n_t.item())

            if n == 0:
                return {"psnr": float("nan"), "ssim": float("nan"), "count": 0}

            return {"psnr": psnr_sum / n, "ssim": ssim_sum / n, "count": n}

        finally:
            
            if old_sampling is not None:
                self.args.sampling_timesteps = old_sampling
            
            if was_training:
                self.model.train()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    

    
    

    
    
    
    
    
    

    
    
    

    
    
    

    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
                        
    
    
    
    
    

    
    

    
    

    
    
    
    
    
    
    
    

    
    
    
    
    

    
    
    

    
    

    

    
    
    
    
    
    
    
    





