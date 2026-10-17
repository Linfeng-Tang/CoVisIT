import os
import sys
import yaml
import argparse
import random
import numpy as np
from typing import List, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF
from PIL import Image

from dataset_test import TEST_DATALOADER
from models.ddm import DenoisingDiffusion, data_transform, inverse_data_transform
from utils.metrics_gray import calculate_psnr, calculate_ssim



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



def _resolve_config_path(config_arg: str) -> str:
    config_arg = str(config_arg).strip()
    if os.path.isfile(config_arg):
        return config_arg
    candidate = os.path.join('configs', config_arg)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f'找不到配置文件: {config_arg}')



def parse_args_and_config():
    parser = argparse.ArgumentParser(description='DDP test script for validation/test set')
    parser.add_argument('--config', default='configs/config.yml', type=str, help='configs 下的配置文件名')
    parser.add_argument('--ckpt_path', default='checkpoints/model.pth', type=str, help='模型权重路径，例如 checkpoints_xxx/ckpt_xxx.pth')
    parser.add_argument('--use_ema', action='store_true', help='若 checkpoint 中有 ema_helper，则加载 EMA 权重到模型')
    parser.add_argument('--sampling_timesteps', type=int, default=50, help='DDIM 采样步数')

    parser.add_argument('--test_root', type=str, default=None, help='测试集根目录')
    parser.add_argument('--raw_dir', type=str, default=None, help='RGB 文件夹名')
    parser.add_argument('--gt_dir', type=str, default=None, help='GT 红外文件夹名')
    parser.add_argument('--guide_dir', type=str, default=None, help='模糊未配准红外文件夹名')

    parser.add_argument('--save_dir', type=str, default='test_results', help='输出目录')
    parser.add_argument('--metrics_txt', type=str, default='metrics.txt', help='输出的 txt 文件名')
    parser.add_argument('--batch_size', type=int, default=10, help='测试 batch size，支持大于 1')
    parser.add_argument('--save_image_every', type=int, default=1, help='每隔多少张测试图片保存一次生成图，<=0 表示不保存')
    parser.add_argument('--save_image_dir', type=str, default='generated_images', help='保存生成图的子目录名')
    parser.add_argument('--seed', default=61, type=int)
    parser.add_argument('--f', '-f', default=None, help='jupyter notebook compatibility')
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    if args.test_root is None:
        args.test_root = config.data.testpath
    if args.raw_dir is None:
        args.raw_dir = config.data.test_raw_dir
    if args.gt_dir is None:
        args.gt_dir = config.data.test_gt_dir
    if args.guide_dir is None:
        args.guide_dir = config.data.test_guide_dir
    args.config = config_path
    return args, config



def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '23118')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)



def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()



def load_test_checkpoint(diffusion: DenoisingDiffusion, ckpt_path: str, use_ema: bool = False):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    if isinstance(diffusion.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        diffusion.model.module.load_state_dict(state_dict, strict=True)
    else:
        diffusion.model.load_state_dict(state_dict, strict=True)

    if use_ema and isinstance(checkpoint, dict) and 'ema_helper' in checkpoint:
        diffusion.ema_helper.load_state_dict(checkpoint['ema_helper'])
        diffusion.ema_helper.ema(diffusion.model)



def _to_gray_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1, keepdim=True)



def _save_prediction_image(pred_tensor: torch.Tensor, save_path: str):
    pred0 = pred_tensor.detach().float().clamp(0, 1).cpu().squeeze(0)
    arr = (pred0.numpy() * 255.0 + 0.5).astype("uint8")
    Image.fromarray(arr, mode="L").save(save_path)



@torch.no_grad()
def run_test(rank, world_size, args, config):
    setup(rank, world_size)
    try:
        device = torch.device(f'cuda:{rank}')
        config.device = device

        seed = args.seed + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = True

        diffusion = DenoisingDiffusion(args, config, rank)
        load_test_checkpoint(diffusion, args.ckpt_path, use_ema=args.use_ema)
        diffusion.model.eval()
        diffusion.ae.eval()

        dataset_builder = TEST_DATALOADER(
            config=config,
            test_root=args.test_root,
            raw_dir=args.raw_dir,
            gt_dir=args.gt_dir,
            guide_dir=args.guide_dir,
            batch_size=args.batch_size,
        )
        test_sampler, test_loader = dataset_builder.get_loader(distributed=True)
        if hasattr(test_sampler, 'set_epoch'):
            test_sampler.set_epoch(0)

        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_image_every > 0:
            image_dir = os.path.join(args.save_dir, args.save_image_dir, f'rank{rank}')
            os.makedirs(image_dir, exist_ok=True)
        else:
            image_dir = None

        part_file = os.path.join(args.save_dir, f'metrics_rank{rank}.txt')
        local_results: List[Dict] = []
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0

        for raw, gt, guide_img, name in test_loader:
            raw = raw.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            guide_img = guide_img.to(device, non_blocking=True)

            raw_z = data_transform(diffusion.ae.encoder(raw))
            guide_z = data_transform(diffusion.ae.encoder(guide_img))
            xt = torch.randn_like(raw_z)

            out_latent = diffusion.sample_image(
                x_cond=raw_z,
                x_guide=guide_z,
                x=xt,
                last=True,
            )
            out_latent = inverse_data_transform(out_latent)
            pred = diffusion.ae.decoder(out_latent)

            pred_eval = _to_gray_tensor(pred).float().clamp(0, 1)
            gt_eval = _to_gray_tensor(gt).float().clamp(0, 1)

            if isinstance(name, (list, tuple)):
                batch_names = list(name)
            else:
                batch_names = [name]

            batch_size = pred_eval.shape[0]
            for i in range(batch_size):
                pred_i = pred_eval[i:i + 1]
                gt_i = gt_eval[i:i + 1]
                img_name = str(batch_names[i])

                psnr_val = float(calculate_psnr(pred_i, gt_i, crop_border=0, test_y_channel=False, input_range='auto'))
                ssim_val = float(calculate_ssim(pred_i, gt_i, crop_border=0, test_y_channel=False, input_range='auto'))

                count += 1
                local_results.append({
                    'name': img_name,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                })
                psnr_sum += psnr_val
                ssim_sum += ssim_val

                if image_dir is not None and count % args.save_image_every == 0:
                    save_name = os.path.splitext(os.path.basename(img_name))[0] + '.png'
                    save_path = os.path.join(image_dir, save_name)
                    _save_prediction_image(pred_i[0], save_path)

        with open(part_file, 'w', encoding='utf-8') as f:
            for item in local_results:
                f.write(f"{item['name']}\tPSNR: {item['psnr']:.6f}\tSSIM: {item['ssim']:.6f}\n")
            if count > 0:
                f.write(f'LOCAL_AVG\tPSNR: {psnr_sum / count:.6f}\tSSIM: {ssim_sum / count:.6f}\tCOUNT: {count}\n')
            else:
                f.write('LOCAL_AVG\tPSNR: nan\tSSIM: nan\tCOUNT: 0\n')

        dist.barrier()

        gather_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_results, gather_list, dst=0)

        total_psnr_t = torch.tensor(psnr_sum, device=device, dtype=torch.float64)
        total_ssim_t = torch.tensor(ssim_sum, device=device, dtype=torch.float64)
        total_count_t = torch.tensor(count, device=device, dtype=torch.float64)
        dist.reduce(total_psnr_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_ssim_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_count_t, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            merged: List[Dict] = []
            for part in gather_list:
                if part is not None:
                    merged.extend(part)
            merged.sort(key=lambda x: x['name'])

            total_count = int(total_count_t.item())
            avg_psnr = total_psnr_t.item() / total_count if total_count > 0 else float('nan')
            avg_ssim = total_ssim_t.item() / total_count if total_count > 0 else float('nan')

            final_txt = os.path.join(args.save_dir, args.metrics_txt)
            with open(final_txt, 'w', encoding='utf-8') as f:
                f.write(f'# ckpt_path: {args.ckpt_path}\n')
                f.write(f'# config: {args.config}\n')
                f.write(f'# test_root: {args.test_root}\n')
                f.write(f'# raw_dir: {args.raw_dir}\n')
                f.write(f'# gt_dir: {args.gt_dir}\n')
                f.write(f'# guide_dir: {args.guide_dir}\n')
                f.write(f'# sampling_timesteps: {args.sampling_timesteps}\n')
                f.write(f'# batch_size: {args.batch_size}\n')
                f.write(f'# save_image_every: {args.save_image_every}\n\n')
                for item in merged:
                    f.write(f"{item['name']}\tPSNR: {item['psnr']:.6f}\tSSIM: {item['ssim']:.6f}\n")
                f.write('\n')
                f.write(f'AVG_PSNR: {avg_psnr:.6f}\n')
                f.write(f'AVG_SSIM: {avg_ssim:.6f}\n')
                f.write(f'TOTAL_COUNT: {total_count}\n')

            print(f'Test finished. AVG_PSNR={avg_psnr:.6f}, AVG_SSIM={avg_ssim:.6f}, TOTAL_COUNT={total_count}')
            print(f'Metrics txt saved to: {final_txt}')

        dist.barrier()

    finally:
        cleanup()



def main():
    args, config = parse_args_and_config()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError('没有检测到可用 GPU。')
    mp.spawn(run_test, args=(world_size, args, config), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
