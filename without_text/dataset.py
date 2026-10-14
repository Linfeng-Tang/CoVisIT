import torch
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,DistributedSampler
import torch.multiprocessing as mp
import random
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from utils.unregestration import randflow 

class DATASET:
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        if self.mode=='train':
            self.train_dataset = dataset_gen(
                self.config.data.trainpath,
                mode='train',
                raw_dir=self.config.data.train_raw_dir,
                gt_dir=self.config.data.train_gt_dir,
                guide_dir=self.config.data.train_guide_dir,
            )
            self.val_dataset   = dataset_gen(
                self.config.data.valpath,
                mode='val',
                raw_dir=self.config.data.val_raw_dir,
                gt_dir=self.config.data.val_gt_dir,
                guide_dir=self.config.data.val_guide_dir,
            ) 
        elif self.mode == 'test':
            self.test_dataset = dataset_gen(self.config.data.testpath, mode='test')
    def get_loaders(self):
        if self.mode=='train':
            train_sampler = DistributedSampler(self.train_dataset,
                                       shuffle=True,
                                      )
            train_loader = DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                                     shuffle=False, num_workers=self.config.data.num_workers,
                                     sampler=train_sampler,
                                     pin_memory=True,
                                     multiprocessing_context=mp.get_context('spawn')
                                      )
            
            val_sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,      
                drop_last=False
            )

            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,       
                shuffle=False,      
                num_workers=self.config.data.num_workers,
                sampler=val_sampler,
                pin_memory=True,
                multiprocessing_context=mp.get_context('spawn')
            )

            return train_sampler, train_loader, val_loader
        
        elif self.mode == 'test':
            test_loader = DataLoader(self.test_dataset, batch_size=1,
                                     shuffle=False, num_workers=self.config.data.num_workers,
                                     pin_memory=True
                                     )
            return test_loader
        return None


class dataset_gen(Dataset):

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        stride: int = 1,
        offset: int = 0,
        per_scene: bool = False,
        raw_dir: str = "visible",
        gt_dir: str = "infrared",
        guide_dir: str = "infrared_x4",
    ):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        
        self.guide_path = os.path.join(data_path, guide_dir)

        if mode in ("train", "val"):
            self.raw_path = os.path.join(data_path, raw_dir)
            self.gt_path  = os.path.join(data_path, gt_dir)
        else:
            self.raw_path = os.path.join(data_path, "in-raw")  

        
        filenames = [
            f for f in os.listdir(self.raw_path)
            if f.lower().endswith((".jpg", ".png"))
        ]

        
        
        filenames = sorted(filenames)

        
        stride = int(stride) if stride is not None else 1
        if stride < 1:
            stride = 1
        offset = int(offset) if offset is not None else 0
        offset = offset % stride if stride > 0 else 0

        if stride > 1:
            if per_scene:
                
                from collections import defaultdict
                groups = defaultdict(list)
                for f in filenames:
                    scene = f[:2]
                    groups[scene].append(f)

                new_list = []
                for scene in sorted(groups.keys()):
                    fs = sorted(groups[scene])
                    new_list.extend(fs[offset::stride])
                filenames = new_list
            else:
                
                filenames = filenames[offset::stride]
        

        self.filenames = filenames
        self.length = len(self.filenames)

        
        self.target_height, self.target_width = 512, 512
        self.guide_target_height, self.guide_target_width = self.target_height // 4, self.target_width // 4

        
        self.transform_params = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        name = self.filenames[index]
        img_raw = Image.open(os.path.join(self.raw_path, name)).convert("RGB")    
        img_guide = Image.open(os.path.join(self.guide_path, name)).convert("RGB")  

        if self.mode in ("train", "val"):
            img_gt = Image.open(os.path.join(self.gt_path, name)).convert("RGB")  
        else:
            img_gt = img_raw.copy()  

        if self.mode=='train':
            
            apply_flip = (random.random() < 0.5)
            rotation_degrees = self.transform_params.transforms[1].degrees
            angle = transforms.RandomRotation.get_params(rotation_degrees)

            
            if apply_flip:
                img_raw = F.hflip(img_raw)
                img_gt = F.hflip(img_gt)
                img_guide = F.hflip(img_guide)

            img_raw = F.rotate(img_raw, angle, F.InterpolationMode.BILINEAR, fill=0)
            img_gt = F.rotate(img_gt, angle, F.InterpolationMode.BILINEAR, fill=0)
            img_guide = F.rotate(img_guide, angle, F.InterpolationMode.BILINEAR, fill=0)

        
        
        w, h = img_raw.size
        if w < self.target_width or h < self.target_height:
            scale = max(self.target_width / w, self.target_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_raw = F.resize(img_raw, [new_h, new_w], F.InterpolationMode.BILINEAR)
            img_gt = F.resize(img_gt, [new_h, new_w], F.InterpolationMode.BILINEAR)

        
        w, h = img_guide.size
        if w < self.target_width or h < self.target_height:
            scale = max(self.target_width / w, self.target_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_guide = F.resize(img_guide, [new_h, new_w], F.InterpolationMode.BILINEAR)

        
        if self.mode == "train":
            i, j, th, tw = transforms.RandomCrop.get_params(
                img_raw, output_size=(self.target_height, self.target_width)
            )
            
            
            
            
            img_raw = F.crop(img_raw, i, j, th, tw)
            img_gt = F.crop(img_gt, i , j, th, tw)

            
            img_guide = F.crop(img_guide, i, j, th, tw)

            
            if img_guide.size != (self.target_width, self.target_height):
                img_guide = F.resize(img_guide, [self.target_height, self.target_width])

        
        img_raw = F.to_tensor(img_raw)
        img_gt = F.to_tensor(img_gt)
        img_guide = F.to_tensor(img_guide)



        
        
        
        
        if self.mode == "train":
            flow_guide, _, _ = randflow(img_guide.unsqueeze(0), 5, 0.05, 0.5)
            img_guide_warped = F2.grid_sample(img_guide.unsqueeze(0), flow_guide, align_corners=False, mode='bilinear').squeeze(0)
        else:
            
            img_guide_warped = img_guide
        
        
        return img_raw, img_gt, img_guide_warped, name
    
































































































        



























































    







































































































