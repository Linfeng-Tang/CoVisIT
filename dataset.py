import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F2
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils.unregestration import randflow




IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
TEXT_PROMPT_PREFIX = "Generate an infrared image. A brief description of the infrared image and the heat sources in it is as follows:"
TEXT_KEYS = ('text', 'caption', 'description', 'prompt')


def _safe_read_json_records(json_path):
    if json_path is None:
        return []
    json_path = str(json_path).strip()
    if not json_path:
        return []
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f'文本 JSON/JSONL 文件不存在: {json_path}')

    if p.suffix.lower() == '.jsonl':
        records = []
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = []
        for k, v in data.items():
            if isinstance(v, dict):
                obj = dict(v)
                obj.setdefault('key', k)
            else:
                obj = {'key': k, 'text': v}
            records.append(obj)
        return records
    raise TypeError(f'不支持的文本标注格式: {type(data)}')


def _normalize_text(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def _parse_numeric_id_from_name(name: str):
    stem = Path(name).stem
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return str(int(digits))
    except Exception:
        return None


def _candidate_keys_from_name(name: str):
    base = os.path.basename(str(name))
    stem = Path(base).stem
    keys = [base, stem]
    numeric_key = _parse_numeric_id_from_name(base)
    if numeric_key is not None:
        keys.append(numeric_key)
    return keys




def _prepend_text_prefix(text):
    text = _normalize_text(text)
    if text == '':
        return ''
    return f'{TEXT_PROMPT_PREFIX} {text}'

def _extract_text_from_record(record):
    if isinstance(record, str):
        return _normalize_text(record)
    if not isinstance(record, dict):
        return _normalize_text(record)
    for k in TEXT_KEYS:
        if k in record:
            return _normalize_text(record[k])
    return ''


def _build_text_lookup(json_path):
    records = _safe_read_json_records(json_path)
    lookup = {}
    for rec in records:
        text = _extract_text_from_record(rec)
        if text == '':
            continue

        if isinstance(rec, dict):
            candidates = []
            for field in ('filename', 'name', 'image', 'image_name', 'img_name'):
                v = rec.get(field)
                if v:
                    candidates.extend(_candidate_keys_from_name(v))
            for field in ('path', 'image_path', 'img_path'):
                v = rec.get(field)
                if v:
                    candidates.extend(_candidate_keys_from_name(os.path.basename(str(v))))
            for field in ('id', 'id_str', 'image_id'):
                v = rec.get(field)
                if v is not None:
                    candidates.append(str(v))
            id_num = rec.get('id_num')
            if id_num is not None:
                candidates.append(str(id_num))
            key = rec.get('key')
            if key is not None:
                candidates.extend(_candidate_keys_from_name(str(key)))
        else:
            candidates = []

        for key in candidates:
            if key:
                lookup[str(key)] = text
    return lookup


class DATASET:
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

        train_text_json = self.config.data.train_text_json
        val_text_json = self.config.data.val_text_json
        test_text_json = self.config.data.test_text_json

        if self.mode == 'train':
            self.train_dataset = dataset_gen(
                self.config.data.trainpath,
                mode='train',
                raw_dir=self.config.data.train_raw_dir,
                gt_dir=self.config.data.train_gt_dir,
                guide_dir=self.config.data.train_guide_dir,
                text_json_path=train_text_json,
            )
            self.val_dataset = dataset_gen(
                self.config.data.valpath,
                mode='val',
                raw_dir=self.config.data.val_raw_dir,
                gt_dir=self.config.data.val_gt_dir,
                guide_dir=self.config.data.val_guide_dir,
                text_json_path=val_text_json,
            )
        elif self.mode == 'test':
            self.test_dataset = dataset_gen(
                self.config.data.trainpath,
                mode='test',
                text_json_path=test_text_json,
            )

    def get_loaders(self):
        if self.mode == 'train':
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                sampler=train_sampler,
                pin_memory=True,
                multiprocessing_context=mp.get_context('spawn'),
            )

            val_sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,
                drop_last=False,
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                sampler=val_sampler,
                pin_memory=True,
                multiprocessing_context=mp.get_context('spawn'),
            )
            return train_sampler, train_loader, val_loader

        elif self.mode == 'test':
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=True,
            )
            return test_loader
        return None


class dataset_gen(Dataset):

    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        stride: int = 1,
        offset: int = 0,
        per_scene: bool = False,
        raw_dir: str = 'visible',
        gt_dir: str = 'infrared',
        guide_dir: str = 'infrared_x4',
        text_json_path: str = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.text_json_path = text_json_path
        self.text_lookup = _build_text_lookup(text_json_path)
        self.require_text = text_json_path is not None and str(text_json_path).strip() != ''

        self.guide_path = os.path.join(data_path, guide_dir)

        if mode in ('train', 'val'):
            self.raw_path = os.path.join(data_path, raw_dir)
            self.gt_path = os.path.join(data_path, gt_dir)
        else:
            self.raw_path = os.path.join(data_path, 'in-raw')

        filenames = [
            f for f in os.listdir(self.raw_path)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]
        filenames = sorted(filenames)

        stride = int(stride) if stride is not None else 1
        if stride < 1:
            stride = 1
        offset = int(offset) if offset is not None else 0
        offset = offset % stride if stride > 0 else 0

        if stride > 1:
            if per_scene:
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

    def _get_text(self, name: str) -> str:
        for key in _candidate_keys_from_name(name):
            if key in self.text_lookup:
                return self.text_lookup[key]
        if self.require_text:
            raise KeyError(
                f'在文本标注文件中找不到图片 {name} 对应的文本。'
                f' 当前 text_json_path={self.text_json_path}'
            )
        return ''

    def __getitem__(self, index):
        name = self.filenames[index]
        raw_text = self._get_text(name)
        text = _prepend_text_prefix(raw_text)

        img_raw = Image.open(os.path.join(self.raw_path, name)).convert('RGB')
        img_guide = Image.open(os.path.join(self.guide_path, name)).convert('RGB')

        if self.mode in ('train', 'val'):
            img_gt = Image.open(os.path.join(self.gt_path, name)).convert('RGB')
        else:
            img_gt = img_raw.copy()

        if self.mode == 'train':
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

        if self.mode == 'train':
            i, j, th, tw = transforms.RandomCrop.get_params(
                img_raw, output_size=(self.target_height, self.target_width)
            )
            img_raw = F.crop(img_raw, i, j, th, tw)
            img_gt = F.crop(img_gt, i, j, th, tw)
            img_guide = F.crop(img_guide, i, j, th, tw)
            if img_guide.size != (self.target_width, self.target_height):
                img_guide = F.resize(img_guide, [self.target_height, self.target_width])

        img_raw = F.to_tensor(img_raw)
        img_gt = F.to_tensor(img_gt)
        img_guide = F.to_tensor(img_guide)

        if self.mode == 'train':
            flow_guide, _, _ = randflow(img_guide.unsqueeze(0), 5, 0.05, 0.5)
            img_guide_warped = F2.grid_sample(
                img_guide.unsqueeze(0), flow_guide,
                align_corners=False, mode='bilinear'
            ).squeeze(0)
        else:
            img_guide_warped = img_guide

        return img_raw, img_gt, img_guide_warped, text, name
