import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

TEXT_PROMPT_PREFIX = (
    'Generate an infrared image. A brief description of the infrared image '
    'and the heat sources in it is as follows:'
)
TEXT_KEYS = (
    'text', 'caption', 'description', 'prompt',
    'long_caption', 'ir_caption', 'thermal_description',
    'answer', 'content', 'gpt_text'
)


def _list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'目录不存在: {folder}')
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)])


def _safe_read_json_records(json_path: Optional[str]) -> List[Any]:
    if json_path is None:
        return []
    json_path = str(json_path).strip()
    if json_path == '':
        return []

    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f'文本 JSON/JSONL 文件不存在: {json_path}')

    if p.suffix.lower() == '.jsonl':
        records = []
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def _parse_numeric_id_from_name(name: str) -> Optional[str]:
    stem = Path(str(name)).stem
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return str(int(digits))
    except Exception:
        return None


def _candidate_keys_from_name(name: str) -> List[str]:
    base = os.path.basename(str(name))
    stem = Path(base).stem
    keys = [base, stem]
    numeric_key = _parse_numeric_id_from_name(base)
    if numeric_key is not None:
        keys.append(numeric_key)
    
    out = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _prepend_text_prefix(text: str) -> str:
    text = _normalize_text(text)
    if text == '':
        return ''
    return f'{TEXT_PROMPT_PREFIX} {text}'


def _extract_text_from_record(record: Any) -> str:
    if isinstance(record, str):
        return _normalize_text(record)
    if not isinstance(record, dict):
        return _normalize_text(record)

    for k in TEXT_KEYS:
        if k in record:
            return _normalize_text(record[k])
    return ''


def _build_text_lookup(json_path: Optional[str]) -> Dict[str, str]:
    records = _safe_read_json_records(json_path)
    lookup: Dict[str, str] = {}

    for idx, rec in enumerate(records):
        text = _extract_text_from_record(rec)
        if text == '':
            continue

        candidates: List[str] = []
        if isinstance(rec, dict):
            for field in ('filename', 'name', 'image', 'image_name', 'img_name'):
                v = rec.get(field)
                if v:
                    candidates.extend(_candidate_keys_from_name(v))

            for field in ('path', 'image_path', 'img_path'):
                v = rec.get(field)
                if v:
                    candidates.extend(_candidate_keys_from_name(os.path.basename(str(v))))

            for field in ('id', 'id_str', 'image_id', 'id_num'):
                v = rec.get(field)
                if v is not None:
                    candidates.append(str(v))

            key = rec.get('key')
            if key is not None:
                candidates.extend(_candidate_keys_from_name(str(key)))
        else:
            candidates.append(str(idx))

        for key in candidates:
            if key:
                lookup[str(key)] = text

    return lookup


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.indices = list(range(len(dataset)))[self.rank::self.num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TestDataset(Dataset):
    def __init__(
        self,
        test_root: str,
        raw_dir: str = 'visible_clip',
        gt_dir: str = 'infrared_clip',
        guide_dir: str = 'infrared_clip_blur_unreg',
        text_json_path: Optional[str] = None,
        require_text: bool = True,
    ):
        super().__init__()
        self.test_root = test_root
        self.raw_path = os.path.join(test_root, raw_dir)
        self.gt_path = os.path.join(test_root, gt_dir)
        self.guide_path = os.path.join(test_root, guide_dir)
        self.text_json_path = text_json_path
        self.require_text = bool(require_text) and text_json_path is not None and str(text_json_path).strip() != ''
        self.text_lookup = _build_text_lookup(text_json_path)

        for p in [self.raw_path, self.gt_path, self.guide_path]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f'测试集目录不存在: {p}')

        raw_files = _list_image_files(self.raw_path)
        gt_files = set(_list_image_files(self.gt_path))
        guide_files = set(_list_image_files(self.guide_path))

        missing_gt = [f for f in raw_files if f not in gt_files]
        missing_guide = [f for f in raw_files if f not in guide_files]
        if missing_gt:
            raise FileNotFoundError(f'以下文件在 GT 目录中缺失，示例前 10 个: {missing_gt[:10]}')
        if missing_guide:
            raise FileNotFoundError(f'以下文件在 guide 目录中缺失，示例前 10 个: {missing_guide[:10]}')

        self.filenames = raw_files

    def __len__(self) -> int:
        return len(self.filenames)

    def _get_text(self, name: str) -> str:
        for key in _candidate_keys_from_name(name):
            if key in self.text_lookup:
                return self.text_lookup[key]

        if self.require_text:
            raise KeyError(
                f'在测试文本标注文件中找不到图片 {name} 对应的文本。'
                f' 当前 text_json_path={self.text_json_path}'
            )
        return ''

    def __getitem__(self, index: int):
        name = self.filenames[index]
        raw_text = self._get_text(name)
        text = _prepend_text_prefix(raw_text)

        raw = Image.open(os.path.join(self.raw_path, name)).convert('RGB')
        gt = Image.open(os.path.join(self.gt_path, name)).convert('RGB')
        guide = Image.open(os.path.join(self.guide_path, name)).convert('RGB')

        raw = TF.to_tensor(raw)
        gt = TF.to_tensor(gt)
        guide = TF.to_tensor(guide)

        return raw, gt, guide, text, name


class TEST_DATALOADER:
    def __init__(
        self,
        config,
        test_root: str,
        raw_dir: str = 'visible_clip',
        gt_dir: str = 'infrared_clip',
        guide_dir: str = 'infrared_clip_blur_unreg',
        batch_size: int = 1,
        text_json_path: Optional[str] = None,
        require_text: bool = True,
    ):
        self.config = config
        self.batch_size = int(batch_size)

        if text_json_path is None:
            text_json_path = getattr(getattr(config, 'data', None), 'test_text_json', None)

        self.test_dataset = TestDataset(
            test_root=test_root,
            raw_dir=raw_dir,
            gt_dir=gt_dir,
            guide_dir=guide_dir,
            text_json_path=text_json_path,
            require_text=require_text,
        )

    def get_loader(self, distributed: bool = True, rank: int = 0, world_size: int = 1):
        if distributed and world_size > 1:
            sampler = DistributedEvalSampler(self.test_dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = None

        num_workers = int(getattr(self.config.data, 'num_workers', 0))
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            multiprocessing_context=mp.get_context('spawn') if num_workers > 0 else None,
        )
        return sampler, loader
