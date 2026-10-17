import os
import torch.multiprocessing as mp
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms.functional as TF


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _list_image_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)]
    return sorted(files)


class TestDataset(Dataset):
    def __init__(
        self,
        test_root,
        raw_dir='visible_clip',
        gt_dir='infrared_clip',
        guide_dir='infrared_clip_blur_unreg',
    ):
        super().__init__()
        self.test_root = test_root
        self.raw_path = os.path.join(test_root, raw_dir)
        self.gt_path = os.path.join(test_root, gt_dir)
        self.guide_path = os.path.join(test_root, guide_dir)

        for p in [self.raw_path, self.gt_path, self.guide_path]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f'测试集目录不存在: {p}')

        raw_files = _list_image_files(self.raw_path)
        gt_files = set(_list_image_files(self.gt_path))
        guide_files = set(_list_image_files(self.guide_path))

        missing_gt = [f for f in raw_files if f not in gt_files]
        missing_guide = [f for f in raw_files if f not in guide_files]
        if missing_gt:
            raise FileNotFoundError(f'以下文件在 GT 目录中缺失（示例前10个）: {missing_gt[:10]}')
        if missing_guide:
            raise FileNotFoundError(f'以下文件在 guide 目录中缺失（示例前10个）: {missing_guide[:10]}')

        self.filenames = raw_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        raw = Image.open(os.path.join(self.raw_path, name)).convert('RGB')
        gt = Image.open(os.path.join(self.gt_path, name)).convert('RGB')
        guide = Image.open(os.path.join(self.guide_path, name)).convert('RGB')

        raw = TF.to_tensor(raw)
        gt = TF.to_tensor(gt)
        guide = TF.to_tensor(guide)

        return raw, gt, guide, name


class TEST_DATALOADER:
    def __init__(
        self,
        config,
        test_root,
        raw_dir='visible_clip',
        gt_dir='infrared_clip',
        guide_dir='infrared_clip_blur_unreg',
        batch_size=1,
    ):
        self.config = config
        self.batch_size = batch_size
        self.test_dataset = TestDataset(
            test_root=test_root,
            raw_dir=raw_dir,
            gt_dir=gt_dir,
            guide_dir=guide_dir,
        )

    def get_loader(self, distributed=True):
        if distributed:
            sampler = DistributedSampler(
                self.test_dataset,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            multiprocessing_context=mp.get_context('spawn') if self.config.data.num_workers > 0 else None,
        )
        return sampler, loader
