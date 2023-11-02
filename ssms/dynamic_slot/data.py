import os
import glob
import torch

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobTaskVideoDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_vids = sorted(glob.glob(root))

        if phase == 'train':
            self.total_vids = self.total_vids[:int(len(self.total_vids) * 0.7)]
        elif phase == 'val':
            self.total_vids = self.total_vids[int(len(self.total_vids) * 0.7):int(len(self.total_vids) * 0.85)]
        elif phase == 'test':
            self.total_vids = self.total_vids[int(len(self.total_vids) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_vids)

    def __getitem__(self, idx):
        vid_loc = self.total_vids[idx]
        source_frame_loc = os.path.join(vid_loc, 'source.png')
        target_frame_loc = os.path.join(vid_loc, 'target.png')

        source_image = Image.open(source_frame_loc).convert("RGB")
        source_image = source_image.resize((self.img_size, self.img_size))
        source_image = self.transform(source_image)

        target_image = Image.open(target_frame_loc).convert("RGB")
        target_image = target_image.resize((self.img_size, self.img_size))
        target_image = self.transform(target_image)

        return torch.stack([source_image, target_image], dim=0)
