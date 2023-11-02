import os
import glob

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobSysVImDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size

        self.total_dirs = sorted(glob.glob(os.path.join(self.root, "*")))

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test_id':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        elif phase == 'full':
            self.total_dirs = self.total_dirs

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_dirs)

    def __getitem__(self, idx):
        img_loc_source = os.path.join(self.total_dirs[idx], 'source.png')
        image_source = Image.open(img_loc_source).convert("RGB")
        image_source = image_source.resize((self.img_size, self.img_size))
        tensor_image_source = self.transform(image_source)

        img_loc_target = os.path.join(self.total_dirs[idx], 'target.png')
        image_target = Image.open(img_loc_target).convert("RGB")
        image_target = image_target.resize((self.img_size, self.img_size))
        tensor_image_target = self.transform(image_target)

        return tensor_image_source, tensor_image_target
