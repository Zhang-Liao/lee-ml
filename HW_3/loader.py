import torch
import torchvision
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        # self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        print(img_loc)
        image = io.imread(img_loc)
        # image = Image.open(img_loc).convert("RGB")
        return img_loc
        # tensor_image = self.transform(image)
        # return tensor_image


image_folder = CustomDataSet("./food-11/training/")
image_folder[1]
