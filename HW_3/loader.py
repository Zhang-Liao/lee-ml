import torch
import torchvision
import os
import natsort
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, kind = sample["image"], sample["kind"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "kind": kind,
        }


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image_name = self.total_imgs[idx]
        image_kind = image_name.split("_", 1)[0]
        img_loc = os.path.join(self.main_dir, image_name)
        image = io.imread(img_loc)
        # print(image)

        sample = {"image": image, "kind": image_kind}

        if self.transform:
            sample = self.transform(sample)

        return sample

        # tensor_image = self.transform(image)
        # return tensor_image


transformed_dataset = CustomDataSet("./food-11/training/", ToTensor())
# print(len(image_folder))
# image_folder[1]

dataloader = DataLoader(transformed_dataset)

