import torch
import torchvision.transforms as transforms
import torchvision
import os
import natsort
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# from torchvision.transforms import ToTensor, Resize


class MyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        PIL_image, kind = sample["image"], sample["kind"]
        resized_image = Resize
        # image = ToTensor()(image).unsqueeze(0)
        image = ToTensor()(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {
            # "image": torch.from_numpy(image),
            "image": image,
            "kind": kind,
        }


train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


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
        image_kind = int(image_name.split("_", 1)[0])
        img_loc = os.path.join(self.main_dir, image_name)
        # image = io.imread(img_loc)
        image = Image.open(img_loc)
        # print(image)

        if self.transform:
            image = self.transform(image)
            # sample = self.transform(sample)
        sample = {"image": image, "kind": image_kind}

        return sample

        # tensor_image = self.transform(image)
        # return tensor_image


transformed_dataset = CustomDataSet("./food-11/training/", train_transform)
# print(transformed_dataset.shape())
# print(transformed_dataset[0])
# image_folder[1]

dataloader = DataLoader(transformed_dataset, shuffle=True, batch_size=4)

valid_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


valid_set = CustomDataSet("./food-11/validation/", valid_test_transform)
validloader = DataLoader(transformed_dataset, batch_size=4)


class TestDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image_name = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, image_name)
        # image = io.imread(img_loc)
        image = Image.open(img_loc)
        # print(image)

        if self.transform:
            image = self.transform(image)
            # sample = self.transform(sample)

        return image


test_set = TestDataSet("./food-11/testing/", valid_test_transform)
testloader = DataLoader(transformed_dataset, batch_size=4)
