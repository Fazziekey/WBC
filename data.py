import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
from transformers import AutoImageProcessor
from tqdm import tqdm

np.random.seed(0)

class WBCImageDataset(Dataset):
    def __init__(self, data_dir, mask_dir=None, transform=None, use_mask=True):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.use_mask = use_mask

        self.class_to_index = {"Basophil": 0, "Eosinophil": 1, "Lymphocyte": 2, "Monocyte": 3, "Neutrophil": 4}

        self.samples = []
        for cls in self.class_to_index:
            class_dir = os.path.join(data_dir, cls)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(".jpg"):
                        self.samples.append((os.path.join(class_dir, file_name), self.class_to_index[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.use_mask:
            mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".jpg"
            mask_path = os.path.join(self.mask_dir, list(self.class_to_index.keys())[label], mask_name)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")  
                image = self.apply_mask(image, mask)
            else:
                mask = None  

        if self.transform:

            image = self.transform(image)

            #     image = self.transform(image, return_tensors="pt")["pixel_values"][0]

        label = torch.tensor(label, dtype=torch.int64) 

        return image, label

    def apply_mask(self, image, mask):
        binary_mask = mask.point(lambda p: p > 128 and 255)
        masked_image = Image.composite(image, Image.new("RGB", mask.size, "black"), binary_mask)
        return masked_image


class SimclrImageDataset(Dataset):
    def __init__(self, data_dir, mask_dir=None, use_mask=True, size=128):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.use_mask = use_mask

        self.transform=ContrastiveLearningViewGenerator(
                            self.get_simclr_pipeline_transform(size=size),
                            n_views=2
                        )

        self.samples = []

        if isinstance(data_dir, list):
            for dir in data_dir:
                if os.path.exists(dir):
                    for file_name in os.listdir(dir):
                        if file_name.endswith(".jpg"):
                            self.samples.append(os.path.join(dir, file_name))
                else:
                    print("data_dir path Error!!!!!")
        else:
            if os.path.exists(data_dir):
                for file_name in os.listdir(data_dir):
                    if file_name.endswith(".jpg"):
                        self.samples.append(os.path.join(data_dir, file_name))
            else:
                print("data_dir path Error!!!!!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.use_mask:
            mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".jpg"
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")  
                image = self.apply_mask(image, mask)
            else:
                mask = None  

        images = self.transform(image)

        return images
        
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def apply_mask(self, image, mask):
        binary_mask = mask.point(lambda p: p > 128 and 255)
        masked_image = Image.composite(image, Image.new("RGB", mask.size, "black"), binary_mask)
        return masked_image


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),  # The Image will transfer to 3*128*128
    #     transforms.ToTensor(),
    # ])

    # transform = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    # data_dir = "/mnt/bd/fazzie-models/data/WBC_10/data/"  
    # mask_dir = "/mnt/bd/fazzie-models/data/WBC_10/mask/"  
    # dataset = WBCImageDataset(data_dir=data_dir, mask_dir=mask_dir, transform=transform)

    # # Create DataLoader
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for images, labels in data_loader:
    #     print(images)
    #     print(images.shape)
    #     print(labels)

    #     break

    data_dir = "/mnt/bd/fazzie-models/data/pRCC_nolabel" 
    dataset = SimclrImageDataset(data_dir=data_dir, mask_dir=None, use_mask=False) 
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True,
        num_workers=12, pin_memory=True, drop_last=True)

    images = dataset[0]
    # print(images)
    print(len(images))
    print(images[0].shape)
    print(images[1].shape)


    for images in tqdm(data_loader):

        print(len(images))
        images = torch.cat(images, dim=0)
        print(images.shape)
        break