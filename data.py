import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor


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
            image = self.transform(image, return_tensors="pt")["pixel_values"][0]

        label = torch.tensor(label, dtype=torch.int64) 

        return image, label

    def apply_mask(self, image, mask):
        binary_mask = mask.point(lambda p: p > 128 and 255)
        masked_image = Image.composite(image, Image.new("RGB", mask.size, "black"), binary_mask)
        return masked_image

class pRCCImageDataset(Dataset):
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
            image = self.transform(image, return_tensors="pt")["pixel_values"][0]

        label = torch.tensor(label, dtype=torch.int64) 

        return image, label

    def apply_mask(self, image, mask):
        """将mask应用到图像上"""
        binary_mask = mask.point(lambda p: p > 128 and 255)
        masked_image = Image.composite(image, Image.new("RGB", mask.size, "black"), binary_mask)
        return masked_image


class CamImageDataset(Dataset):
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
            image = self.transform(image, return_tensors="pt")["pixel_values"][0]

        label = torch.tensor(label, dtype=torch.int64) 

        return image, label

    def apply_mask(self, image, mask):
        """将mask应用到图像上"""
        binary_mask = mask.point(lambda p: p > 128 and 255)
        masked_image = Image.composite(image, Image.new("RGB", mask.size, "black"), binary_mask)
        return masked_image


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),  # The Image will transfer to 3*128*128
    #     transforms.ToTensor(),
    # ])
    transform = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    data_dir = "/mnt/bd/fazzie-models/data/WBC_10/data/"  
    mask_dir = "/mnt/bd/fazzie-models/data/WBC_10/mask/"  
    dataset = WBCImageDataset(data_dir=data_dir, mask_dir=mask_dir, transform=transform)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, labels in data_loader:
        print(images)
        print(images.shape)
        print(labels)

        break
