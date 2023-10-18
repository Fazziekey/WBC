import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None, use_mask=True):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.use_mask = use_mask
        # 类别和索引的对应关系
        self.class_to_index = {"Basophil": 0, "Eosinophil": 1, "Lymphocyte": 2, "Monocyte": 3, "Neutrophil": 4}
        # 收集图像文件的路径和相应的标签
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
        # 获取图像路径和标签
        img_path, label = self.samples[idx]
        # 加载图像
        image = Image.open(img_path).convert("RGB")

        if self.use_mask:
            # 尝试加载掩码（如果存在）
            mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".jpg"
            mask_path = os.path.join(self.mask_dir, list(self.class_to_index.keys())[label], mask_name)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")  # 加载mask为灰度图
                # 应用mask到图像上
                image = self.apply_mask(image, mask)
            else:
                mask = None  # 如果不存在mask，则不需要处理

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int64) 

        return image, label

    def apply_mask(self, image, mask):
        """将mask应用到图像上"""
        # 确保mask是二值的，将所有非黑即白
        binary_mask = mask.point(lambda p: p > 128 and 255)
        # 将mask应用到图像上，黑色区域将被置为0
        masked_image = Image.composite(image, Image.new("RGB", mask.size, "black"), binary_mask)
        return masked_image


if __name__ == '__main__':
    # 定义图像的转换方法
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # The Image will transfer to 3*128*128
        transforms.ToTensor(),
    ])

    # 实例化数据集
    data_dir = "/mnt/bd/fazzie-models/data/WBC_10/data/"  
    mask_dir = "/mnt/bd/fazzie-models/data/WBC_10/mask/"  
    dataset = CustomImageDataset(data_dir=data_dir, mask_dir=mask_dir, transform=transform)

    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 使用 DataLoader 迭代数据
    for images, labels in data_loader:
        print(images)
        print(images.shape)
        print(labels)

        break
