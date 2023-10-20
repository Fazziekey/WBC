import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification, ResNetModel
import wandb
from tqdm import tqdm

from models import ResNetSimCLR
from data import SimclrImageDataset
from loss import info_nce_loss

# Creat Train DataLoader
use_mask = False
batch_size = 128

train_data_dir = [
    "/mnt/bd/fazzie-models/data/pRCC_nolabel",  
    "/mnt/bd/fazzie-models/data/CAM16_100cls_10mask/train/data/normal/",
    "/mnt/bd/fazzie-models/data/CAM16_100cls_10mask/train/data/tumor/",
]

train_mask_dir = None
train_dataset = SimclrImageDataset(data_dir=train_data_dir, mask_dir=train_mask_dir, use_mask=use_mask, size=224)
print(f"train_dataset numbers {len(train_dataset)}")

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# Init model and loss function, optimizar
model = ResNetSimCLR(base_model='resnet50', out_dim=128)

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0,
                                                        last_epoch=-1)

# Init device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train
num_epochs = 50

# wandb.init(project="SimCLR", name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# wandb.watch(model) 


step_bar = tqdm(range(num_epochs * len(train_data_loader)), desc=f'steps')

for epoch in range(num_epochs):
    model.train()  
    train_loss = 0.0

    for images in train_data_loader:
        images = torch.cat(images, dim=0).to(device)

        features = model(images)
        logits, labels = info_nce_loss(features, device, batch_size=batch_size)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        step_bar.set_description('loss: %3.3f' % loss.item())
        step_bar.update()

    epoch_loss = train_loss / len(train_data_loader.dataset)

    # wandb.log({
    #     "Train Loss": epoch_loss,
    # })
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print('Finished Training')

filename = "checkpoint.pth"

torch.save(model.state_dict(), filename)


