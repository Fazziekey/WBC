import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification, ResNetModel
import wandb
from tqdm import tqdm

from data import CamImageDataset, pRCCImageDataset


transform = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# Creat Train DataLoader
use_mask = True
train_data = 'WBC_100'

train_data_dir = "/mnt/bd/fazzie-models/data/WBC_100/train/data/"  
train_mask_dir = "/mnt/bd/fazzie-models/data/WBC_100/train/mask/"  
train_dataset = pRCCImageDataset(data_dir=train_data_dir, mask_dir=train_mask_dir, transform=transform, use_mask=use_mask)

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Creat Test DataLoader
test_data_dir = "/mnt/bd/fazzie-models/data/WBC_100/val/"  
test_dataset = pRCCImageDataset(data_dir=test_data_dir, transform=transform, use_mask=False)

test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Init model and loss function, optimizar
model = ResNetModel.from_pretrained("microsoft/resnet-50")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0,
                                                        last_epoch=-1)

# Init device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train
num_epochs = 5

wandb.init(project="WBC", name= train_data + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + 'use_mask' if use_mask else "")
wandb.watch(model) 

step_bar = tqdm(range(num_epochs * len(train_data_loader)), desc=f'steps')
for epoch in range(num_epochs):
    model.train()  
    train_loss = 0.0

    for images, labels in train_data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        step_bar.set_description('loss: %3.3f' % loss.item())
        step_bar.update()

    epoch_loss = train_loss / len(train_data_loader.dataset)

    wandb.log({
        "Train Loss": epoch_loss,
    })
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print('Finished Training')


# Test
model.eval()
correct = 0
total = 0
step_bar = tqdm(range(len(test_data_loader)), desc=f'steps')
with torch.no_grad():
    for images, labels in test_data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        step_bar.update()
        
accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy}%')

# Save model

model.save_pretrained(path='output')

