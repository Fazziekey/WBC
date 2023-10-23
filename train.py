import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
import wandb
from tqdm import tqdm

from data import WBCImageDataset

# Init device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # The Image will transfer to 3*128*128
    transforms.ToTensor(),
])

# Creat Train DataLoader
use_mask = True    
train_data = 'WBC_100'

train_data_dir = f"path/to/{train_data}/train/data/"  
train_mask_dir = f"path/to/{train_data}/train/mask/"  
train_dataset = WBCImageDataset(data_dir=train_data_dir, mask_dir=train_mask_dir, transform=transform, use_mask=use_mask)

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Creat Test DataLoader
test_data_dir = "path/to/WBC_100/val/"  
test_dataset = WBCImageDataset(data_dir=test_data_dir, transform=transform, use_mask=False)

test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Init model and loss function, optimizar

# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=5)
model = torchvision.models.resnet50(pretrained=False, num_classes=5).to(device)

ckpt = "epoch200-checkpoint.pth"
state_dict = torch.load(ckpt, map_location=device)
# state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):

  if k.startswith('backbone.'):
    if k.startswith('backbone') and not k.startswith('backbone.fc'):
      # remove prefix
      state_dict[k[len("backbone."):]] = state_dict[k]
  del state_dict[k]

log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

# # freeze all layers but the last fc
# for name, param in model.named_parameters():
#     if name not in ['fc.weight', 'fc.bias']:
#         param.requires_grad = False

# parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
# assert len(parameters) == 2  # fc.weight, fc.bias

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0,
                                                        last_epoch=-1)



# Train
num_epochs = 10 

wandb.init(project="WBC", name= train_data + '_use_mask' if use_mask else "")
wandb.watch(model) 

step_bar = tqdm(range(num_epochs * len(train_data_loader)), desc=f'steps')
for epoch in range(num_epochs):
    model.train()  
    train_loss = 0.0

    for images, labels in train_data_loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * images.size(0)

        wandb.log({
            "Loss": loss.item(),
        })
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
        logits = model(images)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        step_bar.update()

accuracy = 100 * correct / total

wandb.log({
    "accuracy": accuracy,
})
print(f'Accuracy of the network on the test images: {accuracy}%')

# Save model

filename = f"{train_data}-epoch{num_epochs}-mask-checkpoint.pth" if use_mask else f"{train_data}-epoch{num_epochs}-checkpoint.pth"

torch.save(model.state_dict(), filename)

