from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# 加载数据集
dataset = load_dataset("huggingface/cats-image")

# 准备图片处理器和模型
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

def preprocess_function(examples):
    # 处理图片并返回PyTorch tensors
    inputs = processor(examples['image'], return_tensors="pt")
    inputs["labels"] = torch.tensor(examples['labels'])  # 假设您的数据集有一个名为'labels'的键
    return inputs

# 对数据集中的图片进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 数据加载器
train_dataloader = DataLoader(encoded_dataset['train'], shuffle=True, batch_size=8)  # 您可以根据需求更改batch_size

# 设置训练参数，例如学习率、训练轮数等
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=8,   # 每个GPU/CPU训练批大小
    warmup_steps=500,                # 预热步骤
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
)

# 定义训练器
trainer = Trainer(
    model=model,                         # 训练的模型
    args=training_args,                  # 训练参数
    train_dataset=encoded_dataset['train'], # 训练数据集
)

# 开始训练
trainer.train()
