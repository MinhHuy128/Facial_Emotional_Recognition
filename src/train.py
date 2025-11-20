import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LightweightEmotionCNN, EdgeEmotionMobileNet, HeavyEmotionResNet
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['vgg', 'mobilenet', 'resnet'])
args = parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f" -> Early Stopping Warning: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = '../data/RAF-DB/DATASET/train' 
VAL_DIR = '../data/RAF-DB/DATASET/test'
BATCH_SIZE = 64
EPOCHS = 50

os.makedirs('../models', exist_ok=True)
if args.model == 'vgg':
    BEST_MODEL_PATH = '../models/vgg_best.pth'
elif args.model == 'mobilenet':
    BEST_MODEL_PATH = '../models/mobilenet_best.pth'
else:
    BEST_MODEL_PATH = '../models/resnet_best.pth'

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if args.model == 'vgg':
    model = LightweightEmotionCNN(num_classes=7).to(device)
elif args.model == 'mobilenet':
    model = EdgeEmotionMobileNet(num_classes=7).to(device)
else:
    model = HeavyEmotionResNet(num_classes=7).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
early_stopping = EarlyStopping(patience=10)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.2f}%")
    
    scheduler.step(val_loss_avg)
    
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    early_stopping(val_loss_avg)
    if early_stopping.early_stop:
        print("Early Stopping has been activated")
        break

print("DONE")
