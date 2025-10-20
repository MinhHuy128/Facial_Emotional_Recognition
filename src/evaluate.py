import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from model import LightweightEmotionCNN

def evaluate_model(model, val_loader, device, model_name):
    print(f"Evaluating {model_name}...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    emotion_labels = val_loader.dataset.classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs('../results', exist_ok=True)
    plt.savefig(f'../results/cm_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    return {
        'Model': model_name,
        'Accuracy': f"{acc*100:.2f}%",
        'Precision': f"{precision*100:.2f}%",
        'Recall': f"{recall*100:.2f}%",
        'F1-Score': f"{f1*100:.2f}%"
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    VAL_DIR = '../data/RAF-DB/DATASET/test'
    BATCH_SIZE = 64
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    results = []
    
    try:
        model_vgg = LightweightEmotionCNN(num_classes=7).to(device)
        model_vgg.load_state_dict(torch.load('../models/vgg_best.pth', map_location=device))
        results.append(evaluate_model(model_vgg, val_loader, device, "Custom Mini-VGG"))
    except Exception as e:
        print(f"Skipping VGG: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv('../results/evaluation_metrics.csv', index=False)
        print("\n--- Evaluation Metrics ---")
        print(df.to_markdown(index=False))
        
if __name__ == '__main__':
    main()
