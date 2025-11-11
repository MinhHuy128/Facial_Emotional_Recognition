import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LightweightEmotionCNN, EdgeEmotionMobileNet, HeavyEmotionResNet

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

    models = {
        "Mini-VGG": (LightweightEmotionCNN(num_classes=7), '../models/vgg_best.pth', True),
        "MobileNetV2": (EdgeEmotionMobileNet(num_classes=7), '../models/mobilenet_best.pth', True),
        "ResNet-50": (HeavyEmotionResNet(num_classes=7), '../models/resnet_best.pth', True)
    }

    for name, (model, path, is_rgb) in models.items():
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
        else:
            print(f"Warning: {path} not found.")

    test_dir = '../data/RAF-DB/DATASET/test'
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found.")
        return

    classes = os.listdir(test_dir)
    sample_images = []
    for cls in random.sample(classes, min(3, len(classes))):
        cls_dir = os.path.join(test_dir, cls)
        img_name = random.choice(os.listdir(cls_dir))
        sample_images.append((os.path.join(cls_dir, img_name), cls))

    for img_path, true_label in sample_images:
        print(f"\n--- Testing Image: {os.path.basename(img_path)} (True Label: {true_label}) ---")
        img_pil = Image.open(img_path)

        for name, (model, _, is_rgb) in models.items():
            if is_rgb:
                img = img_pil.convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                img = img_pil.convert('L')
                transform = transforms.Compose([
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                ])

            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output.data, 1)
                
                sorted_classes = sorted([d.name for d in os.scandir(test_dir) if d.is_dir()])
                predicted_label = sorted_classes[pred.item()]

            print(f"{name:15s} Predicted: {predicted_label}")

if __name__ == '__main__':
    test_models()
