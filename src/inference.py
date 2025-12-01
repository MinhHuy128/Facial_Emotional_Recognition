import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from model import EdgeEmotionMobileNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '../models/mobilenet_best.pth'

model = EdgeEmotionMobileNet(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() 

emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # BUG: Too sensitive parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        
        img_tensor = transform(roi_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            emotion = emotion_labels[predicted.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('PyTorch Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
