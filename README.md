# Emotion Recognition Project

This is my personal project for facial emotion recognition. I built this to learn about Deep Learning, PyTorch, and how to deploy a model using FastAPI and OpenCV. 

It predicts 7 basic emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral.

## How to setup and run

### 1. Create virtual environment
First, you need to create a python virtual environment so it doesn't mess up your global python:

```bash
# For Windows
python -m venv env
.\env\Scripts\activate

# For Mac/Linux
python3 -m venv env
source env/bin/activate
```

### 2. Install packages
Install all the required libraries:
```bash
pip install -r requirements.txt
pip install scikit-learn matplotlib seaborn pandas tabulate
```

### 3. Download weights
Because the model weights are too large for GitHub, please download them from my Google Drive link here:
[Download Models Here](https://drive.google.com/drive/folders/1APRgIuQ5Tef1lhNMSDRfehjbNzNxwlcS?usp=sharing)

Put the 3 `.pth` files inside the `models/` folder.

### 4. Run the code
Move into the `src` folder first:
```bash
cd src
```

**To test with Webcam:**
```bash
python inference.py
```
Press 'q' to stop the webcam.

**To run the Web App (Website):**
```bash
python app.py
```
Then open your browser and go to `http://localhost:8000` to upload an image and test it.

---

## What I Learned
Through this project, I learned a lot of new things:
- How to build a Convolutional Neural Network (CNN) from scratch.
- How to use Transfer Learning with MobileNetV2 and ResNet-50.
- How to use OpenCV to detect faces in an image/webcam using Haar Cascades.
- How to build a simple backend API using FastAPI to serve my deep learning model.
- How to handle data transformations and write training loops in PyTorch.

## Pros and Cons of the Models

I tested 3 models for this project:

**1. My Custom Mini-VGG**
- **Pros:** Easy to code, small, fast to train. Good for learning the basics.
- **Cons:** Accuracy is the lowest (76.53%). It struggles with hard emotions like Disgust or Fear.

**2. MobileNetV2**
- **Pros:** Very lightweight and very fast! It gave 82.14% accuracy. This is the best model for running real-time on webcam because it doesn't lag.
- **Cons:** Not as accurate as heavier models.

**3. ResNet-50**
- **Pros:** Highest accuracy (85.46%). Very good at extracting features.
- **Cons:** Very heavy and slow. My laptop heats up quickly when running this for a long time.

## Future Improvements
If I have more time, I want to improve this project by:
- Using a better face detector like MTCNN or RetinaFace instead of OpenCV Haar Cascade, because Haar Cascade sometimes misses faces if the lighting is bad.
- Trying out Vision Transformers (ViT) to see if it gets better accuracy.
- Collecting more data for "Disgust" and "Fear" because the dataset is imbalanced.
- Deploying the web app to a real server like AWS or Heroku so anyone can try it online.
