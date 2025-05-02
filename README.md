# Pneumonia Detection from Chest X-rays using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Pneumonia** or **Normal**.

## 📁 Dataset

Use the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
Place it inside a `data/` folder like so:

data/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt

2. Train the model

python pneumonia_detection.py

3. Run predictions on all test images

python predict_all.py

## 🧠 Model
The model is a CNN with three convolutional layers and dropout, trained on resized 150x150 X-ray images.

## 📦 Output
Predictions are printed with confidence scores for each test image.

Example Output:

IM-0001.jpeg (NORMAL) ➜ Predicted: Normal (Confidence: 0.12)
person1_bacteria_1.jpeg (PNEUMONIA) ➜ Predicted: Pneumonia (Confidence: 0.91)

## 🤝 Credits
Based on TensorFlow + Keras. Dataset by Paul Mooney on Kaggle.
