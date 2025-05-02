# Pneumonia Detection from Chest X-rays using Deep Learning

A computer vision project using TensorFlow and CNNs to classify chest X-ray images as **Normal** or **Pneumonia**. Built to explore how deep learning can support rapid and accurate medical diagnoses using real-world imaging data.

## 🚀 Features
- Built in **Python** using **TensorFlow/Keras**

- Trains a Convolutional Neural Network (CNN) on labeled X-ray images
  
- Achieves binary classification: `PNEUMONIA` vs `NORMAL`
  
- Predicts on individual images or full test datasets
  
- Outputs per-image prediction confidence scores


## 📁 Dataset

Use the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
Place it inside a `data/` folder like so:

#### Pneumonia Detection from Chest X-rays using Deep Learning/
#### │
#### data/
#### ├── train/
#### │────── NORMAL/
#### │────── PNEUMONIA/
#### └── test/
#### ├────────NORMAL/
#### └────────PNEUMONIA/
#### │
#### pneumonia_detection.py     (model training + saving)
#### predict_all.py             (predict on all test images)
#### requirements.txt           (dependencies)
#### .gitignore                 (ignore model and dataset)
#### README.md                  (project documentation)


## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python pneumonia_detection.py

### 3. Run predictions on all test images

python predict_all.py

## 🧠 Model
The model is a CNN with three convolutional layers and dropout, trained on resized 150x150 X-ray images.

## 🧬 Model Architecture
#### Input: 150x150 grayscale image
#### Conv2D(32) → ReLU → MaxPool
#### Conv2D(64) → ReLU → MaxPool
#### Conv2D(128) → ReLU → MaxPool
#### Flatten → Dense(128) → Dropout → Dense(1, sigmoid)


## 📦 Output
Predictions are printed with confidence scores for each test image.

#### Example Output:

#### IM-0001.jpeg (NORMAL) ➜ Predicted: Normal (Confidence: 0.12)
#### person1_bacteria_1.jpeg (PNEUMONIA) ➜ Predicted: Pneumonia (Confidence: 0.91)


## 📘 What I Learned
- Applied real-world medical imaging to deep learning

- Built end-to-end image classifiers using CNNs in TensorFlow

- Preprocessed grayscale radiographs for model compatibility

- Interpreted model predictions with confidence scores

- Navigated ethical use of AI in healthcare contexts


## 🤝 Credits
Based on TensorFlow + Keras. Dataset by Paul Mooney on Kaggle.
