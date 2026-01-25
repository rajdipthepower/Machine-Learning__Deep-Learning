# 🌸 Flower Classification Using CNN (TensorFlow)

This project implements a **Convolutional Neural Network (CNN)** to classify images of flowers using the **TensorFlow Flowers Dataset**. The model is trained to recognize different flower categories from raw images using deep learning techniques.

---

## 📌 Project Overview

- Dataset: TensorFlow Flowers Dataset  
- Task: Multi-class image classification  
- Frameworks: TensorFlow / Keras  
- Approach: Custom CNN built and trained from scratch  

The notebook covers the complete pipeline:
1. Dataset download  
2. Image loading and preprocessing  
3. Label encoding  
4. CNN model creation  
5. Model training and evaluation  

---

## 🗂 Dataset

The dataset is automatically downloaded from TensorFlow’s official source.

### Flower Categories:
- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  

Each class contains multiple RGB images of flowers.

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Pillow (PIL)  

---

## 🧠 Model Architecture

The CNN model consists of:
- Convolutional layers for feature extraction  
- Pooling layers to reduce spatial dimensions  
- Fully connected (Dense) layers for classification  
- Softmax output layer for multi-class prediction  

---

## ⚙️ Installation & Setup

Clone the repository:

```bash  
git clone https://github.com/your-username/flower-classification-cnn.git
cd flower-classification-cnn


Install dependencies:

pip install tensorflow opencv-python pillow matplotlib numpy


Run the notebook:

jupyter notebook Flowers_dataset_CNN.ipynb

**## 🚀 How It Works**

Downloads and extracts the flower dataset automatically

Reads images using OpenCV and converts them to RGB format

Assigns numeric labels to each flower category

Trains a CNN on the image dataset

Evaluates model performance

---


**📊 Results**

The trained CNN is able to classify flower images into their respective categories with reasonable accuracy.
(Exact performance may vary depending on training configuration and hardware.)


---


**📌 Future Improvements**

Data augmentation for better generalization

Transfer learning using pretrained models (ResNet, MobileNet)

Hyperparameter tuning

Model saving and inference pipeline


