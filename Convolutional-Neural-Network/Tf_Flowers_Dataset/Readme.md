# 🌸 Flower Classification Using CNN (TensorFlow)

This project implements a **Convolutional Neural Network (CNN)** to classify images of flowers using the **TensorFlow Flowers Dataset**. The model is trained to recognize different flower categories from raw images using deep learning techniques.

---

## 📌 Project Overview

- Dataset: TensorFlow Flowers Dataset  
- Task: Multi-class image classification  
- Frameworks: TensorFlow / Keras  
- Approach: Custom CNN built and trained from scratch  

The notebook covers the complete pipeline including dataset download, image preprocessing, label encoding, CNN model construction, training, and evaluation.

---

## 🗂 Dataset

The dataset is automatically downloaded from TensorFlow’s official source and contains five flower categories:

- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  

Each category consists of multiple RGB images.

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

The CNN model is composed of convolutional layers for feature extraction, pooling layers to reduce spatial dimensions, fully connected dense layers for classification, and a softmax output layer for multi-class prediction.

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


🚀 ## Workflow, Results, and Observations

The project automatically downloads and extracts the flower dataset, reads images using OpenCV, converts them into RGB format, and assigns numeric labels to each flower category. The images are preprocessed and fed into a custom-built Convolutional Neural Network, which is trained to learn discriminative visual features from the dataset. After training, the model is evaluated on the dataset and is able to classify flower images into their respective categories with reasonable accuracy. The final performance may vary depending on factors such as training configuration, number of epochs, and hardware used.

📌 ## Future Improvements

Data augmentation for better generalization

Transfer learning using pretrained models such as ResNet or MobileNet

Hyperparameter tuning

Model saving and inference pipeline
