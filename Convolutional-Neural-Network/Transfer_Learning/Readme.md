# Flower Species Classification: CNN vs. Transfer Learning

This project implements image classification on the **TensorFlow Flowers Dataset**. It provides a comparative analysis between a **Custom Convolutional Neural Network (CNN)** built from scratch and a **Transfer Learning** approach using the **MobileNetV2** architecture.



## 🎯 Project Objective
The primary goal is to demonstrate the effectiveness of Transfer Learning in computer vision tasks, specifically focusing on:
* **Developing and training** a multi-layer CNN from scratch.
* **Implementing Transfer Learning** by leveraging pre-trained weights from MobileNetV2.
* **Comparing the performance metrics** (Accuracy, Loss) between both methods.

## 📊 Performance Comparison
The following table summarizes the performance statistics of both models after training on the dataset.

| Model Architecture | Training Accuracy | Testing Accuracy | Training Loss | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | 87.40% | 81.79% | 0.3444 | Fast |
| **Transfer Learning (MobileNetV2)** | 95.09% | 87.32% | 0.1579 | Ultra-Fast |



### Key Findings
* **Generalization:** While the Custom CNN achieved a high training accuracy (87.11%), it showed signs of **overfitting** with a significantly lower validation accuracy (79.88%).
* **Transfer Learning Efficiency:** The MobileNetV2 model achieved a much higher validation accuracy (**88.35%**) despite a slightly lower training accuracy, proving it generalizes far better to unseen data.
* **Convergence:** Transfer Learning reached high performance levels significantly faster because it utilizes features already learned from the ImageNet dataset.

## 🛠 Methodology

### 1. Data Preprocessing
Images are fetched from the Google APIs flower dataset.
* **Resizing:** All images are normalized and resized to $224 \times 224$ for MobileNetV2 compatibility.
* **Augmentation:** Data augmentation (Random Flip, Rotation, and Zoom) was applied to help the model learn more robust features.

### 2. Custom CNN Architecture
A sequential model consisting of:
* Three `Conv2D` layers for spatial feature extraction with `ReLU` activation.
* `MaxPooling2D` layers to reduce spatial dimensions.
* `Flatten` and `Dense` layers for final classification into the 5 flower categories.



### 3. Transfer Learning (MobileNetV2)
* **Base Model:** MobileNetV2 with weights pre-trained on **ImageNet**.
* **Fine-tuning:** The base model layers were frozen, and a custom classification head (Global Average Pooling and a Dense layer) was added for the specific flower classes.



## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Open the Notebook:**
    Launch `Flowers_dataset_Transfer_learning.ipynb` in **Google Colab** or Jupyter Notebook.
3.  **Hardware Accelerator:**
    Ensure your runtime is set to **GPU** for faster training.
4.  **Execute:**
    Run all cells to download the dataset, train both models, and view the comparison plots.

## 📚 Requirements
* `tensorflow`
* `opencv-python`
* `matplotlib`
* `numpy`
* `Pillow`

---
*Developed as a comparative study on the efficiency of Deep Learning architectures.*
