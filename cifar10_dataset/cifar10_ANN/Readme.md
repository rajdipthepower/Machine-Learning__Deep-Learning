# Analysis of Dimensionality & Redundancy in Image Classification (CIFAR-10)

This repository contains a prototype study of image classification on the CIFAR-10 dataset using Feed-Forward Neural Networks (FFNN). The project specifically explores the impact of input dimensionality and inter-channel redundancy on model generalization.

## 🛠️ Technical Workflow

### 1. Data Processing
* **Dataset:** CIFAR-10 (60,000 32x32 color images in 10 classes).
* **Dimensionality Challenge:** Analysis of the shift from Grayscale (1,024 features) to RGB (3,072 features) and its effect on the "Curse of Dimensionality."

### 2. Model Architecture
* **Type:** Multi-layer Feed-Forward Neural Network (FFNN).
* **The "Generalization Gap":** This project documents why FFNNs fail to exploit spatial or channel correlations, leading to higher redundancy and overfitting on RGB inputs compared to grayscale.

### 3. Key Findings & Takeaways
* **Redundancy Analysis:** Observed that RGB increases input dimensionality without a proportional increase in unique information content for dense layers.
* **Architectural Constraints:** Confirmed that without Convolutional layers (CNN), parameter count explodes, making overfitting inevitable unless strong regularization or structural biases are applied.

## 📊 Key Results
* **Performance:** High training variance due to lack of spatial invariance.
* **Critical Insight:** Grayscale preprocessing helps dense models generalize better by removing inter-channel redundancy, bringing input dimensionality closer to true information dimensionality.

## 📂 Tech Stack
* **Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib, Scikit-learn
