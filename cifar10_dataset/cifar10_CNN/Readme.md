# High-Dimensional Image Classification: CNN Architecture on CIFAR-10

This repository contains the implementation of a Convolutional Neural Network (CNN) specifically optimized for the CIFAR-10 dataset. This project serves as an advanced sequel to the ANN prototype, demonstrating how convolutional priors successfully address the challenges of high-dimensional RGB image data.

## 🛠️ Technical Workflow

### 1. Data Engineering
* **Multi-Channel Processing:** Handling 32x32 RGB images (3,072 features) and converting label structures for optimized categorical processing.
* **Normalization:** Scaled pixel intensities to stabilize the learning process and speed up convergence.

### 2. CNN Architecture
* **Feature Extraction:** Utilized multiple `Conv2D` layers to capture local spatial features and inter-channel correlations.
* **Dimensionality Management:** Integrated `MaxPooling2D` layers to reduce the spatial resolution while maintaining dominant feature maps, significantly reducing the parameter count compared to a standard ANN.
* **Classification Head:** A dense layer pipeline designed to map high-level features to the 10 target classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

### 3. Optimization Strategy
* **Invariance:** Developed the model to be translation-invariant, allowing for robust detection regardless of the object's position in the frame.
* **Loss & Optimizer:** Implemented Adam optimization with Sparse Categorical Crossentropy for high-precision classification.

## 📊 Key Insights
* **Spatial Logic:** Unlike the ANN (which treats images as flat vectors), this CNN preserves the 2D structure of the image, leading to a significant increase in validation accuracy.
* **Efficiency:** Successfully demonstrated that structural bias (Convolutional filters) is the key to solving the "Curse of Dimensionality" in image-based signals.

## 📂 Tech Stack
* **Framework:** TensorFlow / Keras (GPU Accelerated)
* **Libraries:** NumPy, Matplotlib, Scikit-learn
