# CIFAR-10 Classification: Architectural Efficiency & Redundancy Analysis

This directory contains a comparative study of image classification on the CIFAR-10 dataset. It explores the transition from standard Feed-Forward Neural Networks (ANN) to Convolutional Neural Networks (CNN), with a specific focus on how different architectures handle high-dimensional RGB data and spatial correlations.

## 📁 Folder Structure
* `ANN_CIFAR10.ipynb`: A study on the limitations of dense architectures, documenting the "Curse of Dimensionality" and parameter explosion.
* `CNN_CIFAR10.ipynb`: An optimized implementation utilizing spatial hierarchies and translation invariance for superior classification accuracy.

## 🛠️ Technical Workflow

### 1. The Dimensionality Challenge
* **Feature Analysis:** Comparative study of Grayscale (1,024 features) vs. RGB (3,072 features).
* **Redundancy Mapping:** Analysis of why FFNNs fail to exploit inter-channel correlations, leading to inevitable overfitting on high-dimensional inputs.

### 2. Architectural Implementations
* **ANN Prototype:** A multi-layer dense network used to benchmark the baseline performance of standard neural layers on complex image data.
* **CNN Prototype:** A hierarchical feature-extraction model using `Conv2D` and `MaxPooling2D` layers to reduce degrees of freedom and focus on local spatial patterns.

### 3. Key Findings
* **Structural Bias:** Demonstrated that without convolutional filters, parameter count increases without a proportional gain in information density.
* **Invariance:** CNNs successfully captured spatial features that were lost in the flattened input of the ANN model.

## 📊 Performance Comparison
| Architecture | Approach | Key Observation |
| :--- | :--- | :--- |
| **ANN** | Flattened Vector | High redundancy; restricted generalization. |
| **CNN** | Spatial Tensors | Efficient feature extraction; translation invariance. |

## 📂 Tech Stack
* **Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib, Scikit-learn
* **Hardware:** Optimized for T4 GPU acceleration.
