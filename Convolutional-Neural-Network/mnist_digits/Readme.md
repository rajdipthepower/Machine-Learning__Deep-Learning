# Comparative Analysis of Neural Architectures: ANN vs. CNN

This repository features a comprehensive deep learning pipeline for image classification using the MNIST dataset. The project focuses on a comparative study between standard Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN), emphasizing spatial feature extraction and architectural efficiency.

## 🔬 Research Alignment
This implementation is designed to showcase core competencies relevant to the faculty research interests at MFSDS&AI, IIT Guwahati:
* **Computer Vision & Identification:** Direct application of CNNs for image classification and recognition (Target: **Dr. Teena Sharma**).
* **Biomedical Image Analysis:** Utilizing deep learning for high-precision classification tasks (Target: **Dr. Debanga Raj Neyog**).
* **Interaction & Perception:** Exploring how AI models interpret visual signals (Target: **Dr. Neeraj Sharma**).

## 🛠️ Technical Workflow

### 1. Data Pipeline
* **Normalization:** Implemented Min-Max scaling to normalize pixel intensities to a [0, 1] range for optimized gradient descent.
* **Input Engineering:** Managed specific input shapes for both 1D flattened vectors (ANN) and 3D grayscale tensors (CNN).

### 2. Architecture Comparison
* **Multi-Layer ANN:** A dense 4-layer sequential model (100 -> 80 -> 50 -> 10 units) using ReLU activations and Softmax output.
* **Deep CNN:** A robust feature-extraction model featuring dual Conv2D layers (4x4 kernels) and MaxPooling2D for translation invariance.

### 3. Optimization & Metrics
* **Optimizer:** Adam.
* **Loss Function:** Sparse Categorical Crossentropy.
* **Evaluation:** Utilized Scikit-learn classification reports to analyze Precision, Recall, and F1-Scores across all 10 digit classes.

## 📊 Performance Summary
* **ANN Accuracy:** ~98%.
* **CNN Accuracy:** ~99% — demonstrating the superior capability of convolutional layers in capturing spatial hierarchies.

## 📂 Tech Stack
* **Framework:** Keras / TensorFlow (GPU Accelerated)
* **Libraries:** NumPy, Matplotlib, Scikit-learn

---
*Curated for application to the Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati.*
