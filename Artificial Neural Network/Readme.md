# Neural Network from Scratch 🧠

A comprehensive, ground-up implementation of a single-neuron neural network (Perceptron) using **Python** and **NumPy**. This project demonstrates the fundamental mathematics of deep learning, focusing on manual derivation of backpropagation and vectorized gradient descent.



## 🚀 Overview
The goal of this project is to demystify the "black box" of machine learning frameworks. By implementing a neuron without using libraries like TensorFlow or PyTorch, we explore:
* **Forward Propagation**: Calculating linear sums and applying activation functions.
* **Loss Function**: Implementing Binary Cross Entropy (Log Loss).
* **Backpropagation**: Deriving gradients using the Chain Rule.
* **Vectorization**: Optimizing calculations for speed using NumPy.

---

## 📐 Mathematical Foundations

### 1. The Setup
For a single neuron with weights $w$ and bias $b$:
* **Linear Combination ($z$):** $$z = w_1x_1 + w_2x_2 + \dots + b$$
* **Activation ($a$):** $$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. The Chain Rule Derivation
To find how the Loss ($L$) changes with respect to weights, we calculate:
1. **Loss w.r.t Activation:** $\frac{\partial L}{\partial a} = \frac{a - y}{a(1 - a)}$
2. **Activation w.r.t Linear Sum:** $\frac{\partial a}{\partial z} = a(1 - a)$
3. **Linear Sum w.r.t Weight:** $\frac{\partial z}{\partial w_1} = x_1$

**Final Gradient:**
$$\frac{\partial L}{\partial w_1} = (a - y)x_1$$



### 3. Vectorization (Matrix Form)
To handle multiple training examples efficiently:
$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (A - Y)$$

---

## 💻 Implementation Highlights

* **Custom Log Loss**: Includes epsilon-clipping to prevent mathematical errors with `log(0)`.
* **Standard Scaling**: Implements data normalization to ensure the gradient descent converges efficiently.
* **OOP Design**: Includes a `Node` class to store trained parameters and perform inference.

## 📊 Results
The model was trained on an insurance dataset to predict purchasing behavior based on age and affordability.
* **Learning Rate**: 0.3
* **Epochs**: 1000
* **Accuracy**: **87.5%**



---

## 🛠️ Requirements
* Python 3.x
* NumPy
* Pandas
* Matplotlib (for visualization)

## 📖 Usage
1. Clone the repository.
2. Ensure dependencies are installed: `pip install numpy pandas matplotlib`.
3. Open `Neural Network from Scratch.ipynb` in your preferred Jupyter environment and run the cells.

---
*Developed as a deep-dive into the mechanics of Artificial Intelligence.*
