# Bank Customer Churn Prediction using Neural Networks

## Overview

This project aims to predict which customers are likely to churn (leave the bank) using historical customer data. Customer churn is a critical metric for banks, and accurate prediction allows for proactive retention strategies. The core of this solution utilizes neural networks (specifically, a Multilayer Perceptron), and a significant portion of the methodology addresses how to handle the common challenge of an imbalanced dataset, where the number of non-churning customers vastly outweighs those who do churn.

## Key Technologies

- **Python**
- **Pandas** and **NumPy** for data manipulation and analysis
- **Scikit-learn** for data preprocessing, handling imbalance, and evaluation metrics
- **TensorFlow** / **Keras** for building and training the neural networks

## Methodology: Handling Imbalanced Data

Imbalanced datasets can lead to models that perform well on the majority class but poorly on the minority (churning) class. This project employs several techniques to mitigate this issue:

1.  **Exploratory Data Analysis (EDA):** Initial analysis to understand the extent of the imbalance.
2.  **Resampling Techniques:**
    *   **Oversampling (SMOTE - Synthetic Minority Over-sampling Technique):** Generating synthetic examples of the minority class to balance the distribution.
3.  **Class Weights:** Adjusting the training process within the neural network framework to penalize misclassifications of the minority class more heavily.
4.  **Appropriate Evaluation Metrics:** Using metrics more suitable for imbalanced classification than simple accuracy, such as:
    *   **Precision, Recall, and F1-Score**
    *   **Confusion Matrix Analysis**

## Model: Neural Network Architecture

A feed-forward neural network (Multilayer Perceptron) is implemented using Keras. The architecture is designed to capture complex, non-linear relationships in the customer data:

- **Input Layer:** Matches the number of features after one-hot encoding and scaling.
- **Hidden Layers:** Multiple dense layers with `ReLU` activation for learning complex patterns.
- **Output Layer:** A single neuron with a `Sigmoid` activation function to output the probability of churn.

## Results :

The final model achieved an ROC AUC score of approximately ROC AUC Score: 0.8663296976146138, demonstrating its effectiveness in identifying high-risk customers despite the inherent data imbalance. The F1-score for the minority class was significantly improved by implementing the described handling techniques.
Accuracy :  0.87  , F1 Score : 0.87 (0) and 0.87 (1)
