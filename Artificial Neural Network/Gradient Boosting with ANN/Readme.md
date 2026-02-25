# Semiconductor Classification: A Hybrid Boosting Approach

This project explores the classification of semiconductors based on physical properties using a custom **Additive Boosting Ensemble** and **Random Forest** architectures.

## 🚀 The Architecture: LR + ANN Boosting
I implemented a custom boosting loop that optimizes the residuals of a linear baseline:
1. **Base Model:** Logistic Regression (captures global linear trends).
2. **Residual Learner:** A 3-layer ANN trained on the error terms ($y - P_{LR}$).
3. **Ensemble:** $P_{final} = P_{LR} + (k \cdot \text{ANN}_{residual})$

By optimizing the weighting factor ($k \approx 1.11$), the hybrid model achieved an accuracy of **86%**, significantly improving upon the standalone Logistic Regression baseline.



## 🌲 Random Forest Performance
While the hybrid model performed well, a **Random Forest (10 trees)** achieved **~99% accuracy**. 

**Key Insights:**
- **Physical Thresholds:** Physical phases are non-linear. Random Forest identifies these "cliffs" in the data (like Formation Energy and Density) more effectively than linear-based models.
- **Feature Importance:** `formation_energy_per_atom` was identified as the primary predictor (28%), followed by `density` and `energy_per_atom`.



## 🛠️ Technologies Used
- Python, TensorFlow/Keras
- Scikit-Learn (Logistic Regression, Random Forest, StratifiedKFold)
- Matplotlib/Seaborn for Physical Feature Analysis
