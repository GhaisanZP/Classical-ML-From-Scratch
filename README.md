# Classical Machine Learning From Scratch

A pure Python and `NumPy` implementation of classical machine learning algorithms, mapped directly to fundamental mathematical concepts. 

This repository serves as a personal knowledge base and a proof-of-work portfolio. It translates core ML theories into clean, readable code without relying on high-level libraries for the core logic. To validate the correctness and benchmark the performance, **each custom algorithm is systematically compared against its `scikit-learn` counterpart** in the provided Jupyter Notebooks.

## Philosophy & Features
- **No Black Boxes:** Core training and prediction logic (e.g., Stochastic Gradient Ascent, Least Squared Estimation, Gini/Entropy calculations) are built entirely from scratch.
- **Mathematical Rigor:** Mathematical formulas and derivations are documented within the code's docstrings using LaTeX format.
- **Benchmarking:** Notebooks include rigorous comparisons with `scikit-learn` in terms of accuracy, R2-score, and execution time.
- **OOP Design:** Code is structured following Object-Oriented Programming principles, resembling standard ML library APIs (`.fit()`, `.predict()`).

## Algorithms Implemented

### Supervised Learning
**Linear Models**
- [x] Linear Regression (Least Squared Estimation / OLS & Gradient Descent)
- [ ] Logistic Regression (Maximum Likelihood Estimator via Stochastic Gradient Ascent)

**Instance-based & Probabilistic**
- [ ] K-Nearest Neighbors (k-NN with Euclidean, Manhattan, and Minkowski distances)
- [ ] Naive Bayes / Bayesian Network

**Tree-based & Ensemble Methods**
- [ ] Decision Tree Learning (ID3 Algorithm using Entropy and Information Gain)
- [ ] Random Forest (Bagging / Bootstrap Aggregation)
- [ ] AdaBoost (Iterative instance weight updates)
- [ ] Gradient Boosting (Iterative residual error minimization)

**Margin-based**
- [ ] Support Vector Machine (SVM)

### Unsupervised Learning
- [ ] K-Means Clustering
- [ ] Principal Component Analysis (PCA)

## Usage & Scikit-Learn Comparison

The algorithms are designed to be intuitive. Check the `notebooks/` directory for full comparative analyses. Here is a brief example of the comparative approach used in this repository:

```python
import numpy as np
from time import time
from classical_ml.linear_models.logistic_regression import LogisticRegression as CustomLogReg
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from utils.metrics import accuracy_score

# 1. Custom Implementation
start = time()
custom_model = CustomLogReg(learning_rate=0.01, n_iters=1000)
custom_model.fit(X_train, y_train)
custom_preds = custom_model.predict(X_test)
print(f"Custom Accuracy: {accuracy_score(y_test, custom_preds)} | Time: {time() - start:.4f}s")

# 2. Scikit-Learn Implementation
start = time()
sk_model = SklearnLogReg()
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)
print(f"Sklearn Accuracy: {accuracy_score(y_test, sk_preds)} | Time: {time() - start:.4f}s")
```