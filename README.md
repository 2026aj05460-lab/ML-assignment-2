# Machine Learning Assignment 2: Classification Models & Deployment

## a. Problem Statement
The goal is to implement and evaluate five supervised machine learning classification algorithms on the Wisconsin Diagnostic Breast Cancer dataset to predict whether a biopsy is benign or malignant, and deploy the evaluation pipeline via a Streamlit web interface.

## b. Dataset Description
* **Dataset:** Breast Cancer Wisconsin (Diagnostic)
* **Source:** UCI Machine Learning Repository / Scikit-learn
* **Instances:** 569 samples (Requirement: >= 500)
* **Features:** 30 continuous numerical features computed from digitized FNA images (Requirement: >= 12)
* **Target:** Binary classification (`target`: 0 = Malignant, 1 = Benign)

## c. GitHub Repository Link
* **GitHub Repository:** https://github.com/2026aj05460-lab/ML-assignment-2
* **Live Streamlit App:** https://ml-assignment--2.streamlit.app/

## d. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes (GaussianNB) | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |

## e. Observations on Model Performance
* **Logistic Regression:** Top performer across all key metrics (Accuracy: 0.9825, AUC: 0.9954, MCC: 0.9623). FNA cell morphological measurements show strong linear separability with malignancy, allowing $L_2$-regularized logistic loss to converge to an optimal decision boundary.
* **Support Vector Machine (RBF):** Closely matches Logistic Regression with an Accuracy of 0.9825, MCC of 0.9623, and AUC of 0.9950. The radial basis function separates subtle non-linear margin variations effectively.
* **K-Nearest Neighbors (KNN, k=5):** Strong local clustering performance (Accuracy: 0.9561, MCC: 0.9054), benefiting from standard scaling across the 30 continuous features.
* **Naive Bayes (GaussianNB):** Good discriminative ranking (AUC: 0.9878, Accuracy: 0.9386), although class-conditional feature independence is partially violated by correlated cell dimension features.
* **Decision Tree (CART):** Lowest scoring model (Accuracy: 0.9211, MCC: 0.8341). Orthogonal axis-aligned splits exhibit higher variance on continuous biological measurements.

### Best Model Selection
**Logistic Regression** and **Support Vector Machine** are the top choices, each missing only a single malignant case. Logistic Regression is selected as the primary production model due to its high interpretability, calibrated output probabilities, and minimal deployment overhead.
