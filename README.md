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
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## e. Observations on Model Performance
* **Logistic Regression:** Performs strongly with an accuracy of 0.9825 and an AUC of 0.9954. Standard scaling enables linear separation across continuous features.
* **Decision Tree:** Yields an accuracy of 0.9211 and MCC of 0.8341. While interpretable, individual decision trees exhibit higher variance on unseen test partitions.
* **KNN:** Achieves an accuracy of 0.9561 and MCC of 0.9054. Local neighborhood voting works effectively across standardized continuous attributes.
* **Naive Bayes:** Scores 0.9386 in accuracy and 0.9878 in AUC. The Gaussian conditional independence assumption holds well, though inter-feature correlations slightly reduce accuracy.
* **Support Vector Machine:** Achieves an accuracy of 0.9825 and an AUC of 0.9950. The radial basis function (RBF) kernel accurately models margin boundaries.
* **Overall Winner:** **Logistic Regression** and **Support Vector Machine (SVM)** are tied for the highest Accuracy (0.9825), Precision (0.9861), Recall (0.9861), F1 (0.9861), and MCC (0.9623), with Logistic Regression holding a slight edge in AUC (0.9954).
