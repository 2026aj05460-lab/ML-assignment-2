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

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall performance across all evaluated metrics (Accuracy: 0.9825, AUC: 0.9954, MCC: 0.9623). FNA cell morphological measurements show strong linear separability with nuclear malignancy, allowing L2-regularized logistic loss to converge to an optimal, robust decision boundary with minimal risk of overfitting. |
| **Decision Tree** | Recorded the lowest metrics among the five algorithms (Accuracy: 0.9211, AUC: 0.9163, MCC: 0.8341). Orthogonal axis-aligned rectangular splits are inherently more sensitive to variance and noise when partitioning continuous biological cell measurements. |
| **KNN** | Delivered strong local clustering classification with an Accuracy of 0.9561, AUC of 0.9788, and MCC of 0.9054. Performance is well preserved due to rigorous standard scaling across all 30 continuous features, overcoming scale disparities. |
| **Naive Bayes** | Demonstrated high discriminative ranking capability with an AUC of 0.9878, Accuracy of 0.9386, and Recall of 0.9583. Although the feature independence assumption is partially violated by correlated dimension features (e.g., radius, perimeter, and area), its probabilistic ranking remains very competitive. |
| **Support Vector Machine** | Closely matched Logistic Regression with an Accuracy of 0.9825, MCC of 0.9623, and AUC of 0.9950. The radial basis function maps subtle non-linear margin variations effectively while maintaining strong geometric margin maximization. |
| **Overall, Winner for your dataset?** | While **Logistic Regression** and **Support Vector Machine (RBF)** tied for peak performance (0.9623 MCC, with just one missed malignant case), **Logistic Regression** was chosen as the deployment champion. In oncology screening, transparent predictions are essential so pathologists can audit which cell traits drive a diagnosis. Its direct log-odds interpretability, natively calibrated confidence scores, and instant inference make it the most dependable choice for real-time clinical deployment. |
