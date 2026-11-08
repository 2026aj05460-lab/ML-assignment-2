import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Evaluation", layout="wide")

st.title("Breast Cancer Classification Model Evaluation")
st.markdown("Upload test dataset to evaluate the 5 pre-trained models without retraining.")

# Sidebar controls
st.sidebar.header("User Controls")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

@st.cache_resource
def load_artifacts(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(test_df.head(5))
    
    if 'target' not in test_df.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        model, scaler = load_artifacts(model_options[selected_model_name])
        
        # Scaling for distance and margin-based models
        if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test
            
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 6 Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")
        
        st.divider()
        
        col_cm, col_cr = st.columns([1, 1])
        
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with col_cr:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(precision=4))
else:
    st.info("Upload test_data.csv from the sidebar to view evaluation metrics.")

