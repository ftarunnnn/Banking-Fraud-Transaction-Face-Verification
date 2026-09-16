import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

def evaluate_models(model_dir='models', output_dir='visualizations'):
    X_test = np.load(os.path.join(model_dir, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(model_dir, 'y_test.npy'))
    
    rf_model = joblib.load(os.path.join(model_dir, 'fraud_model_rf.pkl'))
    xgb_model = joblib.load(os.path.join(model_dir, 'fraud_model_xgb.pkl'))
    
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    os.makedirs(output_dir, exist_ok=True)
    metrics_summary = []
    
    # 1. Confusion Matrix Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        metrics_summary.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xticklabels(['Normal', 'Fraud'])
        axes[idx].set_yticklabels(['Normal', 'Fraud'])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 2. ROC Curve Plot
    plt.figure(figsize=(9, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2.5)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5000)')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve Plot
    plt.figure(figsize=(9, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f'{name}', linewidth=2.5)
        
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
    plt.close()
    
    metrics_df = pd.DataFrame(metrics_summary)
    print("[OK] Phase 6: Model Evaluation Completed.")
    print(metrics_df.to_string(index=False))
    return metrics_df

if __name__ == '__main__':
    evaluate_models()
