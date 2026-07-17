import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "rui-dataset-engineered.csv")
MODEL_EXPORT_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "eda")

def train_advanced_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Engineered dataset not found: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Power_Nm_RPM',
        'Temp_Difference_K'
    ]
    target_col = 'Machine failure'
    
    X = df[feature_cols]
    y = df[target_col]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scale_pos_weight = (len(y) - sum(y)) / sum(y)
    
    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight='balanced', random_state=42, verbosity=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }

    mlflow.set_experiment("Predictive_Maintenance_Models")
    
    best_model_name = None
    best_f1 = -1
    best_tuned_preds = None
    best_oof_probs = None
    best_threshold = 0.5
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"--- Training {model_name} ---")
            
            oof_probs = np.zeros(len(df))
            pr_aucs = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_train_scaled, y_train)
                
                # Predict probabilities
                probs = model.predict_proba(X_val_scaled)[:, 1]
                oof_probs[val_idx] = probs
                
                # PR-AUC for this fold
                p, r, _ = precision_recall_curve(y_val, probs)
                pr_aucs.append(auc(r, p))
            
            mean_pr_auc = np.mean(pr_aucs)
            print(f"{model_name} Mean PR-AUC: {mean_pr_auc:.4f}")
            
            # Find best threshold on OOF predictions
            precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            threshold = thresholds[best_idx]
            
            tuned_preds = (oof_probs >= threshold).astype(int)
            
            final_f1 = f1_score(y, tuned_preds)
            final_precision = precision_score(y, tuned_preds)
            final_recall = recall_score(y, tuned_preds)
            final_roc_auc = roc_auc_score(y, oof_probs)
            final_accuracy = accuracy_score(y, tuned_preds)
            final_pr_auc = auc(recalls, precisions)
            
            print(f"Best Threshold: {threshold:.4f}")
            print(f"Accuracy: {final_accuracy:.4f}")
            print(f"Precision: {final_precision:.4f}")
            print(f"Recall: {final_recall:.4f}")
            print(f"F1-Score: {final_f1:.4f}")
            print(f"ROC-AUC: {final_roc_auc:.4f}")
            print(f"PR-AUC: {final_pr_auc:.4f}\n")
            
            # Log to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", final_accuracy)
            mlflow.log_metric("precision", final_precision)
            mlflow.log_metric("recall", final_recall)
            mlflow.log_metric("f1_score", final_f1)
            mlflow.log_metric("roc_auc", final_roc_auc)
            mlflow.log_metric("pr_auc", final_pr_auc)
            mlflow.log_metric("best_threshold", threshold)
            
            # Save PR curve plot
            plt.figure(figsize=(7, 5))
            plt.plot(recalls, precisions, label=f"{model_name} (PR-AUC = {final_pr_auc:.3f})", color='purple', lw=2)
            plt.axvline(x=final_recall, color='red', linestyle='--')
            plt.scatter(final_recall, final_precision, color='red', marker='o', s=100)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve ({model_name})')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.tight_layout()
            pr_path = os.path.join(PLOTS_DIR, f"pr_curve_{model_name}.png")
            plt.savefig(pr_path, dpi=150)
            plt.close()
            mlflow.log_artifact(pr_path)
            
            # Save Confusion Matrix
            cm = confusion_matrix(y, tuned_preds)
            plt.figure(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Failure'])
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'Confusion Matrix ({model_name})')
            plt.tight_layout()
            cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            mlflow.log_artifact(cm_path)
            
            # Train final model on all data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
            
            if final_f1 > best_f1:
                best_f1 = final_f1
                best_model_name = model_name
                best_threshold = threshold
                best_tuned_preds = tuned_preds
                best_oof_probs = oof_probs
                
                # Save best model to disk for the app
                model_artifacts = {
                    'model': model,
                    'scaler': scaler,
                    'threshold': float(threshold),
                    'feature_cols': feature_cols,
                    'background_data': X_scaled[:100] # Save 100 samples for SHAP
                }
                with open(MODEL_EXPORT_PATH, 'wb') as f:
                    pickle.dump(model_artifacts, f)

    print(f"=== Best Model: {best_model_name} (F1: {best_f1:.4f}) ===")
    
    # Save best PR curve as generic pr_curve.png for README/UI
    precisions, recalls, _ = precision_recall_curve(y, best_oof_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f"Best: {best_model_name}", color='purple', lw=2)
    plt.axvline(x=recall_score(y, best_tuned_preds), color='red', linestyle='--')
    plt.scatter(recall_score(y, best_tuned_preds), precision_score(y, best_tuned_preds), color='red', marker='o', s=100)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Best Model)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150)
    plt.close()

    cm = confusion_matrix(y, best_tuned_preds)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Failure'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Best Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    train_advanced_pipeline()
