import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
import joblib
import os

def preprocess(df):
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] // 3600) % 24
    df = df.drop(columns=['Time', 'Amount'])
    return df

def main():
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection")
    
    # Load data
    try:
        df = pd.read_csv("data/creditcard.csv")
    except FileNotFoundError:
        print("data/creditcard.csv not found. Please place the dataset in the data/ folder.")
        return

    df = preprocess(df)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Handle class imbalance
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scale Amount_log and Hour
    scaler = StandardScaler()
    cols_to_scale = ['Amount_log', 'Hour']
    X_train_res[cols_to_scale] = scaler.fit_transform(X_train_res[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # XGB params
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,
        "eval_metric": "aucpr",
        "random_state": 42,
        "early_stopping_rounds": 30
    }
    
    model = XGBClassifier(**xgb_params)
    
    with mlflow.start_run():
        mlflow.log_params(xgb_params)
        
        # Fit model
        model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)
        
        # Predict probabilities
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate pr_auc and roc_auc
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_probs)
        
        # Find best threshold for f1
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        mlflow.log_metrics({
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "fraud_f1": best_f1,
            "threshold": best_threshold
        })
        
        print(f"Metrics: pr_auc={pr_auc:.4f}, roc_auc={roc_auc:.4f}, fraud_f1={best_f1:.4f}, best_threshold={best_threshold:.2f}")
        
        # Ensure directories exist
        os.makedirs("models", exist_ok=True)
        
        # Save model and scaler locally
        model.save_model("models/model.json")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(scaler, "scaler.pkl")
        
        # Log to MLflow
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact("scaler.pkl")
        
        # Register model if pr_auc > 0.70
        if pr_auc > 0.70:
            print("pr_auc > 0.70, registering model to MLflow...")
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            model_name = "FraudDetectionXGB"
            mlflow.register_model(model_uri, model_name)
            
            client = MlflowClient()
            # Promote to Production
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                latest_version = latest_versions[0].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True
                )
                print(f"Model {model_name} version {latest_version} transitioned to Production.")

if __name__ == "__main__":
    main()
