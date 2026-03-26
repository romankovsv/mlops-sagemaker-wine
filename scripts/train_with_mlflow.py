import os
import json
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")

def get_csv_path(train_dir: str) -> str:
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError(f"No CSV found in {train_dir}")
    return os.path.join(train_dir, csv_files[0])

# 1. Load Data
csv_path = get_csv_path(TRAIN_DIR)
df = pd.read_csv(csv_path)

# IMPORTANT: If using 'wine_converted.csv', target is index 0. 
# If using original 'wine.csv', target is index -1.
y = df.iloc[:, 0]  
X = df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define and Train Model (Do this ONCE)
params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# 3. Evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

# 4. Save Model Locally (Always do this for SageMaker)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

# 5. MLflow Logging (Optional block)
if MLFLOW_URI:
    try:
        import mlflow
        import mlflow.xgboost
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("wine-quality-training")
        
        with mlflow.start_run():
            # Autolog handles params, metrics, and model logging automatically!
            mlflow.xgboost.autolog() 
            model.fit(X_train, y_train) # Autolog triggers during .fit()
            
            # Log custom metrics if autolog misses them
            mlflow.log_metric("custom_rmse", float(rmse))
            mlflow.log_metric("custom_r2", float(r2))
            
        print("[OK] Logged to MLflow")
    except Exception as e:
        print(f"[WARN] MLflow failed: {e}")

print(f"[SUCCESS] Model saved. RMSE: {rmse:.4f}, R2: {r2:.4f}")