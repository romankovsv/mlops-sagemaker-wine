import os
import json
import time
import boto3
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
S3_BUCKET  = os.environ.get("S3_BUCKET", "")

def get_csv_path(train_dir: str) -> str:
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError(f"No CSV found in {train_dir}")
    return os.path.join(train_dir, csv_files[0])


def _build_ge_html(result) -> str:
    """Render a simple HTML page from a GE validation result."""
    rows = []
    for r in result.results:
        status = "&#x2705; PASS" if r.success else "&#x274C; FAIL"
        expectation = r.expectation_config.expectation_type
        kwargs = {k: v for k, v in r.expectation_config.kwargs.items() if k != "batch_id"}
        rows.append(f"<tr><td>{status}</td><td>{expectation}</td><td>{kwargs}</td></tr>")
    overall = "PASSED" if result["success"] else "FAILED"
    color = "#2e7d32" if result["success"] else "#c62828"
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    passed = sum(1 for r in result.results if r.success)
    total = len(result.results)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GE Validation Report</title>
<style>body{{font-family:Arial,sans-serif;padding:24px;max-width:960px;margin:auto}}
h1{{color:#333}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:10px;text-align:left}}
th{{background:#f0f0f0}}tr:nth-child(even){{background:#fafafa}}</style></head>
<body><h1>Wine Quality &#8212; Data Validation Report</h1>
<p>Generated: {ts}</p>
<h2 style="color:{color}">Overall: {overall} ({passed}/{total} passed)</h2>
<table><tr><th>Status</th><th>Expectation</th><th>Parameters</th></tr>
{chr(10).join(rows)}</table></body></html>"""


def validate_dataset(df: pd.DataFrame) -> None:
    """Run Great Expectations checks and upload HTML report to S3."""
    try:
        import great_expectations as gx

        context = gx.get_context(mode="ephemeral")
        validator = context.sources.pandas_default.read_dataframe(dataframe=df)

        validator.expect_table_column_count_to_equal(12)
        validator.expect_table_columns_to_match_ordered_list([
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "quality",
        ])
        validator.expect_table_row_count_to_be_between(min_value=500, max_value=10000)
        validator.expect_column_values_to_not_be_null("quality")

        result = validator.validate()

        if S3_BUCKET:
            report_key = f"reports/data-validation/{int(time.time())}.html"
            boto3.client("s3").put_object(
                Bucket=S3_BUCKET,
                Key=report_key,
                Body=_build_ge_html(result).encode("utf-8"),
                ContentType="text/html",
            )
            print(f"[GE] Report: s3://{S3_BUCKET}/{report_key}")

        if not result["success"]:
            failed_count = sum(1 for r in result.results if not r.success)
            raise RuntimeError(f"[GE] Data validation failed: {failed_count} expectation(s) not met")

        print(f"[GE] {len(result.results)}/{len(result.results)} expectations passed")

    except ImportError:
        print("[GE] great_expectations not installed \u2014 skipping validation")


# 1. Load Data
csv_path = get_csv_path(TRAIN_DIR)
df = pd.read_csv(csv_path)
validate_dataset(df)

y = df.iloc[:, -1]   # quality = last column
X = df.iloc[:, :-1]  # 11 feature columns

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
print(f"rmse={rmse:.4f};")  # scraped by SageMaker metric_definitions regex

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
        mlflow.xgboost.autolog()  # must be BEFORE start_run

        with mlflow.start_run():
            # No second fit — log the already-trained model explicitly
            mlflow.log_params(params)
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", float(r2))
            mlflow.xgboost.log_model(model, artifact_path="model")

            metrics_path = os.path.join(MODEL_DIR, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"rmse": float(rmse), "r2": float(r2)}, f)
            mlflow.log_artifact(metrics_path)

        print("[OK] Logged to MLflow")
    except Exception as e:
        print(f"[WARN] MLflow failed: {e}")
        raise  # re-raise so the real error appears in CloudWatch/GitHub Actions logs

print(f"[SUCCESS] Model saved. RMSE: {rmse:.4f}, R2: {r2:.4f}")