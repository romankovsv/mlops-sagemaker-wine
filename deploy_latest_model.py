"""
Executes ultimetely on Github Actions Runner
"""
import boto3
import os
import time
import tarfile
import tempfile
import shutil
import sagemaker

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
endpoint_name = "wine-quality-endpoint"
model_name = "wine-quality-model"

boto_session = boto3.Session(region_name=region)
sm = boto_session.client("sagemaker")
s3 = boto_session.client("s3")



IMAGE_URI = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
    image_scope="inference"
)
print(f"Image URI: {IMAGE_URI}")

print(f"Region: {region}")
print(f"Role: {role}")

RMSE_THRESHOLD_RATIO = 1.05  # allow up to 5% regression before blocking deploy


def get_champion_rmse(sm_client, ep_name):
    """Read champion RMSE from endpoint tag set after last successful deploy."""
    try:
        arn = sm_client.describe_endpoint(EndpointName=ep_name)["EndpointArn"]
        tags = sm_client.list_tags(ResourceArn=arn)["Tags"]
        for tag in tags:
            if tag["Key"] == "champion_rmse":
                return float(tag["Value"])
    except Exception:
        pass
    return None


def get_challenger_rmse(sm_client, job_name):
    """Read RMSE from SageMaker training job final metrics (scraped from stdout)."""
    try:
        job = sm_client.describe_training_job(TrainingJobName=job_name)
        for m in job.get("FinalMetricDataList", []):
            if m["MetricName"] == "rmse":
                return float(m["Value"])
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Find latest completed training job
# ═══════════════════════════════════════════════════════════════════
print(f"\n[1/6] Finding latest completed training job...")

jobs = sm.list_training_jobs(
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=20,
)["TrainingJobSummaries"]

latest_job = None
for job in jobs:
    if job["TrainingJobStatus"] == "Completed":
        latest_job = job["TrainingJobName"]
        break

if latest_job is None:
    raise Exception("No completed training job found")

print(f"  Training job: {latest_job}")

job_details = sm.describe_training_job(TrainingJobName=latest_job)
model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]
print(f"  Model artifact: {model_artifact}")

# ═══════════════════════════════════════════════════════════════
# Champion / Challenger evaluation
# ═══════════════════════════════════════════════════════════════
print("\n[C/C] Champion vs Challenger evaluation...")
champion_rmse = get_champion_rmse(sm, endpoint_name)
challenger_rmse = get_challenger_rmse(sm, latest_job)

if challenger_rmse is None:
    print("  [WARN] Challenger RMSE not in training job metrics — skipping comparison")
elif champion_rmse is None:
    print(f"  No champion yet (first deploy) — challenger RMSE: {challenger_rmse:.4f}")
else:
    print(f"  Champion RMSE:   {champion_rmse:.4f}")
    print(f"  Challenger RMSE: {challenger_rmse:.4f}")
    threshold = champion_rmse * RMSE_THRESHOLD_RATIO
    if challenger_rmse > threshold:
        print(f"  Challenger ({challenger_rmse:.4f}) exceeds threshold ({threshold:.4f}) — aborting deploy")
        raise SystemExit(0)
    print("  Challenger is equal or better — proceeding with deploy")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Repackage model.tar.gz with inference code inside
# ═══════════════════════════════════════════════════════════════════
print(f"\n[2/6] Repackaging model with inference code...")

bucket = model_artifact.split("/")[2]
key = "/".join(model_artifact.split("/")[3:])

tmpdir = tempfile.mkdtemp()
original_tar = os.path.join(tmpdir, "original.tar.gz")
extract_dir = os.path.join(tmpdir, "contents")
new_tar = os.path.join(tmpdir, "model.tar.gz")

s3.download_file(bucket, key, original_tar)

os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(original_tar, "r:gz") as tar:
    tar.extractall(extract_dir)
print(f"  Extracted: {os.listdir(extract_dir)}")

# Remove any old code/ directory
code_dir = os.path.join(extract_dir, "code")
if os.path.exists(code_dir):
    shutil.rmtree(code_dir)
os.makedirs(code_dir)

# Write inference.py
with open(os.path.join(code_dir, "inference.py"), "w") as f:
    f.write("""import joblib
import os
import numpy as np
import xgboost as xgb


FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type):
    if content_type == "text/csv":
        lines = request_body.strip().split("\\n")
        parsed = []
        for line in lines:
            row = [float(x.strip()) for x in line.split(",")]
            parsed.append(row)
        return np.array(parsed)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURE_NAMES)
    prediction = model.get_booster().predict(dmatrix)
    return prediction


def output_fn(prediction, accept):
    return ",".join(str(round(float(p), 4)) for p in prediction)
""")

# Write MINIMAL requirements.txt - ONLY xgboost
with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write("xgboost==2.0.3\n")

print(f"  code/ contents: {os.listdir(code_dir)}")

# Repackage tar
with tarfile.open(new_tar, "w:gz") as tar:
    for item in os.listdir(extract_dir):
        tar.add(os.path.join(extract_dir, item), arcname=item)

# Upload
new_key = f"models/deploy/model-{int(time.time())}.tar.gz"
s3.upload_file(new_tar, bucket, new_key)
new_model_uri = f"s3://{bucket}/{new_key}"
print(f"  Uploaded: {new_model_uri}")

shutil.rmtree(tmpdir)

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Cleanup ALL existing resources
# ═══════════════════════════════════════════════════════════════════
print(f"\n[3/6] Cleaning up existing resources...")

# Delete endpoint (wait if in-progress)
try:
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    status = ep["EndpointStatus"]
    print(f"  Endpoint status: {status}")

    # Wait for any in-progress state to finish
    while status in ("Creating", "Updating", "RollingBack", "Deleting"):
        print(f"  Endpoint is {status}, waiting 30s...")
        time.sleep(30)
        try:
            status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        except sm.exceptions.ClientError:
            print("  Endpoint gone")
            status = "Gone"
            break

    if status != "Gone":
        # Now it's InService or Failed — safe to delete
        sm.delete_endpoint(EndpointName=endpoint_name)
        print("  Delete requested, waiting for removal...")
        while True:
            try:
                time.sleep(15)
                sm.describe_endpoint(EndpointName=endpoint_name)
            except sm.exceptions.ClientError:
                print("  Endpoint deleted")
                break

except sm.exceptions.ClientError:
    print("  No existing endpoint")

# Delete endpoint configs
try:
    configs = sm.list_endpoint_configs(NameContains="wine-quality")
    for cfg in configs.get("EndpointConfigs", []):
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print(f"  Deleted config: {cfg['EndpointConfigName']}")
except Exception:
    pass

# Delete old models
try:
    sm.delete_model(ModelName=model_name)
    print(f"  Deleted model: {model_name}")
except Exception:
    pass

try:
    models = sm.list_models(SortBy="CreationTime", SortOrder="Descending", MaxResults=10)
    for m in models["Models"]:
        mn = m["ModelName"]
        if "wine" in mn.lower() or "sagemaker-scikit" in mn.lower():
            sm.delete_model(ModelName=mn)
            print(f"  Deleted model: {mn}")
except Exception:
    pass

print("  Waiting 15s...")
time.sleep(15)

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Create model with EXPLICIT env vars (no SDK magic)
# ═══════════════════════════════════════════════════════════════════
print(f"\n[4/6] Creating SageMaker model...")

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": new_model_uri,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": region,
        },
    },
    ExecutionRoleArn=role,
)
print(f"  Model created: {model_name}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Create endpoint config + endpoint
# ═══════════════════════════════════════════════════════════════════
print(f"\n[5/6] Creating endpoint...")

config_name = f"{endpoint_name}-config-{int(time.time())}"

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
        }
    ],
    DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://{bucket}/monitor/captured-data",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"},
        ],
        "CaptureContentTypeHeader": {
            "CsvContentTypes": ["text/csv"],
        },
    },
)
print(f"  Endpoint config: {config_name}")

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name,
)
print(f"  Endpoint creation started: {endpoint_name}")

# Wait for InService
print("  Waiting for endpoint to be InService...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)

final_status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]

# Tag endpoint with challenger RMSE for future champion/challenger comparisons
if challenger_rmse is not None:
    try:
        endpoint_arn = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointArn"]
        sm.add_tags(
            ResourceArn=endpoint_arn,
            Tags=[{"Key": "champion_rmse", "Value": str(round(challenger_rmse, 4))}],
        )
        print(f"  Tagged endpoint: champion_rmse={challenger_rmse:.4f}")
    except Exception as e:
        print(f"  [WARN] Failed to tag endpoint: {e}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Model Monitor — baseline + daily drift schedule
# ═══════════════════════════════════════════════════════════════════
print("\n[6/6] Setting up Model Monitor...")
try:
    from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator
    from sagemaker.model_monitor.dataset_format import DatasetFormat

    sm_session = sagemaker.Session(boto_session)
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.t3.medium",
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=sm_session,
    )

    # Remove existing schedule before recreating
    try:
        sm.delete_monitoring_schedule(MonitoringScheduleName="wine-drift-schedule")
        print("  Removed existing monitoring schedule")
        time.sleep(10)
    except Exception:
        pass

    # Suggest baseline from training data (~5 min Processing job)
    print("  Running baseline processing job (~5 min)...")
    monitor.suggest_baseline(
        baseline_dataset=f"s3://{bucket}/data/wine.csv",
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"s3://{bucket}/monitor/baseline",
        wait=True,
        logs=False,
    )
    print(f"  Baseline: s3://{bucket}/monitor/baseline/")

    # Create daily monitoring schedule
    monitor.create_monitoring_schedule(
        monitor_schedule_name="wine-drift-schedule",
        endpoint_input=endpoint_name,
        output_s3_uri=f"s3://{bucket}/monitor/reports",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True,
    )
    print("  Monitor schedule: wine-drift-schedule (runs daily)")
    print(f"  Reports: s3://{bucket}/monitor/reports/")

except Exception as e:
    print(f"  [WARN] Model Monitor setup failed: {e}")
    print("  Endpoint is running — monitor can be configured manually")

print(f"\n{'='*60}")
print(f"  ENDPOINT STATUS: {final_status}")
print(f"  ENDPOINT NAME:   {endpoint_name}")
print(f"{'='*60}")