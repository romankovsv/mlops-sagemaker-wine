# MLOps Project

- **SageMaker** — training pipeline and providing endpoint
- **MLflow** — experiments tracking and model registry
- **Great Expectations** — data validation
- **SageMaker Monitor** — data drift detection, Champion/Challenger model promotion
- **FastAPI** — web API for inference
- **Terraform** — infrastructure: EC2 for MLflow server, S3 for dataset and model artifacts, IAM roles


![alt text](image.png)

```mermaid
flowchart TD
    subgraph Developer["👨‍💻 Developer"]
        GH["GitHub Push / Manual Trigger"]
    end

    subgraph CI["⚙️ GitHub Actions"]
        TW["train.yml\nTrain Wine Pipeline"]
        DW["deploy.yml\nDeploy Model\nauto-triggers on train success"]
    end

    subgraph Training["🏋️ SageMaker Training Pipeline"]
        CP["create_pipeline.py\nOrchestrates Pipeline"]
        TS["TrainingStep\nml.m5.large container"]
        TF["train_with_mlflow.py\n─────────────────\n1. Great Expectations validation\n2. Train XGBRegressor\n3. Evaluate RMSE / R²\n4. Save model.joblib\n5. Log to MLflow"]
    end

    subgraph Validation["✅ Data Quality"]
        GE["Great Expectations\n7 expectations checked\n─────────────────\n• 12 columns\n• correct column names\n• quality in 3–9\n• no nulls\n• alcohol / pH ranges\n• row count 500–10000"]
        GER["HTML Report\ns3://.../reports/data-validation/"]
    end

    subgraph Artifacts["☁️ AWS S3"]
        S3D["s3://.../data/wine.csv"]
        S3M["s3://.../models/\nmodel.tar.gz"]
        S3R["s3://.../monitor/\nbaseline / reports / captured-data"]
    end

    subgraph Tracking["📊 MLflow on EC2"]
        ML["Experiment: wine-quality-training\n─────────────────\nParams: hyperparameters\nMetrics: RMSE, R²\nArtifact: model + metrics.json"]
    end

    subgraph Deploy["🚀 deploy_latest_model.py"]
        CC["Champion / Challenger\n─────────────────\nRead champion RMSE from endpoint tag\nRead challenger RMSE from training job\nBlock if >5% worse"]
        RP["Repackage model.tar.gz\nInject inference.py + requirements.txt\ninto code/ directory"]
        SM_MODEL["Create SageMaker Model\nSKLearn container 1.2-1"]
        SM_EP["Create Endpoint\nwine-quality-endpoint\nml.m5.large\nDataCapture enabled"]
        TAG["Tag endpoint\nchampion_rmse=X.XXXX"]
        MON["Model Monitor\n─────────────────\nBaseline from wine.csv\nDaily drift schedule\nCloudWatch metrics"]
    end

    subgraph Inference["🌐 FastAPI ml-api"]
        API["app.py\nuvicorn :8000\n─────────────────\nGET  / → HTML form\nPOST /predict → score"]
        UI["index.html\n11 feature inputs"]
    end

    subgraph EndUser["👤 End User"]
        Browser["Browser"]
    end

    GH --> TW
    TW --> CP
    CP --> TS
    TS --> TF
    TF --> GE
    GE --> GER
    GER --> S3D
    S3D -.->|"input data"| TF
    TF --> S3M
    TF --> ML

    TW -->|"on success"| DW
    DW --> CC
    CC -->|"challenger wins"| RP
    S3M --> RP
    RP --> SM_MODEL
    SM_MODEL --> SM_EP
    SM_EP --> TAG
    TAG --> MON
    MON --> S3R

    Browser --> UI
    UI --> API
    API -->|"text/csv"| SM_EP
    SM_EP -->|"predicted quality score"| API
    API --> Browser

    SM_EP -->|"captured requests"| S3R
```

MLFlow version 2.21

starting on EC2

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://{{AWS_S3_BUCKET}}/mlflow/



To start FastApi server with Sagemaker endpoint backend

cd ml-api
pip install -r requirements.txt

# Set AWS credentials (needs sagemaker:InvokeEndpoint permission)
set region, aws access key and secret key

uvicorn app:app --host 0.0.0.0 --port 8000
