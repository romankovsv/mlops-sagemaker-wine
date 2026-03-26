import boto3
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# SageMaker endpoint name
ENDPOINT_NAME = "wine-quality-endpoint"
import os
REGION = os.environ.get("AWS_REGION", "us-east-1")

# SageMaker runtime client
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html"
)


# Prediction API
@app.post("/predict")
def predict(
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):

    # convert to CSV format with headers
    payload = f"{fixed_acidity},{volatile_acidity},{citric_acid},{residual_sugar},{chlorides},{free_sulfur_dioxide},{total_sulfur_dioxide},{density},{pH},{sulphates},{alcohol}"

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        result = response["Body"].read().decode()

        # inference.py returns plain CSV string e.g. "5.3421"
        prediction_value = result.strip().split(",")[0]

        return {
            "prediction": str(round(float(prediction_value), 2))
        }
    except Exception as e:
        return {
            "prediction": f"Error: {str(e)}"
        }
