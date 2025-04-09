from flask import Flask, render_template, request
import mlflow
import os
import pandas as pd
import dagshub
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

# === MLflow / DagsHub Setup ===

# Below code block is for local use
# -------------------------------------------------------------------------------------
""" mlflow.set_tracking_uri('https://dagshub.com/vucina19931906/Calories-Burnt-Predictor.mlflow')
dagshub.init(repo_owner="vucina19931906", repo_name="Calories-Burnt-Predictor", mlflow=True) """
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CALORIES_BURNT_PRED")
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    

dagshub_url = "https://dagshub.com"
repo_owner = "vucina19931906"  
repo_name = "Calories-Burnt-Predictor"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# -------------------------------------------------------------------------------------

# === Flask App Initialization ===
app = Flask(__name__)

# === Prometheus Metrics ===
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total app requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Prediction counts", ["result"], registry=registry)

# === Load Latest Model from MLflow ===
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    return versions[0].version if versions else None

model_name = "calories-burnt-xgb"  # ✅ Replace if needed
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"📦 Loading model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# === Routes ===

@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    try:
        input_data = {
            "Gender": 0 if request.form["Gender"].lower() == "male" else 1,
            "Age": float(request.form["Age"]),
            "Height": float(request.form["Height"]),
            "Weight": float(request.form["Weight"]),
            "Duration": float(request.form["Duration"]),
            "Heart_Rate": float(request.form["Heart_Rate"]),
            "Body_Temp": float(request.form["Body_Temp"])
        }

        df = pd.DataFrame([input_data])
        prediction = round(float(model.predict(df)[0]), 2)

        PREDICTION_COUNT.labels(result="calories").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

        return render_template("index.html", result=prediction)

    except Exception as e:  
        return render_template("index.html", result="Prediction failed. Please check your inputs.")

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
