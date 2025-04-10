 End-to-End MLOps Project: Calories Burnt Predictor

This project demonstrates a production-grade MLOps pipeline that predicts calories burnt during exercise using a trained XGBoost model. It showcases a full lifecycle from data ingestion (via S3), model experimentation (with MLflow & DVC), to CI/CD automation, Docker containerization, and cloud deployment using AWS EKS.

****** Follow projectflow.txt file for each step *******

The system includes:

🧠 Model tracking & versioning with DVC + MLflow (hosted on DagsHub)

🔁 CI/CD pipeline built with GitHub Actions that trains, evaluates, and auto-promotes the best model

🐳 Dockerized Flask app for predictions with secure AWS credentials

☁️ Deployed on AWS EKS, pulling models from ECR, and logging metrics to Prometheus

📊 Live Monitoring using Grafana dashboards connected to Prometheus

✅ Fully tested with unittest, including model and app test cases

Project was built completely from scratch with production scalability in mind and is a great example of real-world MLOps engineering using modern tools. Please folow projectflow.txt file where each step is written and image recorded. 
