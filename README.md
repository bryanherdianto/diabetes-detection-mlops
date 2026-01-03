# Diabetes Detection with TFX MLOps Pipeline

![Image of Pipeline](images/model_plot.png)

<p align="center">
 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
 <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
 <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" />
 <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
 <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" />
 <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" />
 <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" />
</p>

## Problem

Detecting the presence of diabetes in patients often requires manual checking, which can be inaccurate. Using machine learning algorithms, diabetes can be detected quickly and accurately by analyzing clinical features such as Body Mass Index (BMI), age, blood sugar levels, blood pressure, and other health factors.

## Machine Learning Solution

This project utilizes a classification model to predict whether a particular patient has diabetes or not.

## Flowchart

![Program Flowchart](https://i.imgur.com/9KpDbuF.png)  

The pipeline begins when data or code is pushed to GitHub, which triggers the GitHub Actions workflow. The process initiates TFX components starting with ExampleGen for data ingestion, followed by StatisticsGen and SchemaGen, and then ExampleValidator for data validation. Next, Transform performs feature engineering before the Trainer handles model training and the Evaluator conducts a model performance check. At the decision point, if the model is not blessed, the pipeline stops; however, if it is blessed, the process proceeds to the Pusher to export model artifacts. The workflow then commits the new model to the repository and triggers the Render deploy hook to build and deploy the FastAPI Docker image. Finally, the model can also be monitored using Prometheus for performance metrics.

## Why TFX for MLOps

This project uses TensorFlow Extended (TFX) instead of standard TensorFlow to implement a robust MLOps (Machine Learning Operations) workflow. TFX provides a production-ready framework that automates data validation, transformation, and model evaluation. Unlike a simple training script, TFX ensures that only models meeting specific performance thresholds are deployed, preventing model decay and ensuring data consistency across the entire lifecycle.

## CI/CD with GitHub Actions

The MLOps pipeline is integrated with GitHub Actions to achieve Continuous Integration and Continuous Deployment (CI/CD). Whenever new data is committed to the repository or the module code is updated, GitHub Actions automatically triggers the TFX pipeline. This process includes data ingestion, schema validation, feature engineering, model training, and rigorous evaluation.

## Deployment on Render

Once the model is blessed by the evaluator, it is automatically deployed to Render. Due to the 512MB RAM limitation on Render's free tier, the deployment utilizes a custom FastAPI Dockerfile instead of the standard TensorFlow Serving image. This lightweight approach ensures the model remains responsive and stays within the resource constraints of the hosting environment.

## Dataset

[Diabetes Dataset](https://www.kaggle.com/datasets/lara311/diabetes-dataset-using-many-medical-metrics)

## Data Processing Method

Numerical label data (e.g., 0 or 1) is converted into one-hot vectors through one-hot encoding, which represents categorical data as binary vectors. Tensor labels consisting of values 0 or 1 are transformed into binary vectors of length 2. Additionally, feature values are normalized to a range of 0 to 1, ensuring that raw values with different ranges are scaled consistently. Labels are cast to 64-bit integer data type using `tf.cast` to ensure compatibility with the model or subsequent processes.

## Model Architecture

The model consists of several Dense layers for further feature processing. Transformed features (Pregnancies_xf, Glucose_xf, BloodPressure_xf, SkinThickness_xf, Insulin_xf, BMI_xf, DiabetesPedigreeFunction_xf, and Age_xf) are combined using a Concatenate layer. The first Dense layer with 256 units and activation function processes the combined features. The second Dense layer with 64 units processes the output from the previous layer, followed by the third Dense layer with 16 units. The results from these layers are used for the final classification or prediction stage of the model.

## Evaluation Metrics

This project utilizes AUC, Precision, Recall, and Binary Accuracy as metrics to evaluate how well the model handles classification problems.

1) Binary Accuracy measures how often the model's predictions are correct in binary classification, calculated as the ratio of correct predictions to the total number of predictions.
2) AUC (Area Under the ROC Curve) measures the model's ability to distinguish between positive and negative classes, providing an overview of model performance across different classification thresholds.
3) Precision is the ratio of correctly predicted positive observations to the total predicted positives, indicating how many of the model's positive predictions are actually positive.
4) Recall is the ratio of correctly predicted positive observations to all actual positives, measuring how well the model detects all positive examples.

## Model Performance

The model is expected to perform well in detecting diabetes in patients. It achieved 100% accuracy on training data and 70% on validation data, indicating that the model is fairly effective in detecting diabetes.

## Web App

Web app link for accessing model serving: [diabetes-detection-mlops](https://diabetes-detection-mlops.onrender.com/v1/models/serving_model:predict).

## Monitoring

Model serving is monitored using Prometheus. Prometheus provides various metrics related to the FastAPI model serving, such as request counts and latency, allowing for real-time observability of the model's performance in production.
