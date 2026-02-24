# Diabetes Hospital Readmission Predictor

A machine learning web application that predicts the risk of hospital readmission for diabetes patients within 30 days of discharge. Built with Flask and deployed on AWS.

## Project Overview

Hospital readmissions are expensive and often preventable. This application uses machine learning to identify high-risk diabetes patients so healthcare providers can provide targeted follow-up care.

**What it does:**
- Predicts probability of readmission within 30 days
- Categorises patients into Low/Medium/High risk groups
- Provides a web interface for clinicians
- Offers an API for integration with hospital systems

## Model Performance

- **Algorithm:** Gradient Boosting Classifier
- **Training Data:** 100,000+ patient encounters from 130 US hospitals
- **Key Metrics:** ROC AUC, Recall, F1 Score optimized for catching readmissions

The model was trained on patient data including demographics, lab results, diagnoses, medications, and prior hospital visit history.

## Technical Architecture

The application runs on AWS infrastructure:

- **Frontend:** HTML/CSS form for data entry
- **Backend:** Flask application with Gunicorn server
- **Hosting:** AWS EC2 instance (Ubuntu 22.04)
- **Storage:** AWS S3 for model files
- **ML Pipeline:** scikit-learn for preprocessing and prediction

## Technology Stack

**Machine Learning:**
- scikit-learn (Gradient Boosting, preprocessing)
- pandas and numpy (data processing)
- joblib (model serialization)

**Web Application:**
- Flask (web framework)
- Gunicorn (production server)
- HTML/CSS (frontend)

**Cloud Services:**
- AWS EC2 (application server)
- AWS S3 (model storage)
- AWS IAM (access management)
- boto3 (AWS Python SDK)

## Project Structure

```
diabetes-readmission-predictor/
│
├── app.py                    # Flask application and routes
├── predict.py                # Prediction logic
├── preprocessing.py          # Data preprocessing pipeline
├── s3_loader.py             # S3 model loading
│
├── templates/
│   ├── patient_form.html    # Input form
│   └── result.html          # Results display
│
├── requirements.txt         # Dependencies
└── README.md
```
## To Run

```

pip install -r requirements.txt
python app.py

```
