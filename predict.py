# predict.py

import joblib
import pandas as pd
from preprocessing import preprocess_single_patient
from s3_loader import load_model

# Load the trained model 
model = load_model('gradient_boosting_model.pkl')


def predict_readmission(patient_data):
    """
    Predicts whether a patient will be readmitted to the hospital.
    
    Args:
        patient_data: dict containing patient information with keys matching 
                     the original dataset columns (e.g., 'age', 'gender', 
                     'time_in_hospital', 'num_medications', etc.)
    
    Returns:
        dict with:
            - 'prediction': 0 (not readmitted) or 1 (readmitted)
            - 'probability_not_readmitted': float (0-1)
            - 'probability_readmitted': float (0-1)
            - 'risk_level': 'Low', 'Medium', or 'High'
    """
    # Preprocess the input data
    processed_data = preprocess_single_patient(patient_data)
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    probabilities = model.predict_proba(processed_data)[0]
    
    # Determine risk level based on probability of readmission
    prob_readmit = probabilities[1]
    if prob_readmit < 0.3:
        risk_level = "Low"
    elif prob_readmit < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        'prediction': int(prediction),
        'probability_not_readmitted': float(probabilities[0]),
        'probability_readmitted': float(probabilities[1]),
        'risk_level': risk_level
    }


