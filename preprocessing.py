# preprocessing.py

import pandas as pd
import numpy as np
import joblib

# Load saved encoders (these will be loaded once when the app starts)
specialty_encoder = joblib.load('specialty_encoder.pkl')
health_index_discretizer = joblib.load('health_index_discretizer.pkl')
selected_meds = joblib.load('selected_meds.pkl')
# Load expected columns
expected_columns = joblib.load('expected_columns.pkl')

def preprocess_single_patient(patient_data):
    """
    Preprocesses a single patient's data for prediction.
    
    Args:
        patient_data: dict with patient information (matching your form fields)
    
    Returns:
        DataFrame with processed features ready for model
    """
    # Convert to DataFrame
    df = pd.DataFrame([patient_data])
    
    # 1. Clean missing values
    df.replace(['?', 'None', 'N/A', ' ', 'Unknown/Invalid'], np.nan, inplace=True)
    
    # 2. Drop unnecessary columns 
    to_drop = ['weight', 'encounter_id', 'patient_nbr', 'examide', 'citoglipton', 
               'glimepiride-pioglitazone', 'payer_code']
    df = df.drop(columns=[col for col in to_drop if col in df.columns], errors='ignore')
    
    # 3. Medication columns
    all_med_columns = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
    'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 
    'metformin-pioglitazone'
]

    # Drop meds NOT in selected_meds
    meds_to_drop = [col for col in all_med_columns if col not in selected_meds and col in df.columns]
    df = df.drop(columns=meds_to_drop, errors='ignore')

    
    # 4. Handle A1C and glucose
    df.loc[df.A1Cresult.isna(), 'A1Cresult'] = 'Not Available'
    df.loc[df.max_glu_serum.isna(), 'max_glu_serum'] = 'Not Available'
    
    df['A1Cresult'] = df['A1Cresult'].replace({
        "Not Available": 0, "Norm": 1, ">7": 2, ">8": 3
    })
    df['max_glu_serum'] = df['max_glu_serum'].replace({
        "Not Available": 0, "Norm": 1, ">200": 2, ">300": 3
    })
    
    # 5. Encode gender
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    
    # 6. Process age
    mapAge = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].replace(mapAge)
    
    # 7. Process medical specialty
   # Group mapping medical_specialty
    # Define specialty groups
    specialty_groups = {
        # Diabetes-related
        "Endocrinology": "Diabetes",
        "Endocrinology-Metabolism": "Diabetes",
        "Pediatrics-Endocrinology": "Diabetes",
    

        # Internal
        "InternalMedicine": "Internal",
        "Nephrology": "Internal",
        "Pulmonology": "Internal",
        "Gastroenterology": "Internal",
        "Hematology": "Internal",
        "Hematology/Oncology": "Internal",
        "InfectiousDiseases": "Internal",
        "Oncology": "Internal",
        "Hospitalist": "Internal",
        "Pathology": "Internal",
        "Rheumatology": "Internal",
        "AllergyandImmunology": "Internal",
        "Cardiology": "Internal",
        "Neurology": "Internal",
        "Neurophysiology": "Internal",

        # Surgical
        "Surgery-General": "Surgical",
        "Orthopedics": "Surgical",
        "Orthopedics-Reconstructive": "Surgical",
        "Surgery-Cardiovascular/Thoracic": "Surgical",
        "Surgery-Cardiovascular": "Surgical",
        "Surgery-Neuro": "Surgical",
        "Surgery-Vascular": "Surgical",
        "Surgery-Thoracic": "Surgical",
        "Surgery-Pediatric": "Surgical",
        "Surgery-Maxillofacial": "Surgical",
        "Surgery-Colon&Rectal": "Surgical",
        "Surgery-Plastic": "Surgical",
        "Surgery-PlasticwithinHeadandNeck": "Surgical",
        "Surgeon": "Surgical",
        "SurgicalSpecialty": "Surgical",
        "Proctology": "Surgical",
        "Podiatry": "Surgical",
        "Ophthalmology": "Surgical",
        "Otolaryngology": "Surgical",
        "Urology": "Surgical",
        "Anesthesiology": "Surgical",
        "Anesthesiology-Pediatric": "Surgical",
        "Dentistry": "Surgical",

        # Primary Care
        "Family/GeneralPractice": "PrimaryCare",
        "Emergency/Trauma": "PrimaryCare",
        "PhysicianNotFound": "PrimaryCare",
        "Resident": "PrimaryCare",
        "DCPTEAM": "PrimaryCare",
        "OutreachServices": "PrimaryCare",

        # Pediatrics
        "Pediatrics": "Pediatrics",
        "Pediatrics-CriticalCare": "Pediatrics",
        "Pediatrics-Neurology": "Pediatrics",
        "Pediatrics-Pulmonology": "Pediatrics",
        "Pediatrics-Hematology-Oncology": "Pediatrics",
        "Pediatrics-EmergencyMedicine": "Pediatrics",
        "Cardiology-Pediatric": "Pediatrics",
        "Anesthesiology-Pediatric": "Pediatrics",
        "Perinatology": "Pediatrics",

        # OB-GYN
        "ObstetricsandGynecology": "OBGYN",
        "Gynecology": "OBGYN",
        "Obstetrics": "OBGYN",
        "Obsterics&Gynecology-GynecologicOnco": "OBGYN",

        # Psychiatry
        "Psychiatry": "Psychiatry",
        "Psychology": "Psychiatry",
        "Psychiatry-Child/Adolescent": "Psychiatry",
        "Psychiatry-Addictive": "Psychiatry",

        # Other
        "Dermatology": "Other",
        "Speech": "Other",
        "Osteopath": "Other",
        "Radiologist": "Other",
        "Radiology": "Other",
        "SportsMedicine": "Other"
                }

    df['specialty_group'] = df['medical_specialty'].map(specialty_groups).fillna('Other')
    
    # Use the SAVED encoder (don't fit, just transform)
    df['specialty_encoded'] = specialty_encoder.transform(df[['specialty_group']])
    df = df.drop(columns=['medical_specialty'])
    
    # 8. Change medication flag
    df['change'] = df['change'].map({'No': 0, 'Ch': 1})
    
    # 9. Glucose management score
    def glucose_score(row):
        if row['diabetesMed'] == 'No':
            return 0
        elif row['change'] == 0:
            return 1
        elif row['change'] == 1:
            return 2
        else:
            return 1
    
    df['glucose_mgmt_score'] = df.apply(glucose_score, axis=1)
    
    # 10. Medication processing - use SAVED selected_meds list
    # Calculate features
    df['total_med_changes'] = df[selected_meds].apply(
        lambda x: x.isin(['Up', 'Down']).sum(), axis=1
    )
    df['insulin_changed'] = df['insulin'].isin(['Up', 'Down']).astype(int)
    
    def calculate_change_ratio(row):
        prescribed = sum(row[selected_meds] != 'No')
        changed = sum(row[selected_meds].isin(['Up', 'Down']))
        return changed / prescribed if prescribed > 0 else 0
    
    df['med_change_ratio'] = df.apply(calculate_change_ratio, axis=1).fillna(0)
    
    # One-hot encode medications
    df = pd.get_dummies(df, columns=selected_meds, drop_first=True, dtype=int)
    
    # 11. Process diagnoses
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(map_diagnosis_code)
    
    df = pd.get_dummies(df, columns=['diag_1', 'diag_2', 'diag_3'], 
                        prefix=['diag1', 'diag2', 'diag3'], 
                        drop_first=True, dtype=int)
    
    # 12. Transform skewed features
    df['number_inpatient'] = df['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))
    df['number_outpatient'] = df['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
    df['number_emergency'] = df['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
    
    # 13. Process race
    df['race'] = df['race'].replace({"Asian": "Other", "Hispanic": "Other"})
    
    # 14. Feature engineering
    df['num_visits'] = df['number_emergency'] + df['number_inpatient'] + df['number_outpatient']
    df['health_index'] = 1 / (1 + df['number_emergency'] + df['number_inpatient'] + df['number_outpatient'])
    df['total_procedures'] = df['num_lab_procedures'] + df['num_procedures'] + df['num_medications']
    df['log_time_in_hospital'] = np.log1p(df['time_in_hospital'])
    df['log_num_lab_procedures'] = np.log1p(df['num_lab_procedures'])
    
    # Use SAVED discretizer
    df['health_index_binned'] = health_index_discretizer.transform(df[['health_index']])
    
    df['medications_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 0.1)
    df['med_diag_interaction'] = df['num_medications'] * df['number_diagnoses']
    df['chronic_index'] = df['number_inpatient'] + 2 * df['number_emergency']
    df['new_med_or_change'] = ((df['change'] == 1) | (df['diabetesMed'] == 'Yes')).astype(int)
    
    df = df.drop(columns=['time_in_hospital', 'num_lab_procedures', 'health_index'])
    
    # 15. Process admission and discharge
    admission_type_map = {1: "Emergency", 2: "Emergency", 5: "Emergency",
                          3: "Scheduled", 4: "Other", 6: "Other", 8: "Other"}
    df['admission_type'] = df['admission_type_id'].map(admission_type_map)
    df = df.drop(columns=['admission_type_id'])

# Notes on discharge disposition codes and grouping:
# conditions_to_drop = [11,12,16,17,18,19,20]
#
# Mapping of discharge_disposition_id values (for reference):
# 1: "Discharged to home"
# 2: "Discharged to short-term hospital"
# 3: "Discharged to SNF"
# 4: "Discharged to ICF"
# 5: "Discharged to another facility"
# 6: "Discharged home with home health"
# 7: "Left against medical advice"
# 8: "Discharged home with HHA"
# 9: "Admitted as inpatient"
# 10: "To psychiatric hospital"
# 11: "To hospice (home)"
# 12: "To hospice (facility)"
# 13: "To long-term care hospital"
# 14: "To other institution"
# 15: "Not mapped"
# 16: "Expired"
# 17: "Expired at facility"
# 18: "Expired at home"
# 19: "Expired - place unknown"
# 20: "Expired - medical examiner"
# 22: "To rehab facility"
# 23: "To long-term hospital"
# 24: "To Medicaid nursing facility"
# 25: "To critical access hospital"
# 28: "Not available"

# Based on the above map, make a new map - grouping similar reasons
    discharge_disposition_map = {
        1: "Home",
        6: "Home",
        8: "Home",

        3: "Care Facility",
        4: "Care Facility",
        5: "Care Facility",
        13: "Care Facility",
        14: "Care Facility",
        22: "Care Facility",
        23: "Care Facility", 
        24: "Care Facility",
        

        2: "Hospital",
        9: "Hospital",
        10: "Hospital",
        25: "Hospital",

        16: "Expired/Hospice",
        17: "Expired/Hospice",
        18: "Expired/Hospice",
        19: "Expired/Hospice",
        20: "Expired/Hospice",
        11: "Expired/Hospice",
        12: "Expired/Hospice",

        7: "Other/Unknown",
        15: "Other/Unknown",
        28: "Other/Unknown"
    }


    
    df['discharge_disposition'] = df['discharge_disposition_id'].map(discharge_disposition_map)
    
    df = df[~df['discharge_disposition'].isin(['Expired/Hospice'])]
    df = df.drop(columns=['discharge_disposition_id'])
    
    # 16. Final encoding
    df = pd.get_dummies(df, columns=['race', 'admission_type', 'discharge_disposition', 
                                      'diabetesMed', 'specialty_group'], 
                        drop_first=True, dtype=int)
    # Align columns to expected
    df = align_columns(df)
    
    return df


def map_diagnosis_code(x):
    """Categorizes ICD-9 diagnosis codes"""
    try:
        if pd.isna(x) or str(x).strip() in ['', '?', 'unknown']:
            return 'other'
        if str(x).startswith(('V', 'E')):
            return 'other'
        
        code = int(float(x))
        
        if 390 <= code <= 459 or code == 785:
            return 'circulatory'
        elif 460 <= code <= 519 or code == 786:
            return 'respiratory'
        elif 520 <= code <= 579 or code == 787:
            return 'digestive'
        elif code == 250:
            return 'diabetes'
        elif 800 <= code <= 999:
            return 'injury'
        elif 710 <= code <= 739:
            return 'musculoskeletal'
        elif 580 <= code <= 629 or code == 788:
            return 'genitourinary'
        elif 140 <= code <= 239:
            return 'neoplasms'
        elif 630 <= code <= 679:
            return 'pregnancy'
        else:
            return 'other'
    except (ValueError, TypeError):
        return 'other'
    
    


def align_columns(df):
    """
    Ensures the processed dataframe has exactly the columns the model expects.
    Adds missing columns as 0s, removes extra columns.
    """
    # Add missing columns with value 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Remove extra columns not in expected
    df = df[expected_columns]
    
    return df