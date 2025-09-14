# -*- coding: utf-8 -*-
"""
Configuration file for the MAAI Antibiotic Prediction project.
Holds all project constants, paths, and feature lists.
"""
import os

# 1. FILE PATHS
# TODO: User must update these paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'processed')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')

# 2. DATA FILENAMES
CONTINUOUS_VARS_FILE = 'raw_variables.csv'
DISCRETE_MEDS_FILE = 'raw_meds.csv'
PATIENT_OUTCOMES_FILE = 'outcomes.csv' # This file should contain patient_id, hour, and the antibiotic label

# 3. PREPROCESSING PARAMETERS
TIME_RESOLUTION_HOURS = 1
LOOKBACK_WINDOW_HOURS = 12 # How many hours of past data to use for statistical features

# 4. FEATURE LISTS
# Based on clinical relevance for infection/sepsis prediction

# Core continuous features to extract from the raw variables file
# NOTE: BP will be split into BP_Systolic and BP_Diastolic if it's a single column like '120/80'
VITALS_FEATURES = ['Pulse', 'SpO2', 'Resp', 'BP_Systolic', 'BP_Diastolic', 'MAP', 'Temp', 'FiO2', 'Oxygen_Flow', 'Ventilator_Mode', 'Tidal_Volume_Set', 'Coma_Scale_Total', 'Pupil_Left_Size', 'Pupil_Right_Size', 'Urine', 'Volume_Infused']
LABS_FEATURES = ['WBC', 'Platelets', 'Lactic_acid', 'Creatinine', 'Bilirubin', 'Glucose', 'HCO3', 'Calcium', 'Potassium', 'BUN']

# Demographic features (static)
DEMOGRAPHIC_FEATURES = ['Age', 'Weight_kg']

# Groupings for medication features
MEDICATION_GROUPS = {
    'vasoactive_drips': [
        'EPINEPHRINE DRIP', 'NOREPINEPHRINE DRIP', 'DOPAMINE DRIP',
        'VASOPRESSIN DRIP FOR HYPOTENSION', 'MILRINONE DRIP', 'PHENYLEPHRINE DRIP'
    ],
    'sedative_drips': [
        'DEXMEDETOMIDINE DRIP', 'LORAZEPAM DRIP'
    ],
    'corticosteroids': [
        'METHYLPREDNISOLONE', 'DEXAMETHASONE', 'HYDROCORTISONE' # Partial match will find variations
    ],
    'insulin_drips': [
        'INSULIN (NOVOLIN R) STANDARD DRIP'
    ]
}

# 5. MODELING PARAMETERS
TARGET_VARIABLE = 'start_antibiotic' # The name of the target column in the outcomes file
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.1
RANDOM_STATE = 42

