# -*- coding: utf-8 -*-
"""
Configuration file for the MAAI Antibiotic Prediction project.
Holds all project constants, paths, and feature lists.
"""
import os

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TODO: Update this to the absolute path of your raw data directory
RAW_DATA_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'processed_data')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')

# --- DATA FILENAMES ---
RAW_VARIABLES_FILE = 'variables.pkl'
RAW_MEDS_FILE = 'filtered_meds.parquet.gzip'
COHORT_FILE = 'cohort_inf_phoenix.csv'
DIAGNOSES_FILE = 'flagged_adm_diag.parquet.gzip'
PREV_HOSP_FILE = 'previous_hosp.parquet.gzip'
CULTURES_FILE = 'filtered_labs.parquet.gzip'
PROCESSED_FEATURE_MATRIX_FILE = "processed_feature_matrix.parquet"

# --- PREPROCESSING PARAMETERS ---
TIME_RESOLUTION_HOURS = 1
LOOKBACK_WINDOW_HOURS = 12
PATIENT_ID_COL = 'patid'
CSN_COL = 'csn'

# --- FEATURE LISTS (derived from ped-sepsis-prediction-ml) ---
VITALS_FEATURES = [
    'weight', 'pulse', 'map', 'bp_sys', 'bp_dias', 'resp', 'spo2', 'temp', 'fio2',
    'pao2_fio2', 'o2_flow', 'coma_scale_total', 'pupil_left_size',
    'pupil_right_size', 'urine', 'vol_infused'
]

LABS_FEATURES = [
    'ph', 'po2', 'pco2', 'potassium', 'sodium', 'chloride', 'glucose', 'bun',
    'creatinine', 'calcium', 'calcium_ionized', 'co2', 'hemoglobin',
    'bilirubin_total', 'albumin', 'wbc', 'platelets', 'ptt', 'base_excess',
    'bicarbonate', 'lactic_acid', 'base_deficit', 'band_neutrophils', 'alt',
    'ast', 'pt', 'inr'
]

# The keys of this dictionary ARE the variable names for the medication flags.
MEDICATION_GROUPS = {
    'on_asthma_meds': [
        'albuterol', 'dexamethasone', 'epinephrine', 'methylprednisolone', 
        'magnesium sulfate', 'terbutaline', 'levalbuterol', 'xopenex'
    ],
    'on_seizure_meds': [
        'lorazepam', 'levetiracetam', 'fosphenytoin', 'phenobarbital'
    ],
    'on_vasopressors': [
        'epinephrine', 'phenylephrine', 'dopamine', 'norepinephrine', 'vasopressin'
    ],
    'on_antiinf_meds': [
        'acyclovir', 'amikacin', 'amoxicillin', 'amphotericin', 'ampicillin', 
        'azithromycin', 'aztreonam', 'cefazolin', 'cefdinir', 'cefepime', 
        'cefixime', 'cefotaxime', 'cefotetan', 'cefoxitin', 'cefprozil', 
        'ceftazidime', 'ceftriaxone', 'cefuroxime', 'cephalexin', 'cidofovir', 
        'ciprofloxacin', 'clarithromycin', 'clindamycin', 'dapsone', 'daptomycin', 
        'doxycycline', 'ertapenem', 'ethambutol', 'fluconazole', 'foscarnet', 
        'ganciclovir', 'gentamicin', 'imipenem', 'isoniazid', 'levofloxacin', 
        'linezolid', 'meropenem', 'metronidazole', 'micafungin', 'moxifloxacin', 
        'nitrofurantoin', 'oseltamivir', 'oxacillin', 'penicillin', 'piperacillin', 
        'posaconazole', 'rifampin', 'sulfamethoxazole', 'ticarcillin', 'tobramycin', 
        'vancomycin', 'voriconazole'
    ],
    'on_insulin': ['insulin']
}

DIAGNOSIS_FLAGS = [
    'sepsis_septicemia_diag', 'septic_shock_diag', 'sickle_cell_diag',
    'dka_diag', 'asthmaticus_diag'
]

OTHER_FLAGS = [
    'had_cultures_ordered', 'prev_hosp', 'prev_hosp_prev_year'
]

# --- MODELING PARAMETERS ---
TARGET_VARIABLE = 'sepsis_label'
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.1
RANDOM_STATE = 42
