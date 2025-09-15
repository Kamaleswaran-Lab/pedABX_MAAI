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
ANTIINF_MEDS_FILE = 'antiinf_meds.parquet.gzip'
DEPARTMENTS_FILE = 'TAB2_Encounter_Departments.parquet.gzip'
DEMOGRAPHICS_FILE = 'TAB1_Patients.parquet.gzip'
PROBLEM_LIST_FILE = 'TAB3_Problem_List.parquet.gzip'
HOSP_DIAG_FILE = 'TAB4_Hospital_Diagnoses.parquet.gzip'
ADM_DIAG_FILE = 'TAB5_Admitting_Diagnoses.parquet.gzip'
MV_INDICATORS_FILE = 'mv_indicators_raw.parquet.gzip'
APPARATUS_TYPE_FILE = '../files/apparatus_type_ann.csv' # Assuming a 'files' dir at the root
ECMO_FILE = '../files/ECMO_database_2010_2022.csv'

COHORT_FILE = 'cohort_inf_phoenix.csv'
PROCESSED_FEATURE_MATRIX_FILE = "processed_feature_matrix.parquet"

# --- PREPROCESSING PARAMETERS ---
TIME_RESOLUTION_HOURS = 1
LOOKBACK_WINDOW_HOURS = 12
PATIENT_ID_COL = 'patid'
CSN_COL = 'csn'

# --- FEATURE LISTS (derived from ped-sepsis-prediction-ml and clinical review) ---
VITALS_FEATURES = [
    'weight', 'pulse', 'map', 'bp_sys', 'bp_dias', 'resp', 'spo2', 'temp', 'fio2',
    'pao2_fio2', 'o2_flow', 'coma_scale_total', 'pupil_left_size',
    'pupil_right_size', 'urine', 'vol_infused', 'cap_refill'
]

LABS_FEATURES = [
    'ph', 'po2', 'pco2', 'potassium', 'sodium', 'chloride', 'glucose', 'bun',
    'creatinine', 'calcium', 'calcium_ionized', 'co2', 'hemoglobin',
    'bilirubin_total', 'albumin', 'wbc', 'platelets', 'ptt', 'base_excess',
    'bicarbonate', 'lactic_acid', 'base_deficit', 'band_neutrophils', 'alt',
    'ast', 'pt', 'inr', 'ddimer', 'fibrinogen', 'lymphocytes', 'neutrophils'
]

# The keys are the flag names, and the values are the keywords used to identify them
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

DIAGNOSIS_MAP = {
    'sepsis_septicemia_diag': ['sepsis', 'septicemia'],
    'septic_shock_diag': ['septic shock'],
    'sickle_cell_diag': ['sickle'],
    'dka_diag': ['diabetes', 'ketoacidosis'],
    'asthmaticus_diag': ['asthmaticus'],
    'kidney_failure_diag': ['kidney disease', 'kidney failure']
}

OTHER_FLAGS = [
    'had_cultures_ordered', 'prev_hosp', 'prev_hosp_prev_year'
]

# --- PEDIATRIC RATIOS AND SCORES ---
PEDIATRIC_RATIOS_AND_SCORES = [
    'sf_ratio', 'shock_index', 'age_adjusted_shock_index',
    'neutrophil_lymphocyte_ratio', 'delta_anion_gap'
]

# --- MODELING PARAMETERS ---
TARGET_VARIABLE = 'sepsis_label'
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.1
RANDOM_STATE = 42
