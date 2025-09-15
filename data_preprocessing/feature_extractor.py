# -*- coding: utf-8 -*-
"""
Core feature extraction and preprocessing logic for the MAAI project.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import skew, kurtosis

def load_data(config):
    """Loads raw continuous variables, medications, and outcomes data."""
    print("Loading raw data...")
    try:
        df_vars = pd.read_parquet(os.path.join(config.RAW_DATA_PATH, config.RAW_VARIABLES_FILE))
        df_meds = pd.read_parquet(os.path.join(config.RAW_DATA_PATH, config.RAW_MEDS_FILE))
        # Outcomes are generated during cohort creation and feature engineering
        df_outcomes = None
        print("Data loaded successfully.")
        return df_vars, df_meds, df_outcomes
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure your file paths and names are correct in config.py")
        return None, None, None

def preprocess_bp(df):
    """Splits BP column and renames variables."""
    print("Preprocessing BP and renaming variables...")
    # Split BP into systolic and diastolic
    if 'BP' in df['variable_name'].unique():
        sysbp = df[df['variable_name'] == 'BP'].copy()
        sysbp['variable_name'] = 'bp_sys'
        sysbp['value'] = sysbp['value'].apply(lambda x: str(x).split('/')[0])
        
        df.loc[df['variable_name'] == 'BP', 'variable_name'] = 'bp_dias'
        df.loc[df['variable_name'] == 'bp_dias', 'value'] = df.loc[df['variable_name'] == 'bp_dias', 'value'].apply(lambda x: str(x).split('/')[1] if '/' in str(x) else np.nan)
        df = pd.concat([df, sysbp])

    # Rename variables for consistency
    name_map = {
        'Weight': 'weight', 'Code Sheet Weight (kg)': 'weight', 'Pulse': 'pulse', 'MAP': 'map',
        'ABP MAP': 'map', 'ART MAP': 'map', 'Resp': 'resp', 'SpO2': 'spo2', 'Temp': 'temp',
        'FiO2 (%)': 'fio2', 'PaO2/FiO2 (Calculated)': 'pao2_fio2', 'Coma Scale Total': 'coma_scale_total',
        'Oxygen Flow (lpm)': 'o2_flow', 'POC pH': 'ph', 'POC PO2': 'po2', 'POC PCO2': 'pco2',
        'POTASSIUM': 'potassium', 'SODIUM': 'sodium', 'CHLORIDE': 'chloride', 'POC GLUCOSE': 'glucose',
        'GLUCOSE': 'glucose', 'BUN': 'bun', 'CREATININE': 'creatinine', 'CALCIUM': 'calcium',
        'POC CALCIUM IONIZED': 'calcium_ionized', 'CO2': 'co2', 'HEMOGLOBIN': 'hemoglobin',
        'BILIRUBIN TOTAL': 'bilirubin_total', 'ALBUMIN': 'albumin', 'WBC': 'wbc',
        'PLATELETS': 'platelets', 'PTT': 'ptt', 'ARTERIAL BASE EXCESS': 'base_excess',
        'VENOUS BASE EXCESS': 'base_excess', 'CAP BASE EXCESS': 'base_excess',
        'ART BASE DEFICIT': 'base_deficit', 'VENOUS BASE DEFICIT': 'base_deficit',
        'CAP BASE DEFICIT': 'base_deficit', 'HCO3': 'bicarbonate', 'LACTIC ACID': 'lactic_acid',
        'POC LACTIC ACID': 'lactic_acid', 'LACTIC ACID WHOLE BLOOD': 'lactic_acid',
        'BAND NEUTROPHILS % (MANUAL)': 'band_neutrophils', 'ALT (SGPT)': 'alt', 'AST (SGOT)': 'ast',
        'INT NORM RATIO': 'inr', 'PROTIME': 'pt', 'Volume Infused (mL)': 'vol_infused',
        'Urine (mL)': 'urine'
    }
    df['variable_name'] = df['variable_name'].replace(name_map)
    return df

def resample_to_hourly(df, patient_col, time_col='recorded_time'):
    """Pivots and resamples data to a consistent hourly grid for each patient."""
    print(f"Resampling data to hourly resolution for each {patient_col}...")
    
    # Pivot the data
    df = pd.pivot_table(df, values='value', index=[patient_col, time_col], columns=['variable_name'], aggfunc='first')
    df.reset_index(inplace=True)
    
    # Create an hourly grid
    df['hour'] = df.groupby(patient_col)[time_col].transform(lambda x: (x - x.min()).dt.total_seconds() // 3600)
    
    # Resample to hourly by taking the median of values within each hour
    df_resampled = df.groupby([patient_col, 'hour']).median().reset_index()
    return df_resampled

def impute_missing_values(df, patient_col):
    """Imputes missing values using forward-fill followed by population median."""
    print("Imputing missing values...")
    df = df.set_index([patient_col, 'hour'])

    # Patient-wise forward fill with a limit
    df = df.groupby(patient_col).ffill(limit=12) 

    # Population median for remaining NaNs
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    return df.reset_index()

def create_statistical_features(df, config):
    """Creates rolling window statistical features."""
    print("Creating rolling statistical features...")
    
    features_to_agg = [feat for feat in config.VITALS_FEATURES if feat in df.columns] + \
                      [feat for feat in config.LABS_FEATURES if feat in df.columns]
    
    df_stats = df.copy().set_index([config.PATIENT_ID_COL, 'hour'])
    
    grouped = df_stats.groupby(config.PATIENT_ID_COL)[features_to_agg]
    
    rolling_window = grouped.rolling(window=config.LOOKBACK_WINDOW_HOURS, min_periods=1)
    
    with tqdm(total=6, desc="Aggregating features") as pbar:
        df_mean = rolling_window.mean().add_suffix('_mean')
        pbar.update(1)
        df_std = rolling_window.std().add_suffix('_std')
        pbar.update(1)
        df_min = rolling_window.min().add_suffix('_min')
        pbar.update(1)
        df_max = rolling_window.max().add_suffix('_max')
        pbar.update(1)
        df_skew = rolling_window.skew().add_suffix('_skew')
        pbar.update(1)
        df_kurt = rolling_window.kurt().add_suffix('_kurtosis')
        pbar.update(1)

    df_stats = pd.concat([df, df_mean, df_std, df_min, df_max, df_skew, df_kurt], axis=1)
    df_stats = df_stats.reset_index()
    df_stats.fillna(0, inplace=True) # Fill NaNs from std dev, skew, kurtosis
    return df_stats

def create_medication_features(df_meds, config):
    """Engineers binary flags for medication groups."""
    print("Engineering medication features...")
    df_meds['hour'] = df_meds.groupby(config.PATIENT_ID_COL)['mar_time'].transform(lambda x: (x - x.min()).dt.total_seconds() // 3600)
    
    for group_name, med_list in config.MEDICATION_GROUPS.items():
        pattern = '|'.join(med_list)
        df_meds[group_name] = df_meds['med'].str.contains(pattern, case=False, na=False).astype(int)

    med_features_cols = [config.PATIENT_ID_COL, 'hour'] + list(config.MEDICATION_GROUPS.keys())
    med_features = df_meds[med_features_cols]
    
    med_features = med_features.groupby([config.PATIENT_ID_COL, 'hour']).max().reset_index()
    return med_features

def combine_features(df_vars_stats, df_meds_features, df_outcomes, config):
    """Merges all feature sets and the outcome into a final dataframe."""
    print("Combining all feature sets...")
    
    df_final = pd.merge(df_vars_stats, df_meds_features, on=[config.PATIENT_ID_COL, 'hour'], how='left')
    
    # Load and merge the cohort file which contains the sepsis label
    cohort_path = os.path.join(config.PROCESSED_DATA_PATH, config.COHORT_FILE)
    if os.path.exists(cohort_path):
        df_cohort = pd.read_parquet(cohort_path)
        df_final = pd.merge(df_final, df_cohort, on=config.CSN_COL, how='left')
        df_final[config.TARGET_VARIABLE] = df_final['inf_time'].notna().astype(int)
        df_final = df_final.drop(columns=['inf_time', 'rel_day_inf'])
    else:
        # If no cohort file, assume no sepsis label (for testing purposes)
        df_final[config.TARGET_VARIABLE] = 0

    med_group_cols = list(config.MEDICATION_GROUPS.keys())
    df_final[med_group_cols] = df_final[med_group_cols].fillna(0)
    
    return df_final
