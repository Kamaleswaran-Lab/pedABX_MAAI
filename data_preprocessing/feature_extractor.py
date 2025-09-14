# -*- coding: utf-8 -*-
"""
Core feature extraction and preprocessing logic for the MAAI project.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(config):
    """Loads raw continuous variables, medications, and outcomes data."""
    print("Loading raw data...")
    try:
        df_vars = pd.read_csv(os.path.join(config.RAW_DATA_PATH, config.CONTINUOUS_VARS_FILE))
        df_meds = pd.read_csv(os.path.join(config.RAW_DATA_PATH, config.DISCRETE_MEDS_FILE))
        df_outcomes = pd.read_csv(os.path.join(config.RAW_DATA_PATH, config.PATIENT_OUTCOMES_FILE))
        print("Data loaded successfully.")
        return df_vars, df_meds, df_outcomes
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure your file paths and names are correct in config.py")
        return None, None, None

def preprocess_bp(df):
    """Splits BP column like '120/80' into BP_Systolic and BP_Diastolic."""
    if 'BP' in df.columns:
        print("Processing BP column...")
        # Convert to string and handle non-string values
        bp_str = df['BP'].astype(str)
        # Split and expand into two columns
        bp_split = bp_str.str.split('/', expand=True)
        # Convert to numeric, coercing errors to NaN
        df['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
        df = df.drop(columns=['BP'])
    return df

def resample_to_hourly(df, patient_col='patient_id', time_col='hour'):
    """Resamples data to a consistent hourly grid for each patient."""
    print(f"Resampling {patient_col} data to hourly resolution...")
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col, patient_col])
    
    # Create a full hourly grid for each patient
    grid = df.groupby(patient_col)[time_col].agg(['min', 'max']).reset_index()
    grid_dfs = []
    for _, row in grid.iterrows():
        pat_id = row[patient_col]
        start, end = int(row['min']), int(row['max'])
        grid_dfs.append(pd.DataFrame({'patient_id': pat_id, 'hour': range(start, end + 1)}))
    
    if not grid_dfs:
        return pd.DataFrame()

    full_grid = pd.concat(grid_dfs, ignore_index=True)
    
    # Merge original data onto the grid, using median for duplicates
    df_resampled = pd.merge(full_grid, df, on=['patient_id', 'hour'], how='left')
    return df_resampled

def impute_missing_values(df, patient_col='patient_id'):
    """Imputes missing values using forward-fill followed by population median."""
    print("Imputing missing values...")
    df = df.set_index(['patient_id', 'hour'])
    
    # Patient-wise forward fill
    df = df.groupby(patient_col).ffill()
    
    # Population median for remaining NaNs
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    return df.reset_index()

def create_statistical_features(df, config):
    """Creates rolling window statistical features (mean, std, min, max)."""
    print("Creating rolling statistical features...")
    features_to_agg = config.VITALS_FEATURES + config.LABS_FEATURES
    df_stats = df.copy().set_index(['patient_id', 'hour'])
    
    grouped = df_stats.groupby('patient_id')[features_to_agg]
    
    rolling_window = grouped.rolling(window=config.LOOKBACK_WINDOW_HOURS, min_periods=1)
    
    # Use tqdm for progress bar
    with tqdm(total=4, desc="Aggregating features") as pbar:
        df_mean = rolling_window.mean().add_suffix('_mean')
        pbar.update(1)
        df_std = rolling_window.std().add_suffix('_std')
        pbar.update(1)
        df_min = rolling_window.min().add_suffix('_min')
        pbar.update(1)
        df_max = rolling_window.max().add_suffix('_max')
        pbar.update(1)

    df_stats = pd.concat([df_mean, df_std, df_min, df_max], axis=1)
    df_stats = df_stats.reset_index()
    df_stats.fillna(0, inplace=True) # Fill NaNs from std dev
    return df_stats

def create_medication_features(df_meds, config):
    """Engineers binary flags for medication groups."""
    print("Engineering medication features...")
    df_meds_hourly = resample_to_hourly(df_meds, time_col='hour')

    for group_name, med_list in config.MEDICATION_GROUPS.items():
        pattern = '|'.join(med_list)
        df_meds_hourly[group_name] = df_meds_hourly['medication_name'].str.contains(pattern, case=False, na=False).astype(int)

    med_features_cols = ['patient_id', 'hour'] + list(config.MEDICATION_GROUPS.keys())
    med_features = df_meds_hourly[med_features_cols]
    
    med_features = med_features.groupby(['patient_id', 'hour']).max().reset_index()
    return med_features

def combine_features(df_vars_stats, df_meds_features, df_outcomes):
    """Merges all feature sets into a final dataframe."""
    print("Combining all feature sets...")
    
    df_final = pd.merge(df_vars_stats, df_meds_features, on=['patient_id', 'hour'], how='left')
    df_final = pd.merge(df_final, df_outcomes, on=['patient_id', 'hour'], how='left')
    
    med_group_cols = list(df_meds_features.columns.drop(['patient_id', 'hour']))
    df_final[med_group_cols] = df_final[med_group_cols].fillna(0)
    df_final.dropna(subset=[config.TARGET_VARIABLE], inplace=True)
    
    return df_final

