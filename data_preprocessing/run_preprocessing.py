# -*- coding: utf-8 -*-
"""
Main script to execute the full data preprocessing pipeline.
"""
import os
import sys
import pandas as pd

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import config
from data_preprocessing import feature_extractor as fe

def main():
    """Orchestrates the data preprocessing pipeline."""
    print("--- Starting Data Preprocessing Pipeline ---")

    # Create output directories if they don't exist
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    # 1. Load Data
    df_vars, df_meds, df_outcomes = fe.load_data(config)
    if df_vars is None:
        return

    # 2. Preprocess Continuous Variables
    df_vars = fe.preprocess_bp(df_vars)
    df_vars_hourly = fe.resample_to_hourly(df_vars)
    df_vars_imputed = fe.impute_missing_values(df_vars_hourly)
    
    # 3. Create Statistical Features (the main feature set)
    df_vars_stats = fe.create_statistical_features(df_vars_imputed, config)
    
    # 4. Preprocess Medication Data
    df_meds_features = fe.create_medication_features(df_meds, config)

    # 5. Combine all features
    final_df = fe.combine_features(df_vars_stats, df_meds_features, df_outcomes)
    
    # 6. Save Processed Data
    output_filepath = os.path.join(config.PROCESSED_DATA_PATH, "processed_feature_matrix.parquet")
    print(f"Saving final feature matrix to {output_filepath}...")
    final_df.to_parquet(output_filepath, index=False)
    
    print("--- Preprocessing Pipeline Finished Successfully ---")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Saved to {output_filepath}")

if __name__ == '__main__':
    main()

