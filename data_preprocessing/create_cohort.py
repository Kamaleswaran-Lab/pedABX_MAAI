# -*- coding: utf-8 -*-
"""
New script to create the patient cohort based on different sepsis criteria.
"""
import pandas as pd
import argparse
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import config


def define_sirs_cohort(raw_data_path):
    """
    Applies SIRS criteria to define the cohort.
    This is a placeholder function. You should implement the actual SIRS logic here.
    """
    print("Defining cohort based on SIRS criteria...")
    # Placeholder logic: Assumes a 'raw_patients.csv' file exists
    # and filters based on some condition.
    try:
        df_patients = pd.read_csv(os.path.join(raw_data_path, 'raw_patients.csv'))
        # Example: select patients older than 18
        cohort_df = df_patients[df_patients['age'] > 18]
        return cohort_df
    except FileNotFoundError:
        print("Warning: 'raw_patients.csv' not found. Returning an empty SIRS cohort.")
        return pd.DataFrame()


def define_psofa_cohort(raw_data_path):
    """
    Applies pSOFA criteria to define the cohort.
    Placeholder function. Implement your pSOFA logic here.
    """
    print("Defining cohort based on pSOFA criteria...")
    try:
        df_patients = pd.read_csv(os.path.join(raw_data_path, 'raw_patients.csv'))
        # Example: select patients with a specific condition
        cohort_df = df_patients[df_patients['condition'] == 'some_condition']
        return cohort_df
    except FileNotFoundError:
        print("Warning: 'raw_patients.csv' not found. Returning an empty pSOFA cohort.")
        return pd.DataFrame()


def define_phoenix_cohort(raw_data_path):
    """
    Applies Phoenix criteria to define the cohort.
    Placeholder function. Implement your Phoenix logic here.
    """
    print("Defining cohort based on Phoenix criteria...")
    try:
        # For this example, we'll assume the phoenix cohort is simply all patients
        # from the outcomes file, as it's the most inclusive.
        df_outcomes = pd.read_csv(os.path.join(raw_data_path, config.OUTCOMES_FILE))
        cohort_df = df_outcomes[['patient_id']].drop_duplicates()
        return cohort_df
    except FileNotFoundError:
        print(f"Warning: {config.OUTCOMES_FILE} not found. Returning an empty Phoenix cohort.")
        return pd.DataFrame()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a patient cohort based on specified criteria.")
    parser.add_argument('--criteria', choices=['sirs', 'psofa', 'phoenix'], required=True, help="The sepsis criteria to use for cohort definition.")
    args = parser.parse_args()

    if args.criteria == 'sirs':
        cohort_df = define_sirs_cohort(config.RAW_DATA_PATH)
    elif args.criteria == 'psofa':
        cohort_df = define_psofa_cohort(config.RAW_DATA_PATH)
    elif args.criteria == 'phoenix':
        cohort_df = define_phoenix_cohort(config.RAW_DATA_PATH)

    if not cohort_df.empty:
        output_path = os.path.join(config.PROCESSED_DATA_PATH, config.COHORT_FILE)
        cohort_df.to_parquet(output_path, index=False)
        print(f"Cohort of size {len(cohort_df)} created and saved to {output_path}")
    else:
        print("No cohort was created.")
