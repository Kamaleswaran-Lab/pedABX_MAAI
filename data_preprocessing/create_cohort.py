# -*- coding: utf-8 -*-
"""
Creates the patient cohort based on the 'infection + Phoenix' criteria.
"""
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import config

def define_phoenix_cohort(raw_data_path, processed_data_path):
    """
    Applies 'infection + Phoenix' criteria to define the cohort.
    This implementation is based on the logic in 'screening_methods/inf_phoenix.ipynb'.
    """
    print("Defining cohort based on Infection + Phoenix criteria...")

    # Load necessary raw data
    try:
        cultures = pd.read_parquet(os.path.join(raw_data_path, 'DR15269_LABsAndPFTs.parquet.gzip'))
        dept = pd.read_parquet(os.path.join(raw_data_path, 'TAB2_Encounter_Departments.parquet.gzip'))
        antiinf = pd.read_parquet(os.path.join(raw_data_path, 'antiinf_meds.parquet.gzip'))
    except FileNotFoundError as e:
        print(f"Error: A required raw data file is missing. {e}")
        return pd.DataFrame()

    # Process cultures data
    cultures.columns = ['patid', 'mrn', 'csn', 'order_time', 'result_time', 'procedure', 'component', 'result']
    cultures[['order_time', 'result_time']] = cultures[['order_time', 'result_time']].apply(pd.to_datetime)
    cultures['csn'] = cultures['csn'].astype(int)
    cultures = cultures[cultures['procedure'].str.contains('culture', case=False, na=False)]

    # Process department data
    dept = dept[['Encounter CSN', 'Hosp_Admission']]
    dept.columns = ['csn', 'hosp_admission']
    dept['csn'] = dept['csn'].astype(int)
    dept['hosp_admission'] = dept['hosp_admission'].apply(pd.to_datetime)
    dept.drop_duplicates(inplace=True)

    # Add hospital admission to cultures and filter by relative day
    cultures = cultures.merge(dept, how='left', on='csn')
    cultures['rel_day_cult'] = np.ceil((cultures['order_time'] - cultures['hosp_admission']) / pd.Timedelta('1 day'))
    cultures = cultures[cultures['rel_day_cult'] <= 7]
    cultures = cultures[['csn', 'order_time', 'rel_day_cult']].drop_duplicates()

    # Process anti-infective meds
    antiinf['mar_time'] = antiinf['mar_time'].apply(pd.to_datetime)
    antiinf['csn'] = antiinf['csn'].astype(int)
    antiinf = antiinf.merge(dept, how='left', on='csn')
    antiinf['rel_day_ant'] = np.ceil((antiinf['mar_time'] - antiinf['hosp_admission']) / pd.Timedelta('1 day'))
    antiinf = antiinf[antiinf['rel_day_ant'] <= 7]

    # Combine with culture data to define infection time
    antiinf = antiinf.merge(cultures, how='left', on='csn')
    antiinf = antiinf[(antiinf['rel_day_ant'] == 1) | (antiinf['rel_day_ant'] == antiinf['rel_day_cult'])]
    antiinf['inf_time'] = antiinf[['mar_time', 'order_time']].min(axis=1)
    antiinf = antiinf[['csn', 'rel_day_ant', 'inf_time']]
    antiinf.columns = ['csn', 'rel_day_inf', 'inf_time']
    antiinf.drop_duplicates(inplace=True)
    antiinf = antiinf.sort_values(by=['csn', 'rel_day_inf', 'inf_time']).groupby(['csn', 'rel_day_inf'], as_index=False).first()

    # This 'antiinf' dataframe now represents the cohort with suspected infection.
    # The actual Phoenix score calculation would require the full preprocessed feature set.
    # For cohort creation, we will proceed with this suspected infection cohort.
    # The final sepsis label will be determined during the feature engineering phase.

    cohort_df = antiinf[[config.CSN_COL]].drop_duplicates()
    return cohort_df


def create_cohort(criteria, raw_data_path, processed_data_path):
    """
    Creates and saves the patient cohort.
    """
    if criteria == 'phoenix':
        cohort_df = define_phoenix_cohort(raw_data_path, processed_data_path)
    else:
        raise ValueError(f"Criteria '{criteria}' is not implemented. This project focuses on the 'phoenix' criteria.")

    if not cohort_df.empty:
        output_path = os.path.join(processed_data_path, config.COHORT_FILE)
        cohort_df.to_parquet(output_path, index=False)
        print(f"Cohort of size {len(cohort_df)} created and saved to {output_path}")
    else:
        print("No cohort was created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a patient cohort based on specified criteria.")
    parser.add_argument('--criteria', choices=['phoenix'], default='phoenix', help="The sepsis criteria to use for cohort definition.")
    args = parser.parse_args()
    create_cohort(args.criteria, config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
