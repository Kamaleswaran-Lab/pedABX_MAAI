#!/bin/bash

# This script runs the entire pipeline for the MAAI Antibiotic Prediction project.

# Exit immediately if a command exits with a non-zero status.
set -e

# Step 1: Run the data preprocessing pipeline
echo "--- Running Data Preprocessing ---"
python data_preprocessing/run_preprocessing.py

# Step 2: Train the model
echo "--- Running Model Training ---"
python model_development/train_model.py

# Step 3: Evaluate the model
echo "--- Running Model Evaluation ---"
python model_development/evaluate_model.py

echo "--- Full Pipeline Finished Successfully ---"
