# -*- coding: utf-8 -*-
"""
Script to evaluate the trained MAAI model on the test set.
"""
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import config
from model_development.utils import get_feature_sets, create_sequences, plot_roc_curve, plot_pr_curve

def main():
    print("--- Starting Model Evaluation ---")

    # 1. Load Model
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'maai_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # 2. Load and prepare test data
    data_path = os.path.join(config.PROCESSED_DATA_PATH, config.PROCESSED_FEATURE_MATRIX_FILE)
    df = pd.read_parquet(data_path)
    vitals_cols, labs_cols, meds_cols = get_feature_sets(config)

    (X_vitals, X_labs, X_meds), y = create_sequences(
        df, vitals_cols, labs_cols, meds_cols, config.TARGET_VARIABLE, config.LOOKBACK_WINDOW_HOURS
    )

    test_indices_path = os.path.join(config.PROCESSED_DATA_PATH, 'test_indices.npy')
    if not os.path.exists(test_indices_path):
        print("Error: Test set indices not found. Please run training script first.")
        return
    test_indices = np.load(test_indices_path)

    X_test_v, X_test_l, X_test_m = X_vitals[test_indices], X_labs[test_indices], X_meds[test_indices]
    y_test = y[test_indices]

    # 3. Make Predictions
    y_pred_proba = model.predict([X_test_v, X_test_l, X_test_m]).ravel()

    # 4. Generate and Save Plots
    plot_roc_curve(y_test, y_pred_proba, os.path.join(config.RESULTS_PATH, 'roc_curve.png'))
    plot_pr_curve(y_test, y_pred_proba, os.path.join(config.RESULTS_PATH, 'pr_curve.png'))

    # 5. Print Classification Report
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    report = classification_report(y_test, y_pred_class)
    print("\nClassification Report (Threshold = 0.5):")
    print(report)
    with open(os.path.join(config.RESULTS_PATH, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No ABX', 'Start ABX'], yticklabels=['No ABX', 'Start ABX'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(config.RESULTS_PATH, 'confusion_matrix.png'))
    plt.show()

    print(f"--- Evaluation Finished. Results saved to {config.RESULTS_PATH} ---")

if __name__ == '__main__':
    main()
