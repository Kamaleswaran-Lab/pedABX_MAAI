# -*- coding: utf-8 -*-
"""
Helper functions for model training, evaluation, and data preparation.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report

def get_feature_sets(config):
    """Generates lists of feature names for each agent based on config."""
    stats = ['_mean', '_std', '_min', '_max']
    vitals_features = [f"{feat}{s}" for feat in config.VITALS_FEATURES for s in stats]
    labs_features = [f"{feat}{s}" for feat in config.LABS_FEATURES for s in stats]
    meds_features = list(config.MEDICATION_GROUPS.keys())
    return vitals_features, labs_features, meds_features

def create_sequences(df, vitals_cols, labs_cols, meds_cols, target_col, lookback_window):
    """
    Reshapes the flat dataframe into sequences for the LSTM model.
    Output: Tuple of ( (X_vitals, X_labs, X_meds), y )
    """
    print("Creating sequences for LSTM...")
    X_vitals, X_labs, X_meds, y = [], [], [], []
    
    patient_groups = df.groupby('patient_id')
    
    for _, group in tqdm(patient_groups, desc="Processing patients"):
        vitals = group[vitals_cols].values
        labs = group[labs_cols].values
        meds = group[meds_cols].values
        labels = group[target_col].values
        
        for i in range(len(group) - lookback_window):
            X_vitals.append(vitals[i : i + lookback_window])
            X_labs.append(labs[i : i + lookback_window])
            X_meds.append(meds[i : i + lookback_window])
            y.append(labels[i + lookback_window - 1]) # Label corresponds to the end of the window
            
    return (np.array(X_vitals), np.array(X_labs), np.array(X_meds)), np.array(y)

def plot_training_history(history, save_path):
    """Plots AUROC and loss curves from model training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation AUROC values
    ax1.plot(history.history['auroc'])
    ax1.plot(history.history['val_auroc'])
    ax1.set_title('Model AUROC')
    ax1.set_ylabel('AUROC')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_pred, save_path):
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.show()

def plot_pr_curve(y_true, y_pred, save_path):
    """Plots the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    plt.figure()
    plt.step(recall, precision, where='post', label=f'AP = {avg_precision:0.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(save_path)
    plt.show()
