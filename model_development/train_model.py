# -*- coding: utf-8 -*-
"""
Script to train the MAAI model.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import config
from model_development.maai_model import build_maai_model
from model_development.utils import get_feature_sets, create_sequences, plot_training_history

def main():
    print("--- Starting Model Training ---")
    
    # 1. Load Data
    data_path = os.path.join(config.PROCESSED_DATA_PATH, "processed_feature_matrix.parquet")
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        print("Please run the preprocessing pipeline first: python data_preprocessing/run_preprocessing.py")
        return
    df = pd.read_parquet(data_path)
    print(f"Loaded data with shape: {df.shape}")

    # 2. Scale features
    vitals_cols, labs_cols, meds_cols = get_feature_sets(config)
    feature_cols = vitals_cols + labs_cols # Meds are binary, no scaling needed
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Features scaled.")

    # 3. Create sequences
    (X_vitals, X_labs, X_meds), y = create_sequences(
        df, vitals_cols, labs_cols, meds_cols, config.TARGET_VARIABLE, config.LOOKBACK_WINDOW_HOURS
    )

    # 4. Split data
    # Create a dummy array for splitting since we need to split all three X arrays consistently
    indices = np.arange(X_vitals.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    
    X_train_v, X_test_v = X_vitals[train_indices], X_vitals[test_indices]
    X_train_l, X_test_l = X_labs[train_indices], X_labs[test_indices]
    X_train_m, X_test_m = X_meds[train_indices], X_meds[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print("Data split into training and testing sets.")
    
    # 5. Build Model
    model = build_maai_model(
        n_features_vitals=X_train_v.shape[2],
        n_features_labs=X_train_l.shape[2],
        n_features_meds=X_train_m.shape[2],
        lookback_window=config.LOOKBACK_WINDOW_HOURS
    )
    model.summary()
    
    # 6. Train Model
    # Calculate class weights to handle imbalance
    neg, pos = np.bincount(y_train)
    total = neg + pos
    class_weight = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
    print(f"Class weights: {class_weight}")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auprc', mode='max', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_SAVE_PATH, 'maai_model.keras'),
            save_best_only=True,
            monitor='val_auprc',
            mode='max'
        )
    ]
    
    history = model.fit(
        [X_train_v, X_train_l, X_train_m], y_train,
        validation_split=config.VALIDATION_SPLIT_SIZE,
        epochs=100,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks
    )
    
    # 7. Save results
    plot_training_history(history, os.path.join(config.RESULTS_PATH, 'training_history.png'))
    np.save(os.path.join(config.PROCESSED_DATA_PATH, 'test_indices.npy'), test_indices)

    print("--- Model Training Finished ---")

if __name__ == '__main__':
    main()

