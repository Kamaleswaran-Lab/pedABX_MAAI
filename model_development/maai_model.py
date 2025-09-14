# -*- coding: utf-8 -*-
"""
Defines the MAAI model architecture using TensorFlow/Keras.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_maai_model(n_features_vitals, n_features_labs, n_features_meds, lookback_window):
    """
    Builds the Multi-Agent Artificial Intelligence (MAAI) model.

    The architecture consists of three specialized 'agents' (LSTMs) that process
    different data streams, and a final 'decision agent' (Dense layers) that
    integrates their outputs.
    """
    # --- Agent 1: Cardiovascular & Respiratory Agent (Vitals) ---
    input_vitals = Input(shape=(lookback_window, n_features_vitals), name='vitals_input')
    lstm_vitals_1 = LSTM(32, return_sequences=True, name='vitals_lstm_1')(input_vitals)
    bn_vitals = BatchNormalization(name='vitals_bn')(lstm_vitals_1)
    lstm_vitals_2 = LSTM(16, name='vitals_lstm_2')(bn_vitals)
    dropout_vitals = Dropout(0.3, name='vitals_dropout')(lstm_vitals_2)

    # --- Agent 2: Metabolic & Inflammatory Agent (Labs) ---
    input_labs = Input(shape=(lookback_window, n_features_labs), name='labs_input')
    lstm_labs_1 = LSTM(32, return_sequences=True, name='labs_lstm_1')(input_labs)
    bn_labs = BatchNormalization(name='labs_bn')(lstm_labs_1)
    lstm_labs_2 = LSTM(16, name='labs_lstm_2')(bn_labs)
    dropout_labs = Dropout(0.3, name='labs_dropout')(lstm_labs_2)

    # --- Agent 3: Intervention Agent (Medications) ---
    input_meds = Input(shape=(lookback_window, n_features_meds), name='meds_input')
    lstm_meds = LSTM(8, name='meds_lstm')(input_meds)
    dropout_meds = Dropout(0.2, name='meds_dropout')(lstm_meds)

    # --- Decision Agent: Integrates insights from all agents ---
    concatenated_features = Concatenate(name='concatenate_agents')(
        [dropout_vitals, dropout_labs, dropout_meds]
    )

    dense_1 = Dense(64, activation='relu', name='decision_dense_1')(concatenated_features)
    bn_final = BatchNormalization(name='final_bn')(dense_1)
    dropout_final = Dropout(0.5, name='final_dropout')(bn_final)
    output = Dense(1, activation='sigmoid', name='output_prediction')(dropout_final)
    
    # Compile the final model
    model = Model(
        inputs=[input_vitals, input_labs, input_meds],
        outputs=output,
        name='MAAI_Antibiotic_Predictor'
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auroc'),
            tf.keras.metrics.AUC(curve='PR', name='auprc'),
            'accuracy'
        ]
    )

    return model

if __name__ == '__main__':
    # Example of how to build and view the model summary
    # These numbers are placeholders and will be determined by the preprocessed data
    n_vitals_features = 15 * 4 # 15 vitals * 4 stats (mean, std, min, max)
    n_labs_features = 10 * 4 # 10 labs * 4 stats
    n_meds_features = 4 # 4 medication groups
    lookback = 12

    model = build_maai_model(n_vitals_features, n_labs_features, n_meds_features, lookback)
    model.summary()

