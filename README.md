= MAAI for Pediatric Antibiotic Prediction =

This repository contains a complete suite of scripts and notebooks for developing a Multi-Agent Artificial Intelligence (MAAI) model to predict the need for antibiotic initiation on an hour-by-hour basis for pediatric patients in the ICU.

The project is structured to be modular and reproducible, building upon the feature extraction methodologies established in the ped-sepsis-prediction-ml repository.

== Project Goal ==

The primary objective is to build a time-series prediction model that leverages hourly EMR data to provide an early warning score for impending infection, helping clinicians decide whether to initiate antibiotics.

The model uses a Multi-Agent architecture, where different "agents" specialize in processing specific physiological data streams (e.g., vitals, labs). Their insights are then integrated by a final decision-making agent to produce a unified risk score.

== Repository Structure ==

<pre>
.
├── data_preprocessing/
│   ├── config.py               # Holds all project constants (paths, feature lists)
│   ├── feature_extractor.py    # Core logic for data cleaning, imputation, and feature engineering
│   └── run_preprocessing.py      # Main script to execute the full preprocessing pipeline
│
├── model_development/
│   ├── maai_model.py           # Defines the Keras/TensorFlow MAAI model architecture
│   ├── train_model.py          # Script for training the model
│   ├── evaluate_model.py       # Script for evaluating the trained model
│   └── utils.py                # Helper functions for data prep, plotting, etc.
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb       # EDA on raw synthetic data
│   └── 02_Model_Development_Walkthrough.ipynb # Step-by-step guide to training and evaluation
│
├── .gitignore
├── requirements.txt
└── README.md
</pre>

== Setup and Installation ==

'''Clone the repository:'''
<source lang="bash">
git clone <repository_url>
cd maai-antibiotic-prediction
</source>

'''Create a virtual environment (recommended):'''
<source lang="bash">
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
</source>

'''Install the required packages:'''
<source lang="bash">
pip install -r requirements.txt
</source>

== How to Run the Project ==

=== Step 1: Configure the Project ===

Before running any scripts, you must edit data_preprocessing/config.py. Update the file paths to point to your raw synthetic datasets and specify where you want the processed data to be saved.

<source lang="python">

data_preprocessing/config.py
RAW_DATA_PATH = 'path/to/your/raw_data/'
PROCESSED_DATA_PATH = 'path/to/save/processed_data/'
MODEL_SAVE_PATH = 'path/to/save/models/'

... and other configurations
</source>

=== Step 2: Run the Preprocessing Pipeline ===

This script will take the raw synthetic data, perform all necessary cleaning, feature engineering, and save the final feature matrix and labels.

<source lang="bash">
python data_preprocessing/run_preprocessing.py
</source>

=== Step 3: Train the MAAI Model ===

This script loads the preprocessed data and trains the MAAI model as defined in model_development/maai_model.py. The trained model and its weights will be saved to the path specified in the config.

<source lang="bash">
python model_development/train_model.py
</source>

=== Step 4: Evaluate the Model ===

After training, run the evaluation script to generate performance metrics and plots (e.g., AUROC, AUPRC, confusion matrix) on the test set.

<source lang="bash">
python model_development/evaluate_model.py
</source>

=== Alternative: Use the Jupyter Notebooks ===

For a more interactive, step-by-step guide through the entire process, open and run the notebooks in the notebooks/ directory. This is highly recommended for understanding the mechanics of the model.

01_Data_Exploration.ipynb: Understand the raw data.
02_Model_Development_Walkthrough.ipynb: Interactively preprocess data, build, train, and evaluate the model.
