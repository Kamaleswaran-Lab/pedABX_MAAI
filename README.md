# MAAI for Pediatric Antibiotic Prediction

This repository contains a complete suite of scripts and notebooks for developing a Multi-Agent Artificial Intelligence (MAAI) model to predict the need for antibiotic initiation on an hour-by-hour basis for pediatric patients in the ICU.

The project is structured to be modular and reproducible, building upon the feature extraction methodologies established in the `ped-sepsis-prediction-ml` repository.

## Project Goal

The primary objective is to build a time-series prediction model that leverages hourly EMR data to provide an early warning score for impending infection, helping clinicians decide whether to initiate antibiotics.

The model uses a Multi-Agent architecture, where different "agents" specialize in processing specific physiological data streams (e.g., vitals, labs). Their insights are then integrated by a final decision-making agent to produce a unified risk score.

## Repository Structure

```
.
├── data_preprocessing/
│   ├── config.py               # Holds all project constants (paths, feature lists)
│   ├── create_cohort.py        # New script to create the patient cohort based on clinical criteria
│   ├── feature_extractor.py    # Core logic for data cleaning, imputation, and feature engineering
│   └── run_preprocessing.py    # Main script to execute the full preprocessing pipeline
│
├── model_development/
│   ├── maai_model.py           # Defines the Keras/TensorFlow MAAI model architecture
│   ├── train_model.py          # Script for training the model
│   ├── evaluate_model.py       # Script for evaluating the trained model
│   └── utils.py                # Helper functions for data prep, plotting, etc.
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Model_Development_Walkthrough.ipynb
│
├── synthetic_data/
│   ├── raw/                    # Directory for your raw input CSV files
│   └── processed/              # Directory for processed data outputs
│
├── .gitignore
├── requirements.txt
├── README.md
└── run_project.sh              # New master script to run the entire pipeline

```




## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Kamaleswaran-Lab/pedABX_MAAI](https://github.com/Kamaleswaran-Lab/pedABX_MAAI)
    cd pedABX_MAAI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Project

### Step 1: Configure the Project

Before running any scripts, you must edit `data_preprocessing/config.py`. Update the file paths to point to your raw datasets and specify where you want processed data, models, and results to be saved.

```python
# data_preprocessing/config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'processed')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
# ... and other configurations
```

### Step 2: Run the Preprocessing Pipeline

This script will take the raw synthetic data, perform all necessary cleaning, feature engineering, and save the final feature matrix and labels.

```bash
chmod +x run_project.sh
```

Then, run the pipeline:

```bash
./run_project.sh
```

## Alternative: Run Steps Manually
If you prefer more granular control, you can run each major step of the pipeline individually.

Run the Preprocessing Pipeline:
This script will first call create_cohort.py to define the patient cohort and then run all necessary cleaning and feature engineering steps.

```bash
python data_preprocessing/run_preprocessing.py
```
Note: You can change the cohort criteria (e.g., 'sirs', 'psofa', 'phoenix') inside the run_preprocessing.py script.

Train the MAAI Model:
This script loads the preprocessed data and trains the model. The trained model will be saved to the path specified in the config.

```bash
python model_development/train_model.py
```
Evaluate the Model:
After training, run the evaluation script to generate performance metrics and plots on the test set.

```bash
python model_development/evaluate_model.py
```

### Step 3: Train the MAAI Model

This script loads the preprocessed data and trains the MAAI model as defined in `model_development/maai_model.py`. The trained model and its weights will be saved to the path specified in the config.

```bash
python model_development/train_model.py
```

### Step 4: Evaluate the Model

After training, run the evaluation script to generate performance metrics and plots (e.g., AUROC, AUPRC, confusion matrix) on the test set.

```bash
python model_development/evaluate_model.py
```

### Using the Jupyter Notebooks
For a more interactive, step-by-step guide through the entire process, open and run the notebooks in the notebooks/ directory. This is highly recommended for understanding the mechanics of the model.

- 01_Data_Exploration.ipynb: Understand the raw data.

- 02_Model_Development_Walkthrough.ipynb: Interactively preprocess data, build, train, and evaluate the model.

Note: The notebooks may require minor adjustments to align with the refactored script structure, such as importing from the centralized config.py file.
