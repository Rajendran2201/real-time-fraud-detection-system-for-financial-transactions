from pathlib import Path 

# Reproducibility 
RANDOM_STATE = 42

# Paths 
RAW_DATA_PATH = Path('../data/raw/creditcard.csv')

DATA_INTERIM_DIR = Path('../data/interim/')
DATA_PROCESSED_DIR = Path('../data/processed/')
MODELS_DIR = Path('../models/')

# Dataset 

TARGET_COL = "Class"
