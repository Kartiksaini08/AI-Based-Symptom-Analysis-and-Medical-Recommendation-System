import json
import pickle
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
MODEL_DIR = BASE_DIR / "model"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CONFIG_DIR = BASE_DIR / "config"


def load_json_config(filename):
    with open(CONFIG_DIR / filename, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def load_model():
    bundle_path = ARTIFACTS_DIR / "disease_prediction_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)

    with open(MODEL_DIR / "svc.pkl", "rb") as model_file:
        legacy_model = pickle.load(model_file)

    return {
        "model": legacy_model,
        "feature_names": list(load_json_config("symptoms.json").keys()),
        "model_name": "Legacy SVC",
    }


def load_datasets():
    return {
        "symptom_details": pd.read_csv(DATA_DIR / "symtoms_df.csv"),
        "precautions": pd.read_csv(DATA_DIR / "precautions_df.csv"),
        "workout": pd.read_csv(DATA_DIR / "workout_df.csv"),
        "description": pd.read_csv(DATA_DIR / "description.csv"),
        "medications": pd.read_csv(DATA_DIR / "medications.csv"),
        "diets": pd.read_csv(DATA_DIR / "diets.csv"),
    }
