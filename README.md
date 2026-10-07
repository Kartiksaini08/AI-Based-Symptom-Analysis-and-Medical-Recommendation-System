# Symptom-Based Disease Prediction and Recommendation System

A Flask-based machine learning web application that accepts user symptoms, predicts a likely disease, and returns supporting recommendations such as precautions, diet, medications, and workout guidance.

## Features

- Symptom-based disease prediction using a trained scikit-learn model
- Reproducible training pipeline with model comparison and saved evaluation metrics
- Input validation for unsupported or ambiguous symptom names
- Friendly UI with medical disclaimer and recognized-symptom feedback
- Recommendation panels for disease description, precautions, medications, diet, and workouts
- Modular code structure for easier maintenance and future ML upgrades

## Tech Stack

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- Bootstrap 5

## Project Structure

```text
.
|-- app.py
|-- artifacts/
|-- main.py
|-- model_loader.py
|-- predictor.py
|-- routes.py
|-- utils.py
|-- train_model.py
|-- config/
|   |-- diseases.json
|   |-- symptom_aliases.json
|   `-- symptoms.json
|-- Datasets/
|-- model/
|-- static/
`-- templates/
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python main.py
```

4. Open the browser at:

```text
http://127.0.0.1:5001
```

## Rebuild the Model

Run the training pipeline to compare multiple models and export a fresh inference bundle:

```bash
python train_model.py
```

This creates:

- `artifacts/disease_prediction_bundle.joblib`
- `artifacts/model_metrics.json`

The saved metrics now include:

- a standard stratified holdout split
- a stricter deduplicated-profile holdout split
- 5-fold cross-validation on unique symptom profiles
- dataset diagnostics such as duplicate-row ratio and unique profile count

## How It Works

1. The user enters symptoms separated by commas.
2. The app normalizes and validates the symptom names.
3. A trained classifier predicts the most likely disease.
4. The app looks up disease-specific recommendations from CSV datasets.
5. Results are shown in inline guidance cards on the home page.

## Screenshots

- Home page with symptom input and validation feedback
- Prediction result view with recommendation modals

## Current Limitations

- The training dataset is highly repetitive. It contains `4,920` rows but only `304` unique symptom profiles, so naive train/test splits can look unrealistically strong.
- Perfect or near-perfect scores on the standard split should not be treated as real-world clinical performance.
- Model quality still depends heavily on the training dataset and symptom vocabulary.
- The current evaluation is still limited to the provided dataset; it does not include external validation, real patient records, or temporal drift checks.
- Symptom combinations in real use may be noisier, incomplete, or phrased differently than the dataset examples.
- Free-text symptom input still depends on the current symptom vocabulary.
- The recommendation content is dataset-driven and not medically verified for clinical use.



## Disclaimer

This project is for educational and informational use only. It is not a substitute for professional medical diagnosis or treatment.
