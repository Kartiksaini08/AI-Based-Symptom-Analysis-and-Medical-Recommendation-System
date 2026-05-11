import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Datasets" / "Training.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RANDOM_STATE = 42
CV_SPLITS = 5


def load_training_data():
    dataframe = pd.read_csv(DATASET_PATH)
    features = dataframe.drop(columns=["prognosis"])
    target = dataframe["prognosis"]
    return features, target


def build_model_candidates():
    candidates = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "svc": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        pass

    return candidates


def deduplicate_profiles(features, target):
    combined = features.copy()
    combined["prognosis"] = target
    deduplicated = combined.drop_duplicates()
    dedup_features = deduplicated.drop(columns=["prognosis"]).reset_index(drop=True)
    dedup_target = deduplicated["prognosis"].reset_index(drop=True)
    return dedup_features, dedup_target


def calculate_dataset_diagnostics(features, dedup_features, target):
    duplicate_feature_rows = int(features.duplicated().sum())
    return {
        "total_rows": int(len(features)),
        "total_features": int(features.shape[1]),
        "class_count": int(target.nunique()),
        "duplicate_feature_rows": duplicate_feature_rows,
        "duplicate_feature_ratio": round(float(duplicate_feature_rows / len(features)), 4),
        "unique_symptom_profiles": int(len(dedup_features)),
        "rows_per_class": {
            str(label): int(count)
            for label, count in target.value_counts().sort_index().items()
        },
    }


def build_split_metrics(y_true, predictions):
    labels = sorted(set(y_true))
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision_weighted": round(float(precision_score(y_true, predictions, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, predictions, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, predictions, average="weighted", zero_division=0)), 4),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
    }


def evaluate_standard_split(model, features, target):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    fitted_model = clone(model)
    fitted_model.fit(x_train, y_train)
    predictions = fitted_model.predict(x_test)
    return build_split_metrics(y_test, predictions)


def evaluate_deduplicated_split(model, dedup_features, dedup_target):
    x_train, x_test, y_train, y_test = train_test_split(
        dedup_features,
        dedup_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dedup_target,
    )
    fitted_model = clone(model)
    fitted_model.fit(x_train, y_train)
    predictions = fitted_model.predict(x_test)
    return build_split_metrics(y_test, predictions)


def evaluate_cross_validation(model, dedup_features, dedup_target):
    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for train_indices, test_indices in splitter.split(dedup_features, dedup_target):
        x_train = dedup_features.iloc[train_indices]
        x_test = dedup_features.iloc[test_indices]
        y_train = dedup_target.iloc[train_indices]
        y_test = dedup_target.iloc[test_indices]

        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train)
        predictions = fitted_model.predict(x_test)
        fold_scores.append(f1_score(y_test, predictions, average="weighted", zero_division=0))

    return {
        "fold_count": CV_SPLITS,
        "f1_weighted_scores": [round(float(score), 4) for score in fold_scores],
        "f1_weighted_mean": round(float(sum(fold_scores) / len(fold_scores)), 4),
        "f1_weighted_std": round(float(pd.Series(fold_scores).std(ddof=0)), 4),
    }


def evaluate_models(features, target):
    dedup_features, dedup_target = deduplicate_profiles(features, target)
    candidate_models = build_model_candidates()
    metrics = {}
    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in candidate_models.items():
        standard_split = evaluate_standard_split(model, features, target)
        dedup_split = evaluate_deduplicated_split(model, dedup_features, dedup_target)
        cross_validation = evaluate_cross_validation(model, dedup_features, dedup_target)

        metrics[name] = {
            "standard_split": standard_split,
            "deduplicated_split": dedup_split,
            "cross_validation": cross_validation,
        }

        realism_score = (
            dedup_split["f1_weighted"] * 0.7
            + cross_validation["f1_weighted_mean"] * 0.3
        )

        if realism_score > best_score:
            best_score = realism_score
            best_name = name
            best_model = clone(model).fit(features, target)

    dataset_diagnostics = calculate_dataset_diagnostics(features, dedup_features, target)
    return best_name, best_model, metrics, dataset_diagnostics


def save_artifacts(best_name, best_model, metrics, dataset_diagnostics, feature_names):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "model": best_model,
        "feature_names": list(feature_names),
        "model_name": best_name,
        "trained_at_utc": trained_at,
        "metrics": metrics[best_name],
        "dataset_diagnostics": dataset_diagnostics,
    }
    joblib.dump(bundle, ARTIFACTS_DIR / "disease_prediction_bundle.joblib")

    summary = {
        "best_model": best_name,
        "trained_at_utc": trained_at,
        "dataset_diagnostics": dataset_diagnostics,
        "models": metrics,
    }
    with open(ARTIFACTS_DIR / "model_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)


def main():
    features, target = load_training_data()
    best_name, best_model, metrics, dataset_diagnostics = evaluate_models(features, target)
    save_artifacts(best_name, best_model, metrics, dataset_diagnostics, features.columns)

    print("Training complete.")
    print(f"Best model: {best_name}")
    print(
        "Dataset diagnostics:",
        f"rows={dataset_diagnostics['total_rows']},",
        f"unique_profiles={dataset_diagnostics['unique_symptom_profiles']},",
        f"duplicate_ratio={dataset_diagnostics['duplicate_feature_ratio']}",
    )
    for name, result in metrics.items():
        print(
            f"{name}: "
            f"standard_f1={result['standard_split']['f1_weighted']}, "
            f"dedup_f1={result['deduplicated_split']['f1_weighted']}, "
            f"cv_mean_f1={result['cross_validation']['f1_weighted_mean']}"
        )


if __name__ == "__main__":
    main()
