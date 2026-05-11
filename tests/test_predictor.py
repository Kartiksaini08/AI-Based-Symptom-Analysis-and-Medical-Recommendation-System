from predictor import MedicalPredictor


def test_category_browser_covers_all_features():
    predictor = MedicalPredictor.from_project_files()
    categories = predictor.get_symptom_categories()

    category_feature_names = {
        symptom["value"]
        for category in categories
        for symptom in category["symptoms"]
    }

    assert set(predictor.feature_names) == category_feature_names


def test_recommendation_list_normalization_splits_stringified_lists():
    predictor = MedicalPredictor.from_project_files()
    normalized = predictor._normalize_recommendation_list(
        ["['A', 'B', 'C']", "Single Item"]
    )

    assert normalized == ["A", "B", "C", "Single Item"]


def test_predict_top_diseases_returns_ranked_matches():
    predictor = MedicalPredictor.from_project_files()
    top_predictions = predictor.predict_top_diseases(
        ["itching", "skin_rash", "nodal_skin_eruptions"],
        top_k=3,
    )

    assert len(top_predictions) == 3
    assert top_predictions[0]["disease"] == "Fungal infection"
    assert top_predictions[0]["confidence"] >= top_predictions[1]["confidence"]


def test_recommendations_return_clean_lists():
    predictor = MedicalPredictor.from_project_files()
    recommendations = predictor.get_recommendations("Allergy")

    assert "Antihistamines" in recommendations["medications"]
    assert all(not item.startswith("[") for item in recommendations["medications"])
    assert "Elimination Diet" in recommendations["diet"]


def test_recommendation_safety_returns_non_prescriptive_messages():
    predictor = MedicalPredictor.from_project_files()
    safety = predictor.get_recommendation_safety("Heart attack")

    assert "not a prescription or diagnosis" in safety["summary"]
    assert "Do not start, stop, or change medicines" in safety["medication"]
    assert "urgent medical evaluation" in safety["follow_up"]
