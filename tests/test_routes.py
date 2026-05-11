def test_home_page_loads(client):
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Build your symptom list with guided search" in body
    assert "Browse by category" in body


def test_predict_route_returns_prediction_for_valid_input(client):
    response = client.post("/predict", data={"symptoms": "itching, skin rash"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Prediction Results" in body
    assert "Fungal infection" in body


def test_predict_route_returns_validation_message_for_empty_input(client):
    response = client.post("/predict", data={"symptoms": ""})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Enter one or more symptoms separated by commas." in body


def test_predict_route_returns_validation_message_for_invalid_input(client):
    response = client.post("/predict", data={"symptoms": "abc"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "not a supported symptom" in body


def test_predict_route_renders_clean_recommendation_lists(client):
    response = client.post(
        "/predict",
        data={"symptoms": "continuous sneezing, runny nose, itching"},
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Antihistamines" in body
    assert "['Antihistamines'" not in body
    assert "Elimination Diet" in body
    assert "['Elimination Diet'" not in body


def test_predict_route_renders_recommendation_safety_messages(client):
    response = client.post("/predict", data={"symptoms": "itching, skin rash"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Recommendation safety" in body
    assert "Use these suggestions as support, not treatment instructions." in body
    assert "Medicines sometimes used" in body
    assert "Do not start, stop, or change medicines based only on this app." in body
