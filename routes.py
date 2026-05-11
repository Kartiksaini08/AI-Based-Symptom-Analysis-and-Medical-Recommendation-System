from flask import current_app, render_template, request

from utils import build_error_message, format_symptom


def register_routes(app, predictor):
    def render_home(**context):
        return render_template(
            "index.html",
            symptom_options=predictor.get_symptom_options(),
            symptom_categories=predictor.get_symptom_categories(),
            **context,
        )

    @app.route("/")
    def index():
        return render_home()

    @app.route("/predict", methods=["GET", "POST"])
    def home():
        if request.method == "POST":
            symptoms = request.form.get("symptoms", "").strip()
            if not symptoms:
                message = "Enter one or more symptoms separated by commas."
                current_app.logger.warning("Prediction request submitted without symptoms")
                return render_home(message=message, entered_symptoms=symptoms)

            user_symptoms, invalid_symptoms, suggestion_map = predictor.parse_user_symptoms(symptoms)
            if not user_symptoms or invalid_symptoms:
                message = build_error_message(invalid_symptoms, suggestion_map)
                if not user_symptoms:
                    base_message = "Enter at least one supported symptom."
                    message = f"{base_message} {message}".strip() if message else base_message

                current_app.logger.info(
                    "Prediction request rejected",
                    extra={
                        "raw_symptoms": symptoms,
                        "recognized_symptom_count": len(user_symptoms),
                        "invalid_symptom_count": len(invalid_symptoms),
                    },
                )
                return render_home(
                    message=message,
                    entered_symptoms=symptoms,
                    recognized_symptoms=[format_symptom(symptom) for symptom in user_symptoms],
                )

            try:
                predicted_disease = predictor.predict_disease(user_symptoms)
                top_predictions = predictor.predict_top_diseases(user_symptoms)
                recommendations = predictor.get_recommendations(predicted_disease)
                consultation_guidance = predictor.get_consultation_guidance(predicted_disease)
                recommendation_safety = predictor.get_recommendation_safety(predicted_disease)
            except Exception:
                current_app.logger.exception("Prediction request failed during inference")
                message = "We could not generate a prediction right now. Please try again."
                return render_home(
                    message=message,
                    entered_symptoms=symptoms,
                    recognized_symptoms=[format_symptom(symptom) for symptom in user_symptoms],
                )

            current_app.logger.info(
                "Prediction generated",
                extra={
                    "raw_symptoms": symptoms,
                    "recognized_symptom_count": len(user_symptoms),
                    "predicted_disease": predicted_disease,
                },
            )

            return render_home(
                predicted_disease=predicted_disease,
                top_predictions=top_predictions,
                model_name=predictor.model_name,
                consultation_guidance=consultation_guidance,
                recommendation_safety=recommendation_safety,
                dis_des=recommendations["description"],
                my_precautions=list(recommendations["precautions"]),
                medications=recommendations["medications"],
                my_diet=recommendations["diet"],
                workout=recommendations["workout"],
                entered_symptoms=symptoms,
                recognized_symptoms=[format_symptom(symptom) for symptom in user_symptoms],
            )

        return render_home()

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/blog")
    def blog():
        return render_template("blog.html")

    @app.route("/developer")
    def developer():
        return render_template("developer.html")

    @app.route("/contact")
    def contact():
        return render_template("contact.html")
