import ast
import numpy as np
import pandas as pd

from model_loader import load_datasets, load_json_config, load_model
from utils import parse_symptoms


class MedicalPredictor:
    CATEGORY_DEFINITIONS = [
        {
            "key": "skin",
            "label": "Skin",
            "icon": "bi-bandaid",
            "description": "Rashes, itching, nail changes, and surface skin symptoms.",
            "symptoms": {
                "itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches",
                "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
                "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
                "blister", "red_sore_around_nose", "yellow_crust_ooze", "bruising",
            },
        },
        {
            "key": "respiratory",
            "label": "Respiratory",
            "icon": "bi-lungs",
            "description": "Breathing, coughing, throat, and mucus-related symptoms.",
            "symptoms": {
                "continuous_sneezing", "cough", "breathlessness", "phlegm",
                "throat_irritation", "runny_nose", "congestion", "chest_pain",
                "mucoid_sputum", "rusty_sputum", "blood_in_sputum", "loss_of_smell",
                "patches_in_throat",
            },
        },
        {
            "key": "digestive",
            "label": "Digestive",
            "icon": "bi-capsule",
            "description": "Stomach, appetite, bowel, and abdominal discomfort symptoms.",
            "symptoms": {
                "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "indigestion",
                "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea",
                "constipation", "pain_during_bowel_movements", "pain_in_anal_region",
                "bloody_stool", "irritation_in_anus", "passage_of_gases",
                "internal_itching", "belly_pain", "stomach_bleeding",
                "distention_of_abdomen", "swelling_of_stomach", "fluid_overload",
                "fluid_overload.1", "increased_appetite",
            },
        },
        {
            "key": "neurological",
            "label": "Neurological",
            "icon": "bi-activity",
            "description": "Head, balance, speech, and sensory system symptoms.",
            "symptoms": {
                "headache", "dizziness", "spinning_movements", "loss_of_balance",
                "unsteadiness", "weakness_of_one_body_side", "altered_sensorium",
                "slurred_speech", "stiff_neck", "lack_of_concentration",
                "visual_disturbances", "blurred_and_distorted_vision", "coma",
                "pain_behind_the_eyes",
            },
        },
        {
            "key": "musculoskeletal",
            "label": "Musculoskeletal",
            "icon": "bi-person-walking",
            "description": "Joint, muscle, movement, and body pain symptoms.",
            "symptoms": {
                "joint_pain", "back_pain", "neck_pain", "cramps", "knee_pain",
                "hip_joint_pain", "muscle_weakness", "muscle_pain", "weakness_in_limbs",
                "swelling_joints", "movement_stiffness", "painful_walking",
                "swollen_legs", "muscle_wasting",
            },
        },
        {
            "key": "general",
            "label": "General",
            "icon": "bi-thermometer-sun",
            "description": "Fever, fatigue, dehydration, and whole-body warning signs.",
            "symptoms": {
                "shivering", "chills", "fatigue", "weight_gain", "weight_loss",
                "restlessness", "lethargy", "high_fever", "mild_fever", "sweating",
                "dehydration", "malaise", "sunken_eyes", "swelled_lymph_nodes",
                "toxic_look_(typhos)",
            },
        },
        {
            "key": "urinary_reproductive",
            "label": "Urinary & Reproductive",
            "icon": "bi-droplet-half",
            "description": "Urination, menstrual, and pelvic-area symptoms.",
            "symptoms": {
                "burning_micturition", "spotting_ urination", "yellow_urine",
                "dark_urine", "bladder_discomfort", "foul_smell_of urine",
                "continuous_feel_of_urine", "abnormal_menstruation", "polyuria",
            },
        },
        {
            "key": "metabolic",
            "label": "Metabolic & Hormonal",
            "icon": "bi-heart-pulse",
            "description": "Thyroid, sugar, appetite, and mood-related symptoms.",
            "symptoms": {
                "irregular_sugar_level", "anxiety", "cold_hands_and_feets",
                "mood_swings", "obesity", "puffy_face_and_eyes", "enlarged_thyroid",
                "brittle_nails", "swollen_extremeties", "excessive_hunger",
                "depression", "irritability",
            },
        },
        {
            "key": "circulatory",
            "label": "Circulatory",
            "icon": "bi-heart",
            "description": "Heart rate, veins, and circulation-related symptoms.",
            "symptoms": {
                "fast_heart_rate", "palpitations", "swollen_blood_vessels",
                "prominent_veins_on_calf",
            },
        },
        {
            "key": "eyes_ent",
            "label": "Eyes, Nose & Throat",
            "icon": "bi-eye",
            "description": "Vision, sinus, eye, and upper-airway irritation symptoms.",
            "symptoms": {
                "redness_of_eyes", "sinus_pressure", "watering_from_eyes",
                "yellowing_of_eyes",
            },
        },
        {
            "key": "history_risk",
            "label": "History & Risk Factors",
            "icon": "bi-clipboard2-pulse",
            "description": "Personal history and exposure indicators that affect prediction.",
            "symptoms": {
                "family_history", "receiving_blood_transfusion",
                "receiving_unsterile_injections", "history_of_alcohol_consumption",
                "extra_marital_contacts",
            },
        },
    ]

    def __init__(self, model_bundle, datasets, symptom_index, symptom_aliases):
        self.model = model_bundle["model"]
        self.feature_names = model_bundle["feature_names"]
        self.model_name = model_bundle.get("model_name", type(self.model).__name__)
        self.datasets = datasets
        self.symptom_index = symptom_index
        self.symptom_aliases = symptom_aliases

    @classmethod
    def from_project_files(cls):
        return cls(
            model_bundle=load_model(),
            datasets=load_datasets(),
            symptom_index=load_json_config("symptoms.json"),
            symptom_aliases=load_json_config("symptom_aliases.json"),
        )

    def parse_user_symptoms(self, raw_symptoms):
        return parse_symptoms(raw_symptoms, self.symptom_index, self.symptom_aliases)

    def get_symptom_options(self):
        return [
            {
                "value": feature_name,
                "label": feature_name.replace("_", " "),
            }
            for feature_name in self.feature_names
        ]

    def get_symptom_categories(self):
        assigned = set()
        categories = []

        for definition in self.CATEGORY_DEFINITIONS:
            category_symptoms = [
                {
                    "value": feature_name,
                    "label": feature_name.replace("_", " "),
                }
                for feature_name in self.feature_names
                if feature_name in definition["symptoms"]
            ]

            assigned.update(item["value"] for item in category_symptoms)
            categories.append(
                {
                    "key": definition["key"],
                    "label": definition["label"],
                    "icon": definition["icon"],
                    "description": definition["description"],
                    "count": len(category_symptoms),
                    "symptoms": category_symptoms,
                }
            )

        remaining = [
            {
                "value": feature_name,
                "label": feature_name.replace("_", " "),
            }
            for feature_name in self.feature_names
            if feature_name not in assigned
        ]

        if remaining:
            categories.append(
                {
                    "key": "other",
                    "label": "Other",
                    "icon": "bi-grid",
                    "description": "Additional symptoms that do not fit a primary browsing group.",
                    "count": len(remaining),
                    "symptoms": remaining,
                }
            )

        return categories

    @staticmethod
    def _normalize_recommendation_list(values):
        normalized = []

        for value in values:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    try:
                        parsed = ast.literal_eval(stripped)
                    except (ValueError, SyntaxError):
                        parsed = [stripped]

                    if isinstance(parsed, list):
                        normalized.extend(str(item).strip() for item in parsed if str(item).strip())
                        continue

            if str(value).strip():
                normalized.append(str(value).strip())

        return normalized

    def _build_feature_frame(self, patient_symptoms):
        input_vector = np.zeros(len(self.feature_names))
        for item in patient_symptoms:
            input_vector[self.symptom_index[item]] = 1
        return pd.DataFrame([input_vector], columns=self.feature_names)

    def predict_disease(self, patient_symptoms):
        feature_frame = self._build_feature_frame(patient_symptoms)
        return self.model.predict(feature_frame)[0]

    def predict_top_diseases(self, patient_symptoms, top_k=3):
        feature_frame = self._build_feature_frame(patient_symptoms)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_frame)[0]
            classes = self.model.classes_
        else:
            scores = self.model.decision_function(feature_frame)[0]
            scores = np.atleast_1d(scores)
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()
            classes = self.model.classes_

        ranked_indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            {
                "disease": classes[index],
                "confidence": round(float(probabilities[index]) * 100, 2),
            }
            for index in ranked_indices
        ]

    def get_recommendations(self, disease_name):
        description_data = self.datasets["description"]
        precautions_data = self.datasets["precautions"]
        medications_data = self.datasets["medications"]
        diets_data = self.datasets["diets"]
        workout_data = self.datasets["workout"]

        desc = description_data[description_data["Disease"] == disease_name]["Description"]
        desc = " ".join([item for item in desc])

        precautions = precautions_data[
            precautions_data["Disease"] == disease_name
        ][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
        precautions = [item for item in precautions.values]

        medications = medications_data[medications_data["Disease"] == disease_name]["Medication"]
        medications = self._normalize_recommendation_list(medications.values)

        diets = diets_data[diets_data["Disease"] == disease_name]["Diet"]
        diets = self._normalize_recommendation_list(diets.values)

        workouts = workout_data[workout_data["disease"] == disease_name]["workout"]
        workouts = self._normalize_recommendation_list(workouts.values)

        return {
            "description": desc,
            "precautions": precautions[0] if precautions else [],
            "medications": medications,
            "diet": diets,
            "workout": workouts,
        }

    def get_consultation_guidance(self, disease_name):
        urgent_signs = [
            "chest pain",
            "trouble breathing",
            "confusion",
            "persistent high fever",
            "severe dehydration",
            "fainting",
            "blood in stool or vomit",
        ]

        disease_specific = {
            "Heart attack": {
                "level": "Urgent",
                "tone": "urgent",
                "message": "Seek emergency medical care immediately. Do not wait for symptoms to improve on their own.",
            },
            "Paralysis (brain hemorrhage)": {
                "level": "Urgent",
                "tone": "urgent",
                "message": "Go to an emergency department immediately, especially if weakness, speech changes, or facial drooping are present.",
            },
            "Pneumonia": {
                "level": "Same day",
                "tone": "same-day",
                "message": "Speak with a doctor the same day if symptoms include shortness of breath, chest pain, or a fever that is getting worse.",
            },
            "Dengue": {
                "level": "Same day",
                "tone": "same-day",
                "message": "Get prompt medical advice if there is severe abdominal pain, repeated vomiting, bleeding, or unusual weakness.",
            },
            "Typhoid": {
                "level": "Same day",
                "tone": "same-day",
                "message": "Arrange medical review quickly if fever persists, symptoms worsen, or you are unable to keep fluids down.",
            },
            "Jaundice": {
                "level": "Soon",
                "tone": "soon",
                "message": "Consult a doctor soon, especially if yellowing of the eyes, dark urine, or abdominal pain continues.",
            },
            "Tuberculosis": {
                "level": "Soon",
                "tone": "soon",
                "message": "Schedule medical evaluation soon if cough, weight loss, weakness, or fever continues.",
            },
        }

        default_guidance = {
            "level": "Monitor",
            "tone": "monitor",
            "message": "Consult a doctor if symptoms do not improve, become more severe, or interfere with daily activity.",
        }

        guidance = disease_specific.get(disease_name, default_guidance)
        return {
            "level": guidance["level"],
            "tone": guidance["tone"],
            "message": guidance["message"],
            "urgent_signs": urgent_signs,
        }

    def get_recommendation_safety(self, disease_name):
        urgent_diseases = {
            "Heart attack",
            "Paralysis (brain hemorrhage)",
            "Pneumonia",
            "Dengue",
            "Typhoid",
            "Tuberculosis",
        }

        follow_up_priority = "urgent medical evaluation" if disease_name in urgent_diseases else "follow-up with a qualified clinician"

        return {
            "summary": "These recommendation cards are supportive health information, not a prescription or diagnosis.",
            "medication": "Do not start, stop, or change medicines based only on this app. Medication decisions should be confirmed by a doctor or pharmacist.",
            "diet": "Diet suggestions are general wellness ideas and may not fit allergies, chronic conditions, pregnancy, or current treatment plans.",
            "activity": "Reduce or stop activity if symptoms worsen, dizziness develops, breathing becomes difficult, or pain increases.",
            "follow_up": f"If symptoms persist, become more severe, or new symptoms appear, seek {follow_up_priority}.",
        }
