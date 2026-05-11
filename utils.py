def format_symptom(symptom_key):
    return symptom_key.replace("_", " ")


def normalize_symptom(symptom_text):
    cleaned = symptom_text.strip().lower().replace("-", " ").replace("/", " ")
    return "_".join(cleaned.split())


def parse_symptoms(raw_symptoms, supported_symptoms, symptom_aliases):
    parsed_symptoms = []
    invalid_symptoms = []
    suggestion_map = {}

    for symptom in raw_symptoms.split(","):
        normalized = normalize_symptom(symptom)
        if not normalized:
            continue

        if normalized in supported_symptoms:
            parsed_symptoms.append(normalized)
            continue

        alias_matches = symptom_aliases.get(normalized)
        if alias_matches:
            if len(alias_matches) == 1:
                parsed_symptoms.append(alias_matches[0])
            else:
                invalid_symptoms.append(symptom.strip())
                suggestion_map[symptom.strip()] = [format_symptom(item) for item in alias_matches]
            continue

        invalid_symptoms.append(symptom.strip())

    unique_symptoms = list(dict.fromkeys(parsed_symptoms))
    return unique_symptoms, invalid_symptoms, suggestion_map


def build_error_message(invalid_symptoms, suggestion_map):
    if not invalid_symptoms:
        return None

    parts = []
    for item in invalid_symptoms:
        suggestions = suggestion_map.get(item)
        if suggestions:
            parts.append(
                f"'{item}' is ambiguous. Try one of: {', '.join(suggestions)}."
            )
        else:
            parts.append(f"'{item}' is not a supported symptom.")
    return " ".join(parts)
