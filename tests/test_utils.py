from utils import build_error_message, normalize_symptom, parse_symptoms


def test_normalize_symptom_replaces_spacing_and_case():
    assert normalize_symptom(" High Fever ") == "high_fever"
    assert normalize_symptom("runny-nose") == "runny_nose"


def test_parse_symptoms_supports_exact_and_alias_matches():
    supported_symptoms = {"itching": 0, "skin_rash": 1, "fatigue": 2}
    aliases = {"rash": ["skin_rash"], "tiredness": ["fatigue"]}

    parsed, invalid, suggestions = parse_symptoms(
        "itching, rash, tiredness",
        supported_symptoms,
        aliases,
    )

    assert parsed == ["itching", "skin_rash", "fatigue"]
    assert invalid == []
    assert suggestions == {}


def test_parse_symptoms_flags_ambiguous_aliases():
    supported_symptoms = {"high_fever": 0, "mild_fever": 1}
    aliases = {"fever": ["high_fever", "mild_fever"]}

    parsed, invalid, suggestions = parse_symptoms("fever", supported_symptoms, aliases)

    assert parsed == []
    assert invalid == ["fever"]
    assert suggestions == {"fever": ["high fever", "mild fever"]}


def test_build_error_message_includes_supported_guidance():
    message = build_error_message(
        ["fever", "abc"],
        {"fever": ["high fever", "mild fever"]},
    )

    assert "'fever' is ambiguous." in message
    assert "'abc' is not a supported symptom." in message
