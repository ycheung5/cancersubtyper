from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterRule:
    key: str
    default: float | int
    minimum: float | int
    maximum: float | int
    kind: type


BCTYPEFINDER_RULES = (
    ParameterRule("is_using_pretrained_model", 1, 0, 1, int),
    ParameterRule("num_cpg_clusters", 3000, 100, 10000, int),
    ParameterRule("batch_size", 128, 1, 1024, int),
    ParameterRule("target_batch_size", 128, 1, 1024, int),
    ParameterRule("num_hidden_nodes1_feature_extractor", 1024, 8, 8192, int),
    ParameterRule("num_hidden_nodes2_feature_extractor", 512, 8, 8192, int),
    ParameterRule("num_hidden_nodes1_classifier", 256, 8, 4096, int),
    ParameterRule("num_hidden_nodes2_classifier", 64, 8, 4096, int),
    ParameterRule("num_hidden_nodes1_discriminator", 256, 8, 4096, int),
    ParameterRule("num_hidden_nodes2_discriminator", 64, 8, 4096, int),
    ParameterRule("learning_rate_feature_extractor", 0.0001, 1e-7, 1.0, float),
    ParameterRule("learning_rate_classifier", 0.00001, 1e-7, 1.0, float),
    ParameterRule("learning_rate_discriminator", 0.000001, 1e-8, 1.0, float),
    ParameterRule("num_epochs_pretraining", 800, 1, 5000, int),
    ParameterRule("num_epochs_adversarial_training", 500, 1, 5000, int),
    ParameterRule("num_epochs_semi_supervised_learning", 500, 1, 5000, int),
    ParameterRule("num_epochs_fine_tuning", 800, 1, 5000, int),
)

CANCERSUBMINER_RULES = (
    ParameterRule("is_automatically_estimation_required", 1, 0, 1, int),
    ParameterRule("num_subtype_user_defined", 2, 2, 50, int),
    ParameterRule("is_using_pretrained_model", 1, 0, 1, int),
    ParameterRule("num_cpg_clusters", 3000, 100, 10000, int),
    ParameterRule("batch_size", 128, 1, 1024, int),
    ParameterRule("target_batch_size", 128, 1, 1024, int),
    ParameterRule("num_hidden_nodes1_feature_extractor", 1024, 8, 8192, int),
    ParameterRule("num_hidden_nodes2_feature_extractor", 512, 8, 8192, int),
    ParameterRule("num_hidden_nodes1_classifier", 256, 8, 4096, int),
    ParameterRule("num_hidden_nodes2_classifier", 64, 8, 4096, int),
    ParameterRule("num_hidden_nodes1_discriminator", 256, 8, 4096, int),
    ParameterRule("num_hidden_nodes2_discriminator", 64, 8, 4096, int),
    ParameterRule("learning_rate_feature_extractor", 0.0001, 1e-7, 1.0, float),
    ParameterRule("learning_rate_classifier", 0.00001, 1e-7, 1.0, float),
    ParameterRule("learning_rate_discriminator", 0.000001, 1e-8, 1.0, float),
    ParameterRule("num_epochs_pretraining", 800, 1, 5000, int),
    ParameterRule("num_epochs_adversarial_training", 500, 1, 5000, int),
    ParameterRule("num_epochs_semi_supervised_learning", 300, 1, 5000, int),
    ParameterRule("num_epochs_fine_tuning", 300, 1, 5000, int),
)


def _normalize(parameters: list, rules: tuple[ParameterRule, ...]) -> dict[str, int | float]:
    normalized: dict[str, int | float] = {}

    for index, rule in enumerate(rules):
        raw_value = parameters[index] if index < len(parameters) else rule.default
        try:
            value = rule.kind(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{rule.key} must be a valid {rule.kind.__name__}") from exc

        if value < rule.minimum or value > rule.maximum:
            raise ValueError(f"{rule.key} must be between {rule.minimum} and {rule.maximum}")

        normalized[rule.key] = value

    return normalized


def normalize_model_parameters(model_name: str, parameters: list) -> dict[str, int | float]:
    if model_name == "BCtypeFinder":
        return _normalize(parameters, BCTYPEFINDER_RULES)
    if model_name == "CancerSubminer":
        return _normalize(parameters, CANCERSUBMINER_RULES)
    raise ValueError(f"Unsupported model: {model_name}")
