from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diss.core.types import ModelSpec

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
MANIFEST_PATH = CONFIG_DIR / "model_manifest.yaml"

from diss.core.locale_meta import (
    CULTURE_COUNTRY_BY_LOCALE,
    LOCALE_DISPLAY_ORDER,
)

MODEL_PARAMS: dict[str, float] = {
    "goldfish-ko": 125e6,
    "goldfish-es": 125e6,
    "goldfish-zh": 125e6,
    "goldfish-ar": 125e6,
    "goldfish-ja": 125e6,
    "goldfish-fr": 125e6,
    "ko-en-intermediate": 277e6,
    "es-en-intermediate": 277e6,
    "zh-en-intermediate": 277e6,
    "ar-en-intermediate": 277e6,
    "ja-en-intermediate": 277e6,
    "fr-en-intermediate": 277e6,
    "xglm-564m": 564e6,
    "xglm-1p7b": 1.7e9,
    "xglm-2p9b": 2.9e9,
    "xglm-7p5b": 7.5e9,
    "polyglot-ko-1p3b": 1.3e9,
    "gpt-fr-cased-base": 1.0e9,
    "aragpt2-mega": 1.46e9,
    "beetle-zh-en-50-b2": 194e6,
    "rinna-bilingual-neox-4b": 3.8e9,
    "croissantllm-base": 1.3e9,
    "jais-family-1p3b": 1.3e9,
    "kanana-1p5-2p1b-base": 2.1e9,
    "rinna-monolingual-neox-3p6b": 3.6e9,
    "hplt2c-kor": 2.15e9,
    "hplt2c-ara": 2.15e9,
    "hplt2c-fra": 2.15e9,
    "hplt2c-zhos": 2.15e9,
    "hplt2c-spa": 2.15e9,
    "eurollm-1p7b": 1.7e9,
    "tiny-aya-base": 3.35e9,
    "qwen3-0p6b-base": 0.6e9,
    "qwen3-1p7b-base": 1.7e9,
    "qwen3-4b-base": 4e9,
}


def fmt_params(p: float) -> str:
    """Format a parameter count as '125M' or '1.7B'."""
    if p >= 1e9:
        s = f"{p / 1e9:.1f}".rstrip("0").rstrip(".")
        return f"{s}B"
    return f"{int(round(p / 1e6))}M"


TRAIN_TOKENS_BY_MODEL_LOCALE: dict[tuple[str, str], float] = {
    ("goldfish-ko", "ko"): 227.0e6,
    ("goldfish-es", "es"): 221.8e6,
    ("goldfish-zh", "zh"): 206.2e6,
    ("goldfish-ar", "ar"): 196.2e6,
    ("goldfish-ja", "ja"): 219.1e6,
    ("goldfish-fr", "fr"): 251.4e6,
    ("ko-en-intermediate", "ko"): 3.33e9,
    ("es-en-intermediate", "es"): 3.33e9,
    ("zh-en-intermediate", "zh"): 3.33e9,
    ("ar-en-intermediate", "ar"): 3.33e9,
    ("fr-en-intermediate", "fr"): 3.33e9,
    ("ja-en-intermediate", "ja"): 3.33e9,
    ("rinna-monolingual-neox-3p6b", "ja"): 312.5e9,
    ("kanana-1p5-2p1b-base", "ko"): 320.0e9,
    ("croissantllm-base", "fr"): 1240.0e9,
    ("jais-family-1p3b", "ar"): 141.0e9,
    ("hplt2c-kor", "ko"): 19.7e9,
    ("hplt2c-ara", "ar"): 48.14e9,
    ("hplt2c-fra", "fr"): 237.0e9,
    ("hplt2c-zhos", "zh"): 74.03e9,
    ("hplt2c-spa", "es"): 322.0e9,
    ("xglm-564m", "ko"): 40.6e9,
    ("xglm-564m", "ja"): 13.5e9,
    ("xglm-564m", "zh"): 24.2e9,
    ("xglm-564m", "es"): 17.7e9,
    ("xglm-564m", "fr"): 15.7e9,
}

TRAIN_TOKENS_APPROX: set[tuple[str, str]] = {
    ("kanana-1p5-2p1b-base", "ko"),
    ("hplt2c-kor", "ko"),
    ("hplt2c-ara", "ar"),
    ("hplt2c-fra", "fr"),
    ("hplt2c-zhos", "zh"),
    ("hplt2c-spa", "es"),
}


def target_lang_tokens(model_id: str, locale: str) -> float | None:
    """Reported training tokens in `locale` for `model_id`, or None if unknown."""
    return TRAIN_TOKENS_BY_MODEL_LOCALE.get((model_id, locale))


def load_registry(path: Path | None = None) -> list[ModelSpec]:
    import yaml

    resolved = Path(path) if path is not None else MANIFEST_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Manifest not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    _KNOWN_KEYS = {
        "id",
        "name",
        "hf_name",
        "architecture",
        "architecture_family",
        "regime",
        "role",
        "locales",
        "eval_locales",
        "culture_enabled",
        "reasoning_enabled",
        "neuron_enabled",
        "enabled",
        "revision",
    }
    specs: list[ModelSpec] = []
    for item in payload.get("models", []):
        hf_name = item.get("hf_name", item.get("name"))
        if not hf_name:
            raise ValueError(f"Model entry missing hf_name/name in {resolved}")
        extras = {k: v for k, v in item.items() if k not in _KNOWN_KEYS}
        specs.append(
            ModelSpec(
                model_id=item.get("id", hf_name.split("/")[-1]),
                name=hf_name,
                architecture_family=item.get(
                    "architecture_family", item.get("architecture", "")
                ),
                role=item.get("role", item.get("regime", "unknown")),
                locales=item.get("locales", []),
                eval_locales=item.get("eval_locales", item.get("locales", [])),
                culture_enabled=item.get("culture_enabled", True),
                reasoning_enabled=item.get("reasoning_enabled", True),
                neuron_enabled=item.get("neuron_enabled", True),
                enabled=item.get("enabled", True),
                revision=item.get("revision"),
                metadata=extras,
            )
        )
    return specs


def load_enabled_models(path: Path | None = None) -> list[ModelSpec]:
    return [s for s in load_registry(path) if s.enabled]


def get_model_spec(model_name_or_id: str, path: Path | None = None) -> ModelSpec | None:
    try:
        specs = load_registry(path)
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    for spec in specs:
        if spec.name == model_name_or_id or spec.model_id == model_name_or_id:
            return spec
    return None


def infer_architecture_family(model_name: str) -> str:
    """Resolve architecture family from model_manifest.yaml's ``architecture`` field.

    Models outside the manifest (e.g. checkpoint-trace runs) must pass ``family=``
    to ``get_adapter`` directly instead.
    """
    spec = get_model_spec(model_name)
    if spec is not None and spec.architecture_family:
        return spec.architecture_family
    raise ValueError(
        f"No architecture family for {model_name!r}: add the model to "
        f"model_manifest.yaml with an 'architecture' field, or pass "
        f"family= explicitly to get_adapter()."
    )


@dataclass
class ExperimentManifest:
    path: Path
    culture_locales: list[str]
    reasoning_locales: list[str]
    models: list[ModelSpec]


def load_manifest(path: str | Path | None = None) -> ExperimentManifest:
    import yaml

    resolved = Path(path) if path is not None else MANIFEST_PATH
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ExperimentManifest(
        path=resolved,
        culture_locales=payload.get("culture_locales", []),
        reasoning_locales=payload.get("reasoning_locales", []),
        models=load_enabled_models(resolved),
    )
