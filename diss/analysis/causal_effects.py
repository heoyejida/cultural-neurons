#!/usr/bin/env python3
"""Build the causal-summary table, sensitivity values, and BLEnD forest plot.

This script is deliberately standalone: it reads frozen result artefacts and
uses only pandas, NumPy, Matplotlib, and the Python standard library.  It does
not import the ``diss`` package or execute any model code.

Outputs
-------
* ``tables/tab_causal_summary.tex``
* ``framing_sensitivity_values.csv``
* ``pair_slr_gap_values.csv``
* ``tables/tab_pair_slr_gap.tex``
* ``figures/fig_pair_slr_gap.pdf``
* ``figures/fig_pair_slr_gap.png``

The summary point estimate is the frozen mean over the 36 model--locale pairs.
Its SLR gap interval uses two stages of resampling with 10,000 draws.
Each replicate resamples whole model clusters, retaining all locales contributed
by a sampled model, and resamples source questions within every sampled pair.
The five saved same-layer masks are averaged before resampling.

For the forest plot, each BLEnD interval uses paired resampling over
source questions (``org_id`` in eBLEnD). Resampling a source question moves
all of its row variants together and pairs the selected outcome with the mean
outcome across the five saved same-layer masks.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cultural-neurons-causal-effects-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "diss" / "outputs" / "analysis"
OUTPUT_ROOT = PROJECT_ROOT / "diss" / "outputs"

PER_CELL_PATH = OUTPUT_ROOT / "masking_significance_per_cell.csv"
SIGNIFICANCE_JSON_PATH = OUTPUT_ROOT / "masking_significance.json"
EBLEND_PATH = PROJECT_ROOT / "diss" / "data" / "eBLEnD" / "eblend_final.csv"
MODERATION_CELLS_PATH = OUTPUT_ROOT / "masking_moderation_cells.csv"

TABLE_PATH = ANALYSIS_ROOT / "tables" / "tab_causal_summary.tex"
FRAMING_VALUES_PATH = ANALYSIS_ROOT / "framing_sensitivity_values.csv"
VALUES_PATH = ANALYSIS_ROOT / "pair_slr_gap_values.csv"
PAIR_TABLE_PATH = ANALYSIS_ROOT / "tables" / "tab_pair_slr_gap.tex"
FIGURE_PDF_PATH = ANALYSIS_ROOT / "figures" / "fig_pair_slr_gap.pdf"
FIGURE_PNG_PATH = ANALYSIS_ROOT / "figures" / "fig_pair_slr_gap.png"

BOOTSTRAP_B = 10_000
ALPHA = 0.05
EXPECTED_PAIRS = 36
EXPECTED_DATASETS = ("blend_mcq", "belebele")
SLR_MASK_SEEDS = tuple(range(5))
NON_NEURON_TOPICS = {
    "Education",
    "Family",
    "Holidays/Celebration/Leisure",
}

DIRECT_CONTRAST_SEED_OFFSET = 10
NON_KOREAN_SEED_OFFSET = 20

_PAIR_GAP_CACHE: dict[
    tuple[str, str, str], tuple[np.ndarray, np.ndarray, float]
] = {}
_PAIR_MASK_GAP_CACHE: dict[tuple[str, str, str], np.ndarray] = {}

DATASET_LABEL = {
    "blend_mcq": "Extended BLEnD",
    "belebele": "Belebele",
}

LOCALE_LABEL = {
    "ko": "ko-KR",
    "es": "es-ES",
    "zh": "zh-CN",
    "ar": "ar-SA",
    "fr": "fr-FR",
}

MODEL_LABEL = {
    "goldfish-ko": "Goldfish Korean 1GB",
    "goldfish-es": "Goldfish Spanish 1GB",
    "goldfish-zh": "Goldfish Chinese 1GB",
    "goldfish-ar": "Goldfish Arabic 1GB",
    "goldfish-fr": "Goldfish French 1GB",
    "ko-en-intermediate": "Korean-English Intermediate",
    "es-en-intermediate": "Spanish-English Intermediate",
    "zh-en-intermediate": "Chinese-English Intermediate",
    "ar-en-intermediate": "Arabic-English Intermediate",
    "fr-en-intermediate": "French-English Intermediate",
    "xglm-564m": "XGLM 564M",
    "qwen3-1p7b-base": "Qwen3 1.7B Base",
    "kanana-1p5-2p1b-base": "Kanana 1.5 2.1B Base",
    "croissantllm-base": "CroissantLLM Base",
    "jais-family-1p3b": "Jais Family 1.3B",
    "hplt2c-kor": "HPLT 2.0 Korean",
    "hplt2c-ara": "HPLT 2.0 Arabic",
    "hplt2c-fra": "HPLT 2.0 French",
    "tiny-aya-base": "Tiny Aya Base",
    "eurollm-1p7b": "EuroLLM 1.7B",
}

PLOT_MODEL_LABEL = {
    "goldfish-ko": "Goldfish-ko",
    "goldfish-es": "Goldfish-es",
    "goldfish-zh": "Goldfish-zh",
    "goldfish-ar": "Goldfish-ar",
    "goldfish-fr": "Goldfish-fr",
    "ko-en-intermediate": "Korean-English",
    "es-en-intermediate": "Spanish-English",
    "zh-en-intermediate": "Chinese-English",
    "ar-en-intermediate": "Arabic-English",
    "fr-en-intermediate": "French-English",
    "xglm-564m": "XGLM-564M",
    "qwen3-1p7b-base": "Qwen3-1.7B",
    "kanana-1p5-2p1b-base": "Kanana-2.1B",
    "croissantllm-base": "CroissantLLM-1.3B",
    "jais-family-1p3b": "Jais-1.3B",
    "hplt2c-kor": "HPLT-kor",
    "hplt2c-ara": "HPLT-ara",
    "hplt2c-fra": "HPLT-fra",
    "tiny-aya-base": "Tiny Aya-3.35B",
    "eurollm-1p7b": "EuroLLM-1.7B",
}

REGIME_STYLE = {
    "monolingual": {"color": "#222222", "marker": "o", "label": "Monolingual"},
    "bilingual": {"color": "#0072B2", "marker": "s", "label": "Bilingual"},
    "multilingual": {"color": "#D55E00", "marker": "^", "label": "Multilingual"},
}

LOCALE_STYLE = {
    "ko": {"marker": "o", "label": "ko-KR"},
    "es": {"marker": "s", "label": "es-ES"},
    "zh": {"marker": "^", "label": "zh-CN"},
    "ar": {"marker": "D", "label": "ar-SA"},
    "fr": {"marker": "P", "label": "fr-FR"},
}

REQUIRED_COLUMNS = {
    "model_id",
    "slug",
    "regime",
    "locale",
    "country",
    "dataset",
    "revision",
    "n_items",
    "unmasked_acc",
    "cultural_masked_acc",
    "cultural_drop",
    "slr_drop_mean",
    "random_drop_mean",
}


def fail(message: str) -> None:
    """Raise a concise validation error."""

    raise RuntimeError(message)


def load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load and validate the frozen per-cell, metadata, and eBLEnD inputs."""

    for path in (PER_CELL_PATH, SIGNIFICANCE_JSON_PATH, EBLEND_PATH):
        if not path.is_file():
            fail(f"required input is missing: {path}")

    cells = pd.read_csv(PER_CELL_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(cells.columns))
    if missing:
        fail(f"{PER_CELL_PATH} is missing columns: {', '.join(missing)}")

    with SIGNIFICANCE_JSON_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        fail(f"{SIGNIFICANCE_JSON_PATH} must contain a JSON object")
    if int(metadata.get("b_item", -1)) != BOOTSTRAP_B:
        fail(
            f"expected upstream item bootstrap B={BOOTSTRAP_B}, "
            f"found {metadata.get('b_item')!r}"
        )
    try:
        bootstrap_seed = int(metadata["rng_seed"])
        aggregate = metadata["aggregate"]["by_dataset"]
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid significance metadata: {exc}")
    if bootstrap_seed < 0:
        fail("rng_seed in significance metadata must be non-negative")

    eblend = pd.read_csv(EBLEND_PATH, usecols=["id", "org_id"])
    if eblend[["id", "org_id"]].isna().any().any():
        fail(f"{EBLEND_PATH} contains missing id/org_id values")
    eblend["id"] = eblend["id"].astype(str)
    eblend["org_id"] = eblend["org_id"].astype(str)
    if eblend["id"].duplicated().any():
        duplicates = eblend.loc[eblend["id"].duplicated(), "id"].head().tolist()
        fail(f"eBLEnD ids are not unique; examples: {duplicates}")

    for dataset in EXPECTED_DATASETS:
        reported = int(aggregate[dataset]["vs_slr"]["n_cells"])
        if reported != EXPECTED_PAIRS:
            fail(
                f"JSON reports {reported} {dataset} cells; expected {EXPECTED_PAIRS}"
            )

    return cells, metadata, eblend


def select_requested_rows(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Select the 36 primary BLEnD pairs and their exact Belebele matches."""

    selected: dict[str, pd.DataFrame] = {}
    for dataset in EXPECTED_DATASETS:
        frame = cells.loc[cells["dataset"].eq(dataset)].copy()
        if len(frame) != EXPECTED_PAIRS:
            fail(f"found {len(frame)} {dataset} rows; expected {EXPECTED_PAIRS}")
        if frame.duplicated(["model_id", "locale"]).any():
            fail(f"duplicate (model_id, locale) keys in {dataset} rows")
        selected[dataset] = frame.reset_index(drop=True)

    blend_keys = set(
        selected["blend_mcq"][["model_id", "locale"]].itertuples(
            index=False, name=None
        )
    )
    belebele_keys = set(
        selected["belebele"][["model_id", "locale"]].itertuples(
            index=False, name=None
        )
    )
    if blend_keys != belebele_keys:
        missing = sorted(blend_keys - belebele_keys)
        extra = sorted(belebele_keys - blend_keys)
        fail(f"Belebele pair mismatch; missing={missing}, extra={extra}")

    numeric = [
        "n_items",
        "unmasked_acc",
        "cultural_masked_acc",
        "cultural_drop",
        "slr_drop_mean",
        "random_drop_mean",
    ]
    for dataset, frame in selected.items():
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            fail(f"non-finite required numeric values in {dataset} rows")

    unknown_models = sorted(
        set(selected["blend_mcq"]["model_id"]) - set(MODEL_LABEL)
    )
    unknown_locales = sorted(
        set(selected["blend_mcq"]["locale"]) - set(LOCALE_LABEL)
    )
    unknown_regimes = sorted(
        set(selected["blend_mcq"]["regime"]) - set(REGIME_STYLE)
    )
    if unknown_models:
        fail(f"missing full display labels for models: {unknown_models}")
    if unknown_locales:
        fail(f"missing locale display labels: {unknown_locales}")
    if unknown_regimes:
        fail(f"missing regime styles: {unknown_regimes}")

    return selected


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the central 95% percentile interval."""

    lo, hi = np.quantile(values, [ALPHA / 2.0, 1.0 - ALPHA / 2.0])
    return float(lo), float(hi)


def two_stage_cluster_summary(
    frame: pd.DataFrame,
    dataset: str,
    eblend: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Compute pair means and a model plus source-question interval."""

    work = frame.copy()
    work["slr_gap"] = work["cultural_drop"] - work["slr_drop_mean"]
    stat_columns = [
        "unmasked_acc",
        "cultural_drop",
        "slr_drop_mean",
        "random_drop_mean",
        "slr_gap",
    ]
    point = work[stat_columns].mean(axis=0)

    by_model = build_pair_gap_bootstraps(work, dataset, eblend, seed=seed)
    model_ids = sorted(by_model)
    if not model_ids:
        fail("cannot bootstrap an empty model summary")

    outer_rng = np.random.default_rng(seed + 900_000)
    n_models = len(model_ids)
    model_draws = outer_rng.integers(
        0, n_models, size=(BOOTSTRAP_B, n_models)
    )
    boot_gap = np.empty(BOOTSTRAP_B, dtype=float)
    for replicate, model_draw in enumerate(model_draws):
        total = 0.0
        n_pairs = 0
        for model_index in model_draw:
            for pair_bootstrap in by_model[model_ids[int(model_index)]]:
                inner_index = int(outer_rng.integers(0, BOOTSTRAP_B))
                total += float(pair_bootstrap[inner_index])
                n_pairs += 1
        if n_pairs == 0:
            fail("two-stage bootstrap produced an empty replicate")
        boot_gap[replicate] = total / n_pairs

    ci_lo, ci_hi = percentile_interval(boot_gap)
    return {
        "n_pairs": int(len(work)),
        "n_models": int(n_models),
        "mean_unmasked": float(point["unmasked_acc"]),
        "cultural_drop": float(point["cultural_drop"]),
        "slr_drop": float(point["slr_drop_mean"]),
        "random_drop": float(point["random_drop_mean"]),
        "slr_gap": float(point["slr_gap"]),
        "slr_gap_ci_lo": ci_lo,
        "slr_gap_ci_hi": ci_hi,
    }


def format_decimal(value: float, digits: int = 3) -> str:
    """Format decimals without producing a distracting negative zero."""

    rounded = f"{value:.{digits}f}"
    negative_zero = "-0." + ("0" * digits)
    return ("0." + ("0" * digits)) if rounded == negative_zero else rounded


def write_summary_table(
    summaries: dict[str, dict[str, float | int]],
    task_contrast: dict[str, float | int],
    *,
    bootstrap_seed: int,
) -> None:
    """Write a compact booktabs LaTeX snippet."""

    lines = [
        "% Auto-generated by diss/analysis/causal_effects.py; do not edit.",
        "% N is the number of model and locale pairs. Point estimates are means across the fixed cohort.",
        (
            "% The SLR gap confidence interval uses two stages of resampling "
            f"with {BOOTSTRAP_B} draws and base seed {bootstrap_seed}."
        ),
        r"\begin{tabular}{llrrrrrc}",
        r"\toprule",
        (
            r"Dataset & Control & $N$ & \shortstack{Mean\\unmasked} & "
            r"\shortstack{Cultural mask\\drop} & \shortstack{Control\\drop} "
            r"& Gap & 95\% CI \\"
        ),
        r"\midrule",
    ]
    blend = summaries["blend_mcq"]
    blend_interval = (
        f"$[{format_decimal(float(blend['slr_gap_ci_lo']))},\\,"
        f"{format_decimal(float(blend['slr_gap_ci_hi']))}]$"
    )
    lines.append(
        f"Extended BLEnD & SLR & {int(blend['n_pairs'])} & "
        f"{format_decimal(float(blend['mean_unmasked']))} & "
        f"{format_decimal(float(blend['cultural_drop']))} & "
        f"{format_decimal(float(blend['slr_drop']))} & "
        f"{format_decimal(float(blend['slr_gap']))} & {blend_interval} \\\\"
    )
    random_gap = float(blend["cultural_drop"]) - float(blend["random_drop"])
    lines.append(
        f"Extended BLEnD & Count matched & {int(blend['n_pairs'])} & "
        f"{format_decimal(float(blend['mean_unmasked']))} & "
        f"{format_decimal(float(blend['cultural_drop']))} & "
        f"{format_decimal(float(blend['random_drop']))} & "
        f"{format_decimal(random_gap)} & -- \\\\"
    )
    belebele = summaries["belebele"]
    belebele_interval = (
        f"$[{format_decimal(float(belebele['slr_gap_ci_lo']))},\\,"
        f"{format_decimal(float(belebele['slr_gap_ci_hi']))}]$"
    )
    lines.append(
        f"Belebele & SLR & {int(belebele['n_pairs'])} & "
        f"{format_decimal(float(belebele['mean_unmasked']))} & "
        f"{format_decimal(float(belebele['cultural_drop']))} & "
        f"{format_decimal(float(belebele['slr_drop']))} & "
        f"{format_decimal(float(belebele['slr_gap']))} & {belebele_interval} \\\\"
    )
    contrast_interval = (
        f"$[{format_decimal(float(task_contrast['ci_lo']))},\\,"
        f"{format_decimal(float(task_contrast['ci_hi']))}]$"
    )
    lines.append(
        f"Extended BLEnD $-$ Belebele & SLR & {int(task_contrast['n_pairs'])} & "
        f"-- & -- & -- & {format_decimal(float(task_contrast['estimate']))} & "
        f"{contrast_interval} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def revision_directory(value: object) -> str | None:
    """Translate the per-cell revision field to its on-disk cp_* directory."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return f"cp_{text}"


def correctness(series: pd.Series, *, path: Path) -> np.ndarray:
    """Parse the byte-normalised is_correct column as 0/1 values."""

    normalised = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalised) - allowed)
    if unexpected:
        fail(f"unexpected is_correct values in {path}: {unexpected[:5]}")
    return normalised.isin({"true", "1"}).to_numpy(dtype=float)


def aggregate_result_paths(
    row: pd.Series, dataset: str
) -> tuple[Path, Path, list[Path]]:
    """Resolve unmasked, selected, and five same-layer result files."""

    if dataset not in EXPECTED_DATASETS:
        fail(f"unsupported aggregate dataset: {dataset}")

    base = OUTPUT_ROOT / str(row["slug"])
    revision = revision_directory(row["revision"])
    if revision is not None:
        base = base / revision
    locale = str(row["locale"])
    country_tag = str(row["country"]).replace(" ", "")
    if dataset == "blend_mcq":
        unmasked_subdir = f"blend_mcq_{locale}"
        masked_subdir = f"blend_mcq_{locale}_non_neuron"
    else:
        unmasked_subdir = f"belebele_{locale}"
        masked_subdir = unmasked_subdir

    unmasked = base / unmasked_subdir / "results.csv"
    selected = (
        base
        / "masked"
        / f"{country_tag}_neurons_blend_mcq"
        / masked_subdir
        / "results.csv"
    )
    slr = [
        base
        / "masked"
        / f"{country_tag}_all_neurons_random_layer{mask_seed}"
        / masked_subdir
        / "results.csv"
        for mask_seed in SLR_MASK_SEEDS
    ]
    return unmasked, selected, slr


def pair_gap_source_clusters(
    row: pd.Series,
    dataset: str,
    eblend: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return source-cluster sums and counts for one pair's SLR gap."""

    cache_key = (str(row["model_id"]), str(row["locale"]), dataset)
    cached = _PAIR_GAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    unmasked_path, selected_path, slr_paths = aggregate_result_paths(row, dataset)
    for path in (unmasked_path, selected_path, *slr_paths):
        if not path.is_file():
            fail(f"aggregate bootstrap result is missing: {path}")

    unmasked_columns = ["id", "is_correct"]
    if dataset == "blend_mcq":
        unmasked_columns.append("topic")
    unmasked = pd.read_csv(
        unmasked_path,
        usecols=unmasked_columns,
        dtype={"id": str, "topic": str},
    )
    if dataset == "blend_mcq":
        unmasked = unmasked.loc[unmasked["topic"].isin(NON_NEURON_TOPICS)].copy()
    if unmasked["id"].duplicated().any():
        fail(f"duplicate unmasked ids in {unmasked_path}")
    unmasked["u_correct"] = correctness(
        unmasked["is_correct"], path=unmasked_path
    )
    paired = unmasked[["id", "u_correct"]].copy()

    selected = pd.read_csv(
        selected_path,
        usecols=["id", "is_correct"],
        dtype={"id": str},
    )
    if selected["id"].duplicated().any():
        fail(f"duplicate selected-mask ids in {selected_path}")
    selected["selected_correct"] = correctness(
        selected["is_correct"], path=selected_path
    )
    paired = paired.merge(
        selected[["id", "selected_correct"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    slr_columns: list[str] = []
    for mask_seed, slr_path in zip(SLR_MASK_SEEDS, slr_paths):
        slr = pd.read_csv(
            slr_path,
            usecols=["id", "is_correct"],
            dtype={"id": str},
        )
        if slr["id"].duplicated().any():
            fail(f"duplicate same-layer-mask ids in {slr_path}")
        column = f"slr_correct_{mask_seed}"
        slr[column] = correctness(slr["is_correct"], path=slr_path)
        paired = paired.merge(
            slr[["id", column]],
            on="id",
            how="inner",
            validate="one_to_one",
        )
        slr_columns.append(column)

    expected_rows = int(row["n_items"])
    if len(paired) != expected_rows:
        fail(
            f"shared aggregate rows for {row['model_id']} / {row['locale']} / "
            f"{dataset}: found {len(paired)}, expected {expected_rows}"
        )

    selected_delta = paired["u_correct"] - paired["selected_correct"]
    slr_delta = paired["u_correct"].to_numpy(dtype=float)[:, None] - paired[
        slr_columns
    ].to_numpy(dtype=float)
    paired["gap"] = selected_delta.to_numpy(dtype=float) - slr_delta.mean(axis=1)
    _PAIR_MASK_GAP_CACHE[cache_key] = (
        selected_delta.to_numpy(dtype=float)[:, None] - slr_delta
    ).mean(axis=0)

    observed_selected = float(selected_delta.mean())
    observed_slr = float(slr_delta.mean(axis=0).mean())
    observed_gap = float(paired["gap"].mean())
    expected_selected = float(row["cultural_drop"])
    expected_slr = float(row["slr_drop_mean"])
    for label, observed, expected in (
        ("selected drop", observed_selected, expected_selected),
        ("SLR drop", observed_slr, expected_slr),
    ):
        if not np.isclose(observed, expected, atol=5.1e-5, rtol=0.0):
            fail(
                f"raw {label} mismatch for {row['model_id']} / "
                f"{row['locale']} / {dataset}: raw={observed:.8f}, "
                f"frozen={expected:.8f}"
            )
    raw_component_gap = observed_selected - observed_slr
    if not np.isclose(observed_gap, raw_component_gap, atol=1e-12, rtol=0.0):
        fail(
            f"paired SLR gap mismatch for {row['model_id']} / "
            f"{row['locale']} / {dataset}: paired={observed_gap:.12f}, "
            f"components={raw_component_gap:.12f}"
        )

    if dataset == "blend_mcq":
        paired["source_id"] = paired["id"].str.replace(
            r"^eblnd-", "", regex=True
        )
        paired = paired.merge(
            eblend,
            left_on="source_id",
            right_on="id",
            how="left",
            validate="many_to_one",
            suffixes=("_result", "_eblend"),
        )
        if paired["org_id"].isna().any():
            examples = paired.loc[paired["org_id"].isna(), "source_id"].head()
            fail(f"aggregate ids missing from eBLEnD mapping: {examples.tolist()}")
        source_column = "org_id"
    else:
        paired["source_question"] = paired["id"]
        source_column = "source_question"

    clusters = paired.groupby(source_column, sort=True)["gap"].agg(["sum", "count"])
    if clusters.empty:
        fail(
            f"no aggregate source clusters for {row['model_id']} / "
            f"{row['locale']} / {dataset}"
        )
    result = (
        clusters["sum"].to_numpy(dtype=float),
        clusters["count"].to_numpy(dtype=float),
        observed_gap,
    )
    _PAIR_GAP_CACHE[cache_key] = result
    return result


def pair_mask_seed_gaps(
    row: pd.Series,
    dataset: str,
    eblend: pd.DataFrame,
) -> np.ndarray:
    """Return selected-minus-SLR gaps for each saved mask seed."""

    cache_key = (str(row["model_id"]), str(row["locale"]), dataset)
    pair_gap_source_clusters(row, dataset, eblend)
    gaps = _PAIR_MASK_GAP_CACHE.get(cache_key)
    if gaps is None or gaps.shape != (len(SLR_MASK_SEEDS),):
        fail(
            f"missing per-mask gaps for {row['model_id']} / "
            f"{row['locale']} / {dataset}"
        )
    return gaps


def source_bootstrap_vector(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Bootstrap one pair in bounded-memory batches."""

    if len(sums) == 0 or len(sums) != len(counts):
        fail("invalid source clusters for aggregate bootstrap")
    rng = np.random.default_rng(seed)
    result = np.empty(BOOTSTRAP_B, dtype=float)
    batch_size = 512
    for start in range(0, BOOTSTRAP_B, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_B)
        draw = rng.integers(0, len(sums), size=(stop - start, len(sums)))
        result[start:stop] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return result


def build_pair_gap_bootstrap_lookup(
    frame: pd.DataFrame,
    dataset: str,
    eblend: pd.DataFrame,
    *,
    seed: int,
    pair_seed_offset: int = 300_000,
) -> dict[tuple[str, str], np.ndarray]:
    """Build source-question bootstrap vectors keyed by aligned pair."""

    ordered = frame.sort_values(["model_id", "locale"]).reset_index(drop=True)
    lookup: dict[tuple[str, str], np.ndarray] = {}
    observed_gaps: list[float] = []
    for pair_index, (_, row) in enumerate(ordered.iterrows()):
        sums, counts, observed_gap = pair_gap_source_clusters(row, dataset, eblend)
        pair_seed = seed + pair_seed_offset + pair_index
        bootstrap = source_bootstrap_vector(sums, counts, seed=pair_seed)
        key = (str(row["model_id"]), str(row["locale"]))
        if key in lookup:
            fail(f"duplicate bootstrap pair for {dataset}: {key}")
        lookup[key] = bootstrap
        observed_gaps.append(observed_gap)

    frozen_mean = float(
        (ordered["cultural_drop"] - ordered["slr_drop_mean"]).mean()
    )
    if not np.isclose(
        float(np.mean(observed_gaps)), frozen_mean, atol=5.1e-5, rtol=0.0
    ):
        fail(
            f"aggregate raw-gap mean mismatch for {dataset}: "
            f"raw={float(np.mean(observed_gaps)):.8f}, frozen={frozen_mean:.8f}"
        )
    return lookup


def build_pair_gap_bootstraps(
    frame: pd.DataFrame,
    dataset: str,
    eblend: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, list[np.ndarray]]:
    """Build source-question bootstrap vectors grouped by model cluster."""

    lookup = build_pair_gap_bootstrap_lookup(
        frame, dataset, eblend, seed=seed
    )
    by_model: dict[str, list[np.ndarray]] = {}
    for (model_id, _locale), bootstrap in sorted(lookup.items()):
        by_model.setdefault(model_id, []).append(bootstrap)
    return by_model


def direct_task_contrast_summary(
    blend: pd.DataFrame,
    belebele: pd.DataFrame,
    eblend: pd.DataFrame,
    *,
    base_seed: int,
) -> dict[str, float | int]:
    """Contrast task gaps with aligned pairs and independent task resampling."""

    blend_lookup = build_pair_gap_bootstrap_lookup(
        blend,
        "blend_mcq",
        eblend,
        seed=base_seed,
        pair_seed_offset=300_000,
    )
    belebele_lookup = build_pair_gap_bootstrap_lookup(
        belebele,
        "belebele",
        eblend,
        seed=base_seed,
        pair_seed_offset=400_000,
    )
    if set(blend_lookup) != set(belebele_lookup):
        fail("direct task contrast requires identical model-locale keys")

    aligned = blend[["model_id", "locale", "cultural_drop", "slr_drop_mean"]].merge(
        belebele[
            ["model_id", "locale", "cultural_drop", "slr_drop_mean"]
        ],
        on=["model_id", "locale"],
        how="inner",
        validate="one_to_one",
        suffixes=("_blend", "_belebele"),
    )
    if len(aligned) != EXPECTED_PAIRS:
        fail(
            f"direct task contrast found {len(aligned)} aligned pairs; "
            f"expected {EXPECTED_PAIRS}"
        )
    aligned["contrast"] = (
        aligned["cultural_drop_blend"]
        - aligned["slr_drop_mean_blend"]
        - aligned["cultural_drop_belebele"]
        + aligned["slr_drop_mean_belebele"]
    )
    point = float(aligned["contrast"].mean())

    model_pairs: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(blend_lookup):
        model_pairs.setdefault(key[0], []).append(key)
    model_ids = sorted(model_pairs)
    n_models = len(model_ids)
    outer_seed = base_seed + 1_300_000
    rng = np.random.default_rng(outer_seed)
    model_draws = rng.integers(0, n_models, size=(BOOTSTRAP_B, n_models))
    bootstrap = np.empty(BOOTSTRAP_B, dtype=float)
    for replicate, draw in enumerate(model_draws):
        total = 0.0
        n_pairs = 0
        for model_index in draw:
            model_id = model_ids[int(model_index)]
            for key in model_pairs[model_id]:
                blend_index = int(rng.integers(0, BOOTSTRAP_B))
                belebele_index = int(rng.integers(0, BOOTSTRAP_B))
                total += float(blend_lookup[key][blend_index])
                total -= float(belebele_lookup[key][belebele_index])
                n_pairs += 1
        if n_pairs == 0:
            fail("direct task bootstrap produced an empty replicate")
        bootstrap[replicate] = total / n_pairs

    ci_lo, ci_hi = percentile_interval(bootstrap)
    return {
        "n_pairs": int(len(aligned)),
        "n_models": int(n_models),
        "estimate": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "bootstrap_seed": int(outer_seed),
    }


def conditional_model_source_interval(
    frame: pd.DataFrame,
    dataset: str,
    eblend: pd.DataFrame,
    *,
    base_seed: int,
    outer_seed_offset: int,
    bootstrap_frame: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """Condition on a pair subset, then resample sources and whole models."""

    seed_frame = frame if bootstrap_frame is None else bootstrap_frame
    full_lookup = build_pair_gap_bootstrap_lookup(
        seed_frame,
        dataset,
        eblend,
        seed=base_seed,
        pair_seed_offset=300_000,
    )
    cohort_keys = set(
        frame[["model_id", "locale"]].itertuples(index=False, name=None)
    )
    missing_keys = sorted(cohort_keys - set(full_lookup))
    if missing_keys:
        fail(f"conditional bootstrap keys missing from seed frame: {missing_keys}")
    lookup = {key: full_lookup[key] for key in sorted(cohort_keys)}
    model_pairs: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(lookup):
        model_pairs.setdefault(key[0], []).append(key)
    model_ids = sorted(model_pairs)
    if not model_ids:
        fail("conditional interval requires at least one model")

    rng = np.random.default_rng(base_seed + outer_seed_offset)
    n_models = len(model_ids)
    model_draws = rng.integers(0, n_models, size=(BOOTSTRAP_B, n_models))
    bootstrap = np.empty(BOOTSTRAP_B, dtype=float)
    for replicate, draw in enumerate(model_draws):
        total = 0.0
        n_pairs = 0
        for model_index in draw:
            model_id = model_ids[int(model_index)]
            for key in model_pairs[model_id]:
                inner_index = int(rng.integers(0, BOOTSTRAP_B))
                total += float(lookup[key][inner_index])
                n_pairs += 1
        if n_pairs == 0:
            fail("conditional bootstrap produced an empty replicate")
        bootstrap[replicate] = total / n_pairs
    return percentile_interval(bootstrap)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction."""

    max_iterations = 300
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        numerator = iteration * (b - iteration) * x
        denominator = (qam + twice) * (a + twice)
        delta = numerator / denominator
        d = 1.0 + delta * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + delta / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x
        denominator = (a + twice) * (qap + twice)
        delta = numerator / denominator
        d = 1.0 + delta * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + delta / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < epsilon:
            return result
    fail("incomplete-beta continued fraction did not converge")
    raise AssertionError("unreachable")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return I_x(a,b) using only the Python standard library."""

    if a <= 0.0 or b <= 0.0 or not 0.0 <= x <= 1.0:
        fail("invalid incomplete-beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(test_statistic: float, degrees_freedom: int) -> float:
    """Return a two-sided Student-t p-value without a SciPy dependency."""

    if degrees_freedom <= 0 or not math.isfinite(test_statistic):
        fail("invalid Student-t test arguments")
    x = degrees_freedom / (degrees_freedom + test_statistic**2)
    return regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)


def load_moderation_design(blend: pd.DataFrame) -> pd.DataFrame:
    """Merge the frozen parameter metadata onto the exact primary cohort."""

    if not MODERATION_CELLS_PATH.is_file():
        fail(f"required input is missing: {MODERATION_CELLS_PATH}")
    required = {"model_id", "locale", "regime", "log10_params"}
    moderation = pd.read_csv(MODERATION_CELLS_PATH)
    missing = sorted(required - set(moderation.columns))
    if missing:
        fail(
            f"{MODERATION_CELLS_PATH} is missing columns: {', '.join(missing)}"
        )
    moderation = moderation[list(sorted(required))].copy()
    if len(moderation) != EXPECTED_PAIRS:
        fail(
            f"found {len(moderation)} moderation rows; expected {EXPECTED_PAIRS}"
        )
    if moderation.duplicated(["model_id", "locale"]).any():
        fail("duplicate model-locale keys in moderation design")
    moderation["log10_params"] = pd.to_numeric(
        moderation["log10_params"], errors="coerce"
    )
    if not np.isfinite(moderation["log10_params"].to_numpy(dtype=float)).all():
        fail("non-finite log10_params values in moderation design")

    merged = blend.merge(
        moderation,
        on=["model_id", "locale"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_moderation"),
    )
    if len(merged) != EXPECTED_PAIRS:
        fail("moderation design does not match the 36-pair primary cohort")
    if not merged["regime"].eq(merged["regime_moderation"]).all():
        fail("regime mismatch between per-cell and moderation inputs")
    return merged.sort_values(["model_id", "locale"]).reset_index(drop=True)


def clustered_baseline_ols(blend: pd.DataFrame) -> list[dict[str, float | str]]:
    """Fit the baseline-adjusted OLS with model-clustered CR1 uncertainty."""

    design = load_moderation_design(blend)
    regimes = design["regime"].astype(str)
    columns = [
        np.ones(len(design), dtype=float),
        design["log10_params"].to_numpy(dtype=float) - 9.0,
        regimes.eq("bilingual").to_numpy(dtype=float),
        regimes.eq("multilingual").to_numpy(dtype=float),
        design["unmasked_acc"].to_numpy(dtype=float) - 0.5,
    ]
    matrix = np.column_stack(columns)
    outcome = (
        design["cultural_drop"] - design["slr_drop_mean"]
    ).to_numpy(dtype=float)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        fail("baseline-adjusted OLS design matrix is rank deficient")
    coefficients = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    residuals = outcome - matrix @ coefficients

    n_rows, n_terms = matrix.shape
    model_groups = design.groupby("model_id", sort=True).indices
    n_clusters = len(model_groups)
    if n_clusters <= 1 or n_rows <= n_terms:
        fail("insufficient observations for model-clustered OLS")
    meat = np.zeros((n_terms, n_terms), dtype=float)
    for indices in model_groups.values():
        cluster_matrix = matrix[indices]
        cluster_residuals = residuals[indices]
        score = cluster_matrix.T @ cluster_residuals
        meat += np.outer(score, score)
    bread = np.linalg.inv(matrix.T @ matrix)
    correction = (n_clusters / (n_clusters - 1.0)) * (
        (n_rows - 1.0) / (n_rows - n_terms)
    )
    covariance = correction * bread @ meat @ bread
    standard_errors = np.sqrt(np.diag(covariance))
    test_statistics = coefficients / standard_errors
    degrees_freedom = n_clusters - 1
    p_values = np.array(
        [student_t_two_sided_p(float(value), degrees_freedom) for value in test_statistics]
    )

    term_names = (
        "intercept",
        "log10_params_centered_9",
        "regime_bilingual",
        "regime_multilingual",
        "unmasked_accuracy",
    )
    return [
        {
            "term": term,
            "estimate": float(coefficients[index]),
            "standard_error": float(standard_errors[index]),
            "test_statistic": float(test_statistics[index]),
            "p_value": float(p_values[index]),
        }
        for index, term in enumerate(term_names)
    ]


def cohort_pair_string(frame: pd.DataFrame) -> str:
    """Serialize a model-locale cohort in a stable, human-readable order."""

    pairs = sorted(
        set(frame[["model_id", "locale"]].itertuples(index=False, name=None))
    )
    if len(pairs) != len(frame):
        fail("sensitivity cohort contains duplicate model-locale pairs")
    return ";".join(f"{model_id}:{locale}" for model_id, locale in pairs)


def sensitivity_record(
    analysis_id: str,
    frame: pd.DataFrame,
    *,
    dataset: str = "blend_mcq",
    subset: str = "all_primary_pairs",
    weighting: str,
    estimate: float,
    ci_lo: float | None = None,
    ci_hi: float | None = None,
    standard_error: float | None = None,
    test_statistic: float | None = None,
    p_value: float | None = None,
    bootstrap_draws: int | None = None,
    bootstrap_seed: int | None = None,
    mask_seed: int | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    n_source_clusters: int | None = None,
    n_row_instances: int | None = None,
    interval_method: str = "none",
    term: str = "",
) -> dict[str, object]:
    """Build one tidy sensitivity-value record with explicit cohort metadata."""

    return {
        "analysis_id": analysis_id,
        "dataset": dataset,
        "subset": subset,
        "weighting": weighting,
        "term": term,
        "estimate": float(estimate),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "standard_error": standard_error,
        "test_statistic": test_statistic,
        "p_value": p_value,
        "range_min": range_min,
        "range_max": range_max,
        "n_pairs": int(len(frame)),
        "n_models": int(frame["model_id"].nunique()),
        "n_locales": int(frame["locale"].nunique()),
        "n_source_clusters": n_source_clusters,
        "n_row_instances": n_row_instances,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "mask_seed": mask_seed,
        "interval_method": interval_method,
        "cohort_pairs": cohort_pair_string(frame),
    }


def build_framing_sensitivity_values(
    blend: pd.DataFrame,
    belebele: pd.DataFrame,
    eblend: pd.DataFrame,
    *,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute and write all frozen-output framing sensitivity values."""

    work = blend.copy()
    work["slr_gap"] = work["cultural_drop"] - work["slr_drop_mean"]
    korean = work.loc[work["locale"].eq("ko")].copy()
    non_korean = work.loc[work["locale"].ne("ko")].copy()
    if (len(korean), len(non_korean)) != (8, 28):
        fail(
            f"expected Korean/non-Korean sensitivity cohorts of 8/28; "
            f"found {len(korean)}/{len(non_korean)}"
        )

    direct = direct_task_contrast_summary(
        work, belebele, eblend, base_seed=base_seed
    )
    task_pairs = work[["model_id", "locale", "slr_gap"]].merge(
        belebele[["model_id", "locale", "cultural_drop", "slr_drop_mean"]],
        on=["model_id", "locale"],
        how="inner",
        validate="one_to_one",
    )
    if len(task_pairs) != EXPECTED_PAIRS:
        fail(
            f"equal-model task contrast found {len(task_pairs)} aligned pairs; "
            f"expected {EXPECTED_PAIRS}"
        )
    task_pairs["contrast"] = task_pairs["slr_gap"] - (
        task_pairs["cultural_drop"] - task_pairs["slr_drop_mean"]
    )
    direct_equal_model = float(
        task_pairs.groupby("model_id", sort=True)["contrast"].mean().mean()
    )
    non_ko_ci_lo, non_ko_ci_hi = conditional_model_source_interval(
        non_korean,
        "blend_mcq",
        eblend,
        base_seed=base_seed,
        outer_seed_offset=1_400_000,
        bootstrap_frame=work,
    )

    equal_model = float(
        work.groupby("model_id", sort=True)["slr_gap"].mean().mean()
    )

    ordered = work.sort_values(["model_id", "locale"]).reset_index(drop=True)
    mask_gap_matrix = np.vstack(
        [
            pair_mask_seed_gaps(row, "blend_mcq", eblend)
            for _, row in ordered.iterrows()
        ]
    )
    mask_means = mask_gap_matrix.mean(axis=0)

    source_balanced_pair_means: list[float] = []
    source_cluster_count = 0
    row_instance_count = 0
    for _, row in ordered.iterrows():
        sums, counts, _observed_gap = pair_gap_source_clusters(
            row, "blend_mcq", eblend
        )
        source_balanced_pair_means.append(float(np.mean(sums / counts)))
        source_cluster_count += int(len(sums))
        row_instance_count += int(counts.sum())
    equal_source_within_pair = float(np.mean(source_balanced_pair_means))

    correlation = float(
        np.corrcoef(
            work["unmasked_acc"].to_numpy(dtype=float),
            work["slr_gap"].to_numpy(dtype=float),
        )[0, 1]
    )
    correlation_t = correlation * math.sqrt(
        (len(work) - 2.0) / (1.0 - correlation**2)
    )
    correlation_p = student_t_two_sided_p(correlation_t, len(work) - 2)
    ols_terms = clustered_baseline_ols(work)

    rows: list[dict[str, object]] = []
    rows.append(
        sensitivity_record(
            "direct_task_contrast",
            work,
            dataset="blend_mcq_minus_belebele",
            subset="aligned_primary_pairs",
            weighting="equal_pair_with_whole_model_outer_clusters",
            estimate=float(direct["estimate"]),
            ci_lo=float(direct["ci_lo"]),
            ci_hi=float(direct["ci_hi"]),
            bootstrap_draws=BOOTSTRAP_B,
            bootstrap_seed=int(direct["bootstrap_seed"]),
            interval_method=(
                "independent_source_questions_within_task_then_whole_models"
            ),
        )
    )
    rows.append(
        sensitivity_record(
            "direct_task_contrast_equal_model",
            work,
            dataset="blend_mcq_minus_belebele",
            subset="aligned_primary_pairs",
            weighting="equal_model_after_mean_over_model_locales",
            estimate=direct_equal_model,
        )
    )
    rows.append(
        sensitivity_record(
            "eblend_korean_pair_weighted",
            korean,
            subset="locale_equals_ko",
            weighting="equal_model_locale_pair",
            estimate=float(korean["slr_gap"].mean()),
        )
    )
    rows.append(
        sensitivity_record(
            "eblend_non_korean_pair_weighted",
            non_korean,
            subset="locale_not_ko",
            weighting="equal_model_locale_pair_with_whole_model_outer_clusters",
            estimate=float(non_korean["slr_gap"].mean()),
            ci_lo=non_ko_ci_lo,
            ci_hi=non_ko_ci_hi,
            bootstrap_draws=BOOTSTRAP_B,
            bootstrap_seed=base_seed + 1_400_000,
            interval_method="source_questions_then_whole_models",
        )
    )
    rows.append(
        sensitivity_record(
            "eblend_equal_model_weighted",
            work,
            weighting="equal_model_after_mean_over_model_locales",
            estimate=equal_model,
        )
    )
    for model_id, model_frame in work.groupby("model_id", sort=True):
        if set(model_frame["locale"]) != {"ko", "es", "zh", "ar", "fr"}:
            continue
        korean_gap = float(
            model_frame.loc[model_frame["locale"].eq("ko"), "slr_gap"].iloc[0]
        )
        other_gap = float(
            model_frame.loc[model_frame["locale"].ne("ko"), "slr_gap"].mean()
        )
        rows.append(
            sensitivity_record(
                f"eblend_korean_within_{model_id}",
                model_frame,
                subset=f"model_equals_{model_id}",
                weighting="korean_pair_minus_mean_of_four_other_locales",
                estimate=korean_gap - other_gap,
            )
        )
    for mask_index, mask_seed in enumerate(SLR_MASK_SEEDS):
        rows.append(
            sensitivity_record(
                f"eblend_slr_mask_seed_{mask_seed}",
                work,
                weighting="equal_model_locale_pair_single_slr_mask",
                estimate=float(mask_means[mask_index]),
                mask_seed=mask_seed,
            )
        )
    rows.append(
        sensitivity_record(
            "eblend_slr_mask_range",
            work,
            weighting="range_across_five_equal_pair_slr_mask_estimates",
            estimate=float(mask_means.max() - mask_means.min()),
            range_min=float(mask_means.min()),
            range_max=float(mask_means.max()),
        )
    )
    rows.append(
        sensitivity_record(
            "eblend_equal_source_weighted",
            work,
            weighting=(
                "equal_source_within_pair_then_equal_model_locale_pair"
            ),
            estimate=equal_source_within_pair,
            n_source_clusters=source_cluster_count,
            n_row_instances=row_instance_count,
        )
    )
    rows.append(
        sensitivity_record(
            "eblend_unmasked_gap_pearson",
            work,
            weighting="pairwise_pearson_correlation",
            estimate=correlation,
            test_statistic=correlation_t,
            p_value=correlation_p,
            term="unmasked_accuracy_vs_slr_gap",
        )
    )
    for result in ols_terms:
        term = str(result["term"])
        rows.append(
            sensitivity_record(
                f"eblend_ols_{term}",
                work,
                weighting="model_clustered_ols_cr1_student_t_df_19",
                estimate=float(result["estimate"]),
                standard_error=float(result["standard_error"]),
                test_statistic=float(result["test_statistic"]),
                p_value=float(result["p_value"]),
                term=term,
            )
        )

    values = pd.DataFrame(rows)
    for integer_column in ("bootstrap_draws", "bootstrap_seed", "mask_seed"):
        values[integer_column] = values[integer_column].map(
            lambda value: "" if pd.isna(value) else str(int(value))
        )
    FRAMING_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(
        FRAMING_VALUES_PATH,
        index=False,
        float_format="%.12f",
        na_rep="",
    )
    return values, direct


def build_pair_values(
    blend: pd.DataFrame, eblend: pd.DataFrame, *, bootstrap_seed: int
) -> pd.DataFrame:
    """Attach paired gap intervals, readable labels, and plot ordering."""

    ordered_for_seeds = blend.sort_values(["model_id", "locale"]).reset_index(drop=True)
    interval_records: list[dict[str, float | int | str]] = []
    for pair_index, (_, row) in enumerate(ordered_for_seeds.iterrows()):
        sums, counts, observed_gap = pair_gap_source_clusters(
            row, "blend_mcq", eblend
        )
        mask_gaps = pair_mask_seed_gaps(row, "blend_mcq", eblend)
        pair_seed = bootstrap_seed + 100_000 + pair_index
        boot_gap = source_bootstrap_vector(sums, counts, seed=pair_seed)
        ci_lo, ci_hi = percentile_interval(boot_gap)
        interval_records.append(
            {
                "model_id": str(row["model_id"]),
                "locale": str(row["locale"]),
                "slr_gap_ci_lo": ci_lo,
                "slr_gap_ci_hi": ci_hi,
                "slr_gap_mask_min": float(mask_gaps.min()),
                "slr_gap_mask_max": float(mask_gaps.max()),
                "n_source_questions": int(len(sums)),
                "n_row_instances": int(counts.sum()),
                "raw_slr_gap": observed_gap,
                "source_cluster_bootstrap_seed": pair_seed,
            }
        )

    values = blend.merge(
        pd.DataFrame(interval_records),
        on=["model_id", "locale"],
        how="left",
        validate="one_to_one",
    )
    values["model_label"] = values["model_id"].map(MODEL_LABEL)
    values["locale_label"] = values["locale"].map(LOCALE_LABEL)
    values["plot_marker"] = values["locale"].map(
        {locale: style["marker"] for locale, style in LOCALE_STYLE.items()}
    )
    values["slr_gap"] = values["cultural_drop"] - values["slr_drop_mean"]
    values = values.sort_values(
        ["slr_gap", "model_label", "locale_label"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    values.insert(0, "rank", np.arange(1, len(values) + 1, dtype=int))
    values["source_cluster_bootstrap_b"] = BOOTSTRAP_B

    columns = [
        "rank",
        "model_id",
        "model_label",
        "slug",
        "locale",
        "locale_label",
        "plot_marker",
        "country",
        "regime",
        "n_items",
        "n_source_questions",
        "n_row_instances",
        "unmasked_acc",
        "cultural_masked_acc",
        "cultural_drop",
        "slr_drop_mean",
        "slr_gap",
        "slr_gap_ci_lo",
        "slr_gap_ci_hi",
        "slr_gap_mask_min",
        "slr_gap_mask_max",
        "raw_slr_gap",
        "source_cluster_bootstrap_b",
        "source_cluster_bootstrap_seed",
    ]
    result = values[columns].copy()
    VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(VALUES_PATH, index=False, float_format="%.6f")
    return result


def write_pair_table(values: pd.DataFrame) -> None:
    """Write the pair-level appendix table without significance markings."""

    lines = [
        "% Auto-generated by diss/analysis/causal_effects.py; do not edit.",
        (
            "% SLR gap intervals use paired resampling of source questions "
            f"with {BOOTSTRAP_B} draws."
        ),
        r"\begin{tabular}{llrcrr}",
        r"\toprule",
        (
            r"Model & Locale & Cultural mask drop & SLR mask drop & SLR gap "
            r"& Gap 95\% CI \\"
        ),
        r"\midrule",
    ]
    locale_order = {locale: index for index, locale in enumerate(LOCALE_LABEL)}
    table_values = values.assign(
        _locale_order=values["locale"].map(locale_order)
    ).sort_values(
        ["_locale_order", "slr_gap", "rank"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    for locale_index, locale in enumerate(LOCALE_LABEL):
        if locale_index > 0:
            lines.append(r"\midrule")
        locale_values = table_values[table_values["locale"] == locale]
        for _, row in locale_values.iterrows():
            interval = (
                f"$[{format_decimal(float(row['slr_gap_ci_lo']))},\\,"
                f"{format_decimal(float(row['slr_gap_ci_hi']))}]$"
            )
            lines.append(
                f"{row['model_label']} & {row['locale_label']} & "
                f"{format_decimal(float(row['cultural_drop']))} & "
                f"{format_decimal(float(row['slr_drop_mean']))} & "
                f"{format_decimal(float(row['slr_gap']))} & {interval} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    PAIR_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAIR_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_pair_values(values: pd.DataFrame) -> None:
    """Render a two-column, sorted, grayscale-readable SLR-gap forest plot."""

    if len(values) != EXPECTED_PAIRS:
        fail(f"plot requires {EXPECTED_PAIRS} rows, received {len(values)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    midpoint = len(values) // 2
    panels = (values.iloc[:midpoint], values.iloc[midpoint:])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4), sharex=True)

    all_x = np.concatenate(
        [
            values["slr_gap_ci_lo"].to_numpy(dtype=float),
            values["slr_gap_ci_hi"].to_numpy(dtype=float),
            values["slr_gap_mask_min"].to_numpy(dtype=float),
            values["slr_gap_mask_max"].to_numpy(dtype=float),
            np.array([0.0]),
        ]
    )
    span = float(all_x.max() - all_x.min())
    padding = max(0.025, 0.045 * span)
    x_limits = (float(all_x.min() - padding), float(all_x.max() + padding))

    for ax, panel in zip(axes, panels):
        panel = panel.reset_index(drop=True)
        y = np.arange(len(panel), dtype=float)
        for yi, (_, row) in zip(y, panel.iterrows()):
            style = LOCALE_STYLE[str(row["locale"])]
            center = float(row["slr_gap"])
            lo = float(row["slr_gap_ci_lo"])
            hi = float(row["slr_gap_ci_hi"])
            mask_lo = float(row["slr_gap_mask_min"])
            mask_hi = float(row["slr_gap_mask_max"])
            ax.hlines(
                yi,
                mask_lo,
                mask_hi,
                color="0.76",
                linewidth=4.0,
                zorder=1,
            )
            ax.errorbar(
                center,
                yi,
                xerr=np.array([[center - lo], [hi - center]]),
                fmt=style["marker"],
                markersize=5.4,
                markerfacecolor="#222222",
                markeredgecolor="black",
                markeredgewidth=0.55,
                color="#222222",
                ecolor="#222222",
                elinewidth=1.05,
                capsize=2.1,
                capthick=0.9,
                zorder=3,
            )
        labels = [
            f"{int(row['rank']):02d}  {PLOT_MODEL_LABEL[row['model_id']]} - "
            f"{row['locale_label']}"
            for _, row in panel.iterrows()
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(*x_limits)
        ax.axvline(0.0, color="0.25", linewidth=0.8, zorder=0)
        ax.grid(axis="x", color="0.83", linewidth=0.55, linestyle=(0, (2, 2)))
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelbottom=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("0.35")

    fig.supxlabel("SLR gap (Cultural mask drop minus mean SLR drop)", y=0.045)

    locale_handles = [
        Line2D(
            [0],
            [0],
            marker=LOCALE_STYLE[locale]["marker"],
            markerfacecolor="#222222",
            markeredgecolor="black",
            markeredgewidth=0.55,
            color="none",
            markersize=5.8,
            label=LOCALE_STYLE[locale]["label"],
        )
        for locale in ("ko", "es", "zh", "ar", "fr")
    ]
    fig.legend(
        handles=locale_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=5,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.985, bottom=0.13, wspace=0.46)

    FIGURE_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF_PATH, bbox_inches="tight")
    fig.savefig(FIGURE_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cells, metadata, eblend = load_inputs()
    selected = select_requested_rows(cells)
    base_seed = int(metadata["rng_seed"])

    summaries = {
        dataset: two_stage_cluster_summary(
            frame,
            dataset,
            eblend,
            seed=base_seed + index,
        )
        for index, (dataset, frame) in enumerate(selected.items())
    }
    framing_values, task_contrast = build_framing_sensitivity_values(
        selected["blend_mcq"],
        selected["belebele"],
        eblend,
        base_seed=base_seed,
    )
    write_summary_table(
        summaries, task_contrast, bootstrap_seed=base_seed
    )
    values = build_pair_values(
        selected["blend_mcq"], eblend, bootstrap_seed=base_seed
    )
    direct_gap_screen = int((values["slr_gap_ci_lo"] > 0.0).sum())
    write_pair_table(values)
    plot_pair_values(values)

    print(
        f"wrote {TABLE_PATH} (2 datasets; {EXPECTED_PAIRS} pairs each; "
        f"direct task contrast; two-stage model/source B={BOOTSTRAP_B})"
    )
    print(
        f"wrote {FRAMING_VALUES_PATH} ({len(framing_values)} deterministic "
        f"sensitivity rows)"
    )
    print(
        f"wrote {VALUES_PATH} ({len(values)} BLEnD pairs; "
        f"source-question-cluster B={BOOTSTRAP_B}; "
        f"direct-gap lower bound above zero={direct_gap_screen})"
    )
    print(f"wrote {PAIR_TABLE_PATH} ({len(values)} BLEnD pairs)")
    print(f"wrote {FIGURE_PDF_PATH}")
    print(f"wrote {FIGURE_PNG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
