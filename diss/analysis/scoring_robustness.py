#!/usr/bin/env python3
"""Build saved-output scoring and cross-locale-overlap robustness tables.

This script is deliberately standalone.  It reads completed result CSV/JSON
artefacts, uses only pandas, NumPy, and the Python standard library, and does
not import ``diss`` or execute a model.  The 36 model--locale cohort is taken
verbatim from ``masking_significance_per_cell.csv``.

The scoring analysis holds the byte-attribution-selected MLP sets fixed and
only re-scores their saved behavioural outputs.  It therefore is not a neuron
reselection analysis.  Extended BLEnD is restricted to the held-out topics
used by the causal analysis.

The overlap analysis uses the five saved layer-matched random sets for each of
five locales.  Its null distribution enumerates all 5! assignments of distinct
random seeds to locales, then computes the mean of the ten locale-pair
Jaccards. The comparison is reported descriptively because only five saved
masks are available for each locale.

Outputs
-------
* ``scoring_robustness_values.csv``
* ``tables/tab_scoring_robustness.tex``
* ``overlap_null_values.csv``
* ``tables/tab_overlap_null.tex``
"""

from __future__ import annotations

from collections import Counter
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "diss" / "outputs" / "analysis"
OUTPUT_ROOT = PROJECT_ROOT / "diss" / "outputs"

PER_CELL_PATH = OUTPUT_ROOT / "masking_significance_per_cell.csv"
SIGNIFICANCE_PATH = OUTPUT_ROOT / "masking_significance.json"

SCORING_VALUES_PATH = ANALYSIS_ROOT / "scoring_robustness_values.csv"
SCORING_TABLE_PATH = OUTPUT_ROOT / "tables" / "tab_scoring_robustness.tex"
OVERLAP_VALUES_PATH = ANALYSIS_ROOT / "overlap_null_values.csv"
OVERLAP_TABLE_PATH = OUTPUT_ROOT / "tables" / "tab_overlap_null.tex"

EXPECTED_PAIRS = 36
EXPECTED_DATASETS = ("blend_mcq", "belebele", "countryrc_localized")
SLR_SEEDS = tuple(range(5))
NON_NEURON_TOPICS = {
    "Education",
    "Family",
    "Holidays/Celebration/Leisure",
}

SCORING_COLUMNS = {
    "byte": "is_correct",
    "token": "is_correct_token",
    "sum": "is_correct_sum",
}
SCORING_LABELS = {
    "byte": "Normalised by bytes",
    "token": "Normalised by tokens",
    "sum": "Summed NLL",
}
DATASET_LABELS = {
    "blend_mcq": "Extended BLEnD",
    "belebele": "Belebele",
    "countryrc_localized": "CountryRC",
}

OVERLAP_MODELS = {
    "qwen3-1p7b-base": "Qwen3 1.7B",
    "eurollm-1p7b": "EuroLLM 1.7B",
    "tiny-aya-base": "Tiny Aya",
}
OVERLAP_LOCALES = ("ko", "es", "zh", "ar", "fr")

REQUIRED_CELL_COLUMNS = {
    "model_id",
    "slug",
    "locale",
    "country",
    "dataset",
    "revision",
    "n_items",
    "unmasked_acc",
    "cultural_drop",
    "slr_drop_mean",
}


def fail(message: str) -> None:
    """Raise a concise validation error."""

    raise RuntimeError(message)


def load_cohort() -> tuple[pd.DataFrame, dict]:
    """Load and strictly validate the frozen 36-pair, three-dataset cohort."""

    for path in (PER_CELL_PATH, SIGNIFICANCE_PATH):
        if not path.is_file():
            fail(f"required input is missing: {path}")

    cells = pd.read_csv(PER_CELL_PATH)
    missing = sorted(REQUIRED_CELL_COLUMNS - set(cells.columns))
    if missing:
        fail(f"{PER_CELL_PATH} is missing columns: {', '.join(missing)}")

    if set(cells["dataset"]) != set(EXPECTED_DATASETS):
        fail(
            "unexpected dataset cohort: "
            f"{sorted(set(cells['dataset']))}; expected {list(EXPECTED_DATASETS)}"
        )

    reference_keys: set[tuple[str, str]] | None = None
    for dataset in EXPECTED_DATASETS:
        frame = cells.loc[cells["dataset"].eq(dataset)]
        if len(frame) != EXPECTED_PAIRS:
            fail(f"found {len(frame)} {dataset} rows; expected {EXPECTED_PAIRS}")
        if frame.duplicated(["model_id", "locale"]).any():
            fail(f"duplicate (model_id, locale) keys for {dataset}")
        keys = set(frame[["model_id", "locale"]].itertuples(index=False, name=None))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            fail(f"model--locale cohort differs for {dataset}")

    if cells["locale"].eq("ja").any():
        fail("Japanese cells must not enter the frozen 36-pair cohort")
    if len(cells) != EXPECTED_PAIRS * len(EXPECTED_DATASETS):
        fail(f"found {len(cells)} total cells; expected 108")

    cells = cells.copy()
    cells["n_items"] = pd.to_numeric(cells["n_items"], errors="raise").astype(int)
    for column in ("unmasked_acc", "cultural_drop", "slr_drop_mean"):
        cells[column] = pd.to_numeric(cells[column], errors="raise")
    if not np.isfinite(
        cells[["unmasked_acc", "cultural_drop", "slr_drop_mean"]].to_numpy()
    ).all():
        fail("non-finite frozen accuracy/drop values")

    with SIGNIFICANCE_PATH.open("r", encoding="utf-8") as handle:
        significance = json.load(handle)
    try:
        aggregate = significance["aggregate"]["by_dataset"]
        for dataset in EXPECTED_DATASETS:
            if int(aggregate[dataset]["vs_slr"]["n_cells"]) != EXPECTED_PAIRS:
                fail(f"significance JSON does not report 36 {dataset} cells")
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid significance JSON: {exc}")

    return cells.reset_index(drop=True), significance


def revision_directory(value: object) -> str | None:
    """Translate a revision value to its optional on-disk ``cp_*`` directory."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return f"cp_{text}"


def cell_base(row: pd.Series) -> Path:
    """Return the output directory for one frozen cell."""

    base = OUTPUT_ROOT / str(row["slug"])
    revision = revision_directory(row["revision"])
    if revision is not None:
        base = base / revision
    return base


def dataset_leaf(dataset: str, locale: str, *, masked: bool) -> str:
    """Return the dataset result directory used by saved evaluations."""

    if dataset == "blend_mcq":
        suffix = "_non_neuron" if masked else ""
        return f"blend_mcq_{locale}{suffix}"
    if dataset == "belebele":
        return f"belebele_{locale}"
    if dataset == "countryrc_localized":
        return f"countryrc_localized_{locale}_non_neuron"
    fail(f"unsupported dataset: {dataset}")


def result_paths(row: pd.Series) -> tuple[Path, Path, list[Path]]:
    """Resolve unmasked, selected-mask, and five SLR-mask result CSVs."""

    base = cell_base(row)
    dataset = str(row["dataset"])
    locale = str(row["locale"])
    country_tag = str(row["country"]).replace(" ", "")
    unmasked = base / dataset_leaf(dataset, locale, masked=False) / "results.csv"
    masked_leaf = dataset_leaf(dataset, locale, masked=True)
    selected = (
        base
        / "masked"
        / f"{country_tag}_neurons_blend_mcq"
        / masked_leaf
        / "results.csv"
    )
    slr = [
        base
        / "masked"
        / f"{country_tag}_all_neurons_random_layer{seed}"
        / masked_leaf
        / "results.csv"
        for seed in SLR_SEEDS
    ]
    return unmasked, selected, slr


def parse_correctness(series: pd.Series, *, path: Path, column: str) -> np.ndarray:
    """Parse a saved correctness column into a numeric zero/one array."""

    if series.isna().any():
        fail(f"missing values in {column}: {path}")
    normalised = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "1.0", "0.0"}
    unexpected = sorted(set(normalised) - allowed)
    if unexpected:
        fail(f"unexpected {column} values in {path}: {unexpected[:5]}")
    return normalised.isin({"true", "1", "1.0"}).to_numpy(dtype=float)


def read_behaviour(path: Path, dataset: str) -> pd.DataFrame:
    """Read one saved result and apply the causal analysis's row restriction."""

    if not path.is_file():
        fail(f"required result is missing: {path}")
    required = ["id", "is_correct", "is_correct_byte", "is_correct_token", "is_correct_sum"]
    if dataset == "blend_mcq":
        required.append("topic")
    frame = pd.read_csv(path, usecols=required, dtype={"id": str})
    if dataset == "blend_mcq":
        frame = frame.loc[frame["topic"].isin(NON_NEURON_TOPICS)].copy()
    if frame.empty:
        fail(f"no evaluation rows after restriction: {path}")
    if frame["id"].duplicated().any():
        fail(f"duplicate result ids: {path}")

    parsed = pd.DataFrame({"id": frame["id"].astype(str)})
    for scoring, column in SCORING_COLUMNS.items():
        parsed[scoring] = parse_correctness(frame[column], path=path, column=column)

    byte_alias = parse_correctness(
        frame["is_correct_byte"], path=path, column="is_correct_byte"
    )
    if not np.array_equal(parsed["byte"].to_numpy(), byte_alias):
        fail(f"is_correct and is_correct_byte disagree: {path}")
    return parsed.set_index("id", verify_integrity=True).sort_index()


def validate_same_rows(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_path: Path,
    candidate_path: Path,
) -> None:
    """Require all interventions to contain exactly the unmasked evaluation ids."""

    if not reference.index.equals(candidate.index):
        missing = reference.index.difference(candidate.index).tolist()[:5]
        extra = candidate.index.difference(reference.index).tolist()[:5]
        fail(
            f"result-id mismatch between {reference_path} and {candidate_path}; "
            f"missing={missing}, extra={extra}"
        )


def score_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Re-score each frozen cell under three continuation scoring rules."""

    records: list[dict[str, object]] = []
    for _, row in cells.sort_values(["dataset", "model_id", "locale"]).iterrows():
        dataset = str(row["dataset"])
        unmasked_path, selected_path, slr_paths = result_paths(row)
        unmasked = read_behaviour(unmasked_path, dataset)
        selected = read_behaviour(selected_path, dataset)
        slr_frames = [read_behaviour(path, dataset) for path in slr_paths]

        if len(unmasked) != int(row["n_items"]):
            fail(
                f"{row['model_id']} / {row['locale']} / {dataset}: "
                f"found {len(unmasked)} rows, expected {int(row['n_items'])}"
            )
        validate_same_rows(
            unmasked,
            selected,
            reference_path=unmasked_path,
            candidate_path=selected_path,
        )
        for slr_path, slr_frame in zip(slr_paths, slr_frames):
            validate_same_rows(
                unmasked,
                slr_frame,
                reference_path=unmasked_path,
                candidate_path=slr_path,
            )

        for scoring in SCORING_COLUMNS:
            unmasked_acc = float(unmasked[scoring].mean())
            selected_acc = float(selected[scoring].mean())
            slr_accs = np.array(
                [float(frame[scoring].mean()) for frame in slr_frames], dtype=float
            )
            selected_drop = unmasked_acc - selected_acc
            slr_drop = float(np.mean(unmasked_acc - slr_accs))
            records.append(
                {
                    "model_id": str(row["model_id"]),
                    "slug": str(row["slug"]),
                    "locale": str(row["locale"]),
                    "dataset": dataset,
                    "scoring": scoring,
                    "n_items": len(unmasked),
                    "unmasked_acc": unmasked_acc,
                    "selected_acc": selected_acc,
                    "selected_drop": selected_drop,
                    "slr_drop": slr_drop,
                    "slr_gap": selected_drop - slr_drop,
                }
            )

            if scoring == "byte":
                checks = {
                    "unmasked": (unmasked_acc, float(row["unmasked_acc"])),
                    "selected drop": (selected_drop, float(row["cultural_drop"])),
                    "SLR drop": (slr_drop, float(row["slr_drop_mean"])),
                }
                for label, (raw, frozen) in checks.items():
                    if not np.isclose(raw, frozen, atol=5.1e-5, rtol=0.0):
                        fail(
                            f"byte {label} mismatch for {row['model_id']} / "
                            f"{row['locale']} / {dataset}: raw={raw:.8f}, "
                            f"frozen={frozen:.8f}"
                        )

    result = pd.DataFrame.from_records(records)
    expected_rows = EXPECTED_PAIRS * len(EXPECTED_DATASETS) * len(SCORING_COLUMNS)
    if len(result) != expected_rows:
        fail(f"created {len(result)} cell-score rows; expected {expected_rows}")
    return result


def aggregate_scoring(cell_scores: pd.DataFrame, significance: dict) -> pd.DataFrame:
    """Compute equal-pair-weighted scoring summaries and validate byte aggregates."""

    grouped = (
        cell_scores.groupby(["dataset", "scoring"], sort=False)
        .agg(
            n_pairs=("model_id", "size"),
            pair_weighted_unmasked=("unmasked_acc", "mean"),
            pair_weighted_selected_drop=("selected_drop", "mean"),
            pair_weighted_slr_drop=("slr_drop", "mean"),
            pair_weighted_slr_gap=("slr_gap", "mean"),
            min_items=("n_items", "min"),
            max_items=("n_items", "max"),
        )
        .reset_index()
    )
    grouped["dataset_label"] = grouped["dataset"].map(DATASET_LABELS)
    grouped["scoring_label"] = grouped["scoring"].map(SCORING_LABELS)
    grouped["selection_scoring"] = "byte"
    grouped["analysis_kind"] = "behavioural_rescoring_without_reselection"

    for _, row in grouped.loc[grouped["scoring"].eq("byte")].iterrows():
        dataset = str(row["dataset"])
        frozen = significance["aggregate"]["by_dataset"][dataset]["vs_slr"]
        checks = {
            "selected drop": (
                float(row["pair_weighted_selected_drop"]),
                float(frozen["mean_cultural_drop"]),
            ),
            "SLR drop": (
                float(row["pair_weighted_slr_drop"]),
                float(frozen["mean_baseline_drop"]),
            ),
            "SLR gap": (
                float(row["pair_weighted_slr_gap"]),
                float(frozen["mean_gap"]),
            ),
        }
        for label, (raw, expected) in checks.items():
            if not np.isclose(raw, expected, atol=1e-12, rtol=0.0):
                fail(
                    f"byte aggregate {label} mismatch for {dataset}: "
                    f"raw={raw:.12f}, frozen={expected:.12f}"
                )

    dataset_order = {name: index for index, name in enumerate(EXPECTED_DATASETS)}
    scoring_order = {name: index for index, name in enumerate(SCORING_COLUMNS)}
    grouped["_dataset_order"] = grouped["dataset"].map(dataset_order)
    grouped["_scoring_order"] = grouped["scoring"].map(scoring_order)
    grouped = grouped.sort_values(["_dataset_order", "_scoring_order"]).drop(
        columns=["_dataset_order", "_scoring_order"]
    )
    columns = [
        "dataset",
        "dataset_label",
        "scoring",
        "scoring_label",
        "selection_scoring",
        "analysis_kind",
        "n_pairs",
        "min_items",
        "max_items",
        "pair_weighted_unmasked",
        "pair_weighted_selected_drop",
        "pair_weighted_slr_drop",
        "pair_weighted_slr_gap",
    ]
    return grouped[columns].reset_index(drop=True)


def format_decimal(value: float) -> str:
    """Format a signed decimal compactly for a LaTeX table."""

    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def format_overlap(value: float) -> str:
    """Retain enough precision to distinguish the small overlap-null values."""

    text = f"{value:.4f}"
    return "0.0000" if text == "-0.0000" else text


def write_scoring_outputs(values: pd.DataFrame) -> None:
    """Write machine-readable and LaTeX scoring-robustness summaries."""

    SCORING_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(SCORING_VALUES_PATH, index=False, float_format="%.9f")

    lines = [
        "% Auto-generated by diss/analysis/scoring_robustness.py; do not edit.",
        "% Equal weight is given to each of the 36 model--locale pairs.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Dataset & Scoring & Unmasked & Cultural mask drop & SLR drop "
            r"& SLR gap \\"
        ),
        r"\midrule",
    ]
    previous_dataset: str | None = None
    for _, row in values.iterrows():
        dataset = str(row["dataset"])
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\addlinespace")
        lines.append(
            f"{row['dataset_label']} & {row['scoring_label']} & "
            f"{format_decimal(float(row['pair_weighted_unmasked']))} & "
            f"{format_decimal(float(row['pair_weighted_selected_drop']))} & "
            f"{format_decimal(float(row['pair_weighted_slr_drop']))} & "
            f"{format_decimal(float(row['pair_weighted_slr_gap']))} \\\\"
        )
        previous_dataset = dataset
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            (
                r"{\footnotesize \emph{Note.} All rows retain the MLP sets "
                r"selected using attribution normalised by byte length. The "
                r"rows based on token count and summed NLL use saved behaviour "
                r"under those scoring rules. They do not select units again.}"
            ),
            "",
        ]
    )
    SCORING_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORING_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


Neuron = tuple[str, str, int, int]


def neuron_identity(item: dict, *, path: Path) -> Neuron:
    """Return the architecture-safe identity of one saved MLP coordinate."""

    try:
        return (
            str(item["family"]),
            str(item["logical_module_name"]),
            int(item["layer_idx"]),
            int(item["neuron_idx"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid neuron entry in {path}: {exc}")


def load_neuron_set(path: Path) -> tuple[set[Neuron], dict]:
    """Load a selected or random neuron set and reject duplicated coordinates."""

    if not path.is_file():
        fail(f"required neuron set is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("top_neurons"), list):
        fail(f"invalid neuron-set JSON: {path}")
    identities = [neuron_identity(item, path=path) for item in payload["top_neurons"]]
    if len(set(identities)) != len(identities):
        fail(f"duplicated neuron coordinates: {path}")
    if not identities:
        fail(f"empty neuron set: {path}")
    return set(identities), payload


def layer_profile(neurons: Iterable[Neuron]) -> Counter[tuple[str, str, int]]:
    """Count coordinates by architecture family, logical module, and layer."""

    return Counter((family, module, layer) for family, module, layer, _ in neurons)


def jaccard(left: set[Neuron], right: set[Neuron]) -> float:
    """Compute Jaccard overlap for two non-empty coordinate sets."""

    union = left | right
    if not union:
        fail("cannot compute Jaccard for two empty sets")
    return len(left & right) / len(union)


def overlap_model_rows(cells: pd.DataFrame, model_id: str) -> dict[str, pd.Series]:
    """Resolve one frozen row per requested locale for an overlap model."""

    subset = cells.loc[
        cells["model_id"].eq(model_id) & cells["dataset"].eq("blend_mcq")
    ].copy()
    if set(subset["locale"]) != set(OVERLAP_LOCALES) or len(subset) != len(
        OVERLAP_LOCALES
    ):
        fail(
            f"{model_id} does not have exactly the five requested overlap locales: "
            f"{sorted(set(subset['locale']))}"
        )
    return {str(row["locale"]): row for _, row in subset.iterrows()}


def build_overlap_values(cells: pd.DataFrame) -> pd.DataFrame:
    """Compare observed locale overlap with a finite layer-matched random null."""

    records: list[dict[str, object]] = []
    locale_pairs = list(itertools.combinations(OVERLAP_LOCALES, 2))
    seed_assignments = list(itertools.permutations(SLR_SEEDS, len(OVERLAP_LOCALES)))
    if len(locale_pairs) != 10 or len(seed_assignments) != 120:
        fail("unexpected overlap-pair or finite-null enumeration size")

    for model_id, model_label in OVERLAP_MODELS.items():
        rows = overlap_model_rows(cells, model_id)
        selected: dict[str, set[Neuron]] = {}
        random_sets: dict[str, dict[int, set[Neuron]]] = {}

        for locale in OVERLAP_LOCALES:
            row = rows[locale]
            base = cell_base(row) / "cultural_neuron"
            country_tag = str(row["country"]).replace(" ", "")
            selected_path = base / f"{country_tag}_neurons_blend_mcq.json"
            selected_set, selected_payload = load_neuron_set(selected_path)
            if selected_payload.get("target_country") not in (
                str(row["country"]),
                country_tag,
            ):
                fail(f"selected target-country mismatch: {selected_path}")
            if "total_selected" in selected_payload and int(
                selected_payload["total_selected"]
            ) != len(selected_set):
                fail(f"selected-set count mismatch: {selected_path}")
            selected[locale] = selected_set
            selected_profile = layer_profile(selected_set)

            random_sets[locale] = {}
            for seed in SLR_SEEDS:
                random_path = base / f"{country_tag}_all_neurons_random_layer{seed}.json"
                random_set, random_payload = load_neuron_set(random_path)
                thresholds = random_payload.get("thresholds", {})
                if not isinstance(thresholds, dict) or int(
                    thresholds.get("random_seed", -1)
                ) != seed:
                    fail(f"random-seed metadata mismatch: {random_path}")
                if len(random_set) != len(selected_set):
                    fail(f"random-set size mismatch: {random_path}")
                if layer_profile(random_set) != selected_profile:
                    fail(f"random set is not layer/module matched: {random_path}")
                random_sets[locale][seed] = random_set

        observed_pairwise = [
            jaccard(selected[left], selected[right]) for left, right in locale_pairs
        ]
        observed_mean = float(np.mean(observed_pairwise))

        null_means: list[float] = []
        null_pairwise: list[float] = []
        for assignment in seed_assignments:
            locale_seed = dict(zip(OVERLAP_LOCALES, assignment))
            if len(set(locale_seed.values())) != len(OVERLAP_LOCALES):
                fail("finite-null assignment contains repeated seeds")
            replicate = [
                jaccard(
                    random_sets[left][locale_seed[left]],
                    random_sets[right][locale_seed[right]],
                )
                for left, right in locale_pairs
            ]
            null_pairwise.extend(replicate)
            null_means.append(float(np.mean(replicate)))

        null = np.asarray(null_means, dtype=float)
        pair_null = np.asarray(null_pairwise, dtype=float)
        q025, q975 = np.quantile(null, [0.025, 0.975])
        pair_q025, pair_q975 = np.quantile(pair_null, [0.025, 0.975])
        records.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "n_locales": len(OVERLAP_LOCALES),
                "n_observed_locale_pairs": len(locale_pairs),
                "observed_mean_jaccard": observed_mean,
                "observed_min_jaccard": float(np.min(observed_pairwise)),
                "observed_max_jaccard": float(np.max(observed_pairwise)),
                "n_saved_seeds_per_locale": len(SLR_SEEDS),
                "n_distinct_seed_assignments": len(seed_assignments),
                "n_null_pairwise_comparisons": len(null_pairwise),
                "null_mean_jaccard": float(np.mean(null)),
                "null_mean_q025": float(q025),
                "null_mean_q975": float(q975),
                "null_pairwise_q025": float(pair_q025),
                "null_pairwise_q975": float(pair_q975),
                "selected_size_min": min(len(value) for value in selected.values()),
                "selected_size_max": max(len(value) for value in selected.values()),
                "null_kind": "finite_saved_layer_matched_distinct_seed_assignments",
                "formal_population_test": False,
            }
        )

    return pd.DataFrame.from_records(records)


def write_overlap_outputs(values: pd.DataFrame) -> None:
    """Write machine-readable and LaTeX finite-null overlap summaries."""

    OVERLAP_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(OVERLAP_VALUES_PATH, index=False, float_format="%.9f")

    lines = [
        "% Auto-generated by diss/analysis/scoring_robustness.py; do not edit.",
        "% Null interval is over 120 model-level mean-Jaccard assignments.",
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Model & Observed mean & SLR mean & SLR 95\% range \\",
        r"\midrule",
    ]
    for _, row in values.iterrows():
        interval = (
            f"$[{format_overlap(float(row['null_mean_q025']))},\\,"
            f"{format_overlap(float(row['null_mean_q975']))}]$"
        )
        lines.append(
            f"{row['model_label']} & "
            f"{format_overlap(float(row['observed_mean_jaccard']))} & "
            f"{format_overlap(float(row['null_mean_jaccard']))} & "
            f"{interval} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            (
                r"{\footnotesize \emph{Note.} The null enumerates all 120 "
                r"assignments of the five saved random seeds matched by layer to "
                r"five locales, without reusing a seed within an assignment. "
                r"The 95\% range is the 2.5th--97.5th percentile of the "
                r"resulting mean pairwise Jaccards. The comparison is reported "
                r"descriptively because only five saved masks are available for each locale.}"
            ),
            "",
        ]
    )
    OVERLAP_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERLAP_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cells, significance = load_cohort()
    cell_scores = score_cells(cells)
    scoring_values = aggregate_scoring(cell_scores, significance)
    write_scoring_outputs(scoring_values)

    overlap_values = build_overlap_values(cells)
    write_overlap_outputs(overlap_values)

    print(
        f"wrote {SCORING_VALUES_PATH} "
        f"({len(scoring_values)} dataset--scoring rows; 36 pairs each)"
    )
    print(f"wrote {SCORING_TABLE_PATH}")
    print(
        f"wrote {OVERLAP_VALUES_PATH} "
        f"({len(overlap_values)} models; 120 finite-null assignments each)"
    )
    print(f"wrote {OVERLAP_TABLE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
