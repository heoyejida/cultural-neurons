#!/usr/bin/env python3
"""Build the HPLT checkpoint case-study figure and appendix table.

The script reads only the three frozen checkpoint summaries.  It neither
imports :mod:`diss` nor runs model inference.  The selected set is the set
chosen at the final checkpoint and traced backwards, so the plot describes a
bounded case study rather than estimating a universal emergence time.

Outputs
-------
* ``figures/fig_checkpoint_accuracy.{pdf,png}``
* ``figures/fig_checkpoint_rank.{pdf,png}``
* ``figures/fig_checkpoint_slr_gap.{pdf,png}``
* ``figures/fig_checkpoint_legend.{pdf,png}``
* ``tables/tab_checkpoint_trajectories.tex``
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile


_CACHE_ROOT = Path(tempfile.gettempdir()) / "cultural-neurons-checkpoint-effects-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_style import (
    LOCALE_COLOURS,
    apply_plot_style,
    save_legend,
    save_pdf_png,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "diss" / "outputs" / "analysis"
OUTPUT_ROOT = PROJECT_ROOT / "diss" / "outputs"
EBLEND_PATH = PROJECT_ROOT / "diss" / "data" / "eBLEnD" / "eblend_final.csv"

FIGURE_ROOT = OUTPUT_ROOT / "figures"
TABLE_PATH = OUTPUT_ROOT / "tables" / "tab_checkpoint_trajectories.tex"

EXPECTED_LABELS = [
    "checkpoint_0001000",
    "checkpoint_0002000",
    "checkpoint_0004000",
    "checkpoint_0007000",
    "checkpoint_0016000",
    "checkpoint_0032000",
    "checkpoint_0047000",
]
CHECKPOINT_TICKS = ["1k", "2k", "4k", "7k", "16k", "32k", "47k"]

SOURCE_RESAMPLE_DRAWS = 10_000
SOURCE_RESAMPLE_BASE_SEED = 20_760_529
SLR_MASK_SEEDS = tuple(range(5))
NON_NEURON_TOPICS = {
    "Education",
    "Family",
    "Holidays/Celebration/Leisure",
}
EXPECTED_EVALUATION_ROWS = {"kor": 146, "ara": 70, "fra": 58}
EXPECTED_SOURCE_QUESTIONS = {"kor": 41, "ara": 36, "fra": 31}

METRICS = [
    "blend_acc_non_neuron",
    "anchor_rank_percentile_mean",
    "anchor_drop_over_slr_blend",
]
PANEL_SPECS = (
    ("blend_acc_non_neuron", "fig_checkpoint_accuracy"),
    ("anchor_rank_percentile_mean", "fig_checkpoint_rank"),
    ("anchor_drop_over_slr_blend", "fig_checkpoint_slr_gap"),
)
_PANEL_AXES = {
    "blend_acc_non_neuron": (
        "Accuracy",
        (0.24, 0.74),
        (0.3, 0.5, 0.7),
    ),
    "anchor_rank_percentile_mean": (
        "Rank percentile",
        (50.0, 102.0),
        (60.0, 80.0, 100.0),
    ),
    "anchor_drop_over_slr_blend": (
        "Cultural mask drop minus SLR drop",
        (-0.13, 0.33),
        (-0.1, 0.0, 0.1, 0.2, 0.3),
    ),
}
FIGURE_ASSET_PATHS = tuple(
    FIGURE_ROOT / f"{stem}.{suffix}"
    for _metric, stem in PANEL_SPECS
    for suffix in ("pdf", "png")
) + tuple(
    FIGURE_ROOT / f"fig_checkpoint_legend.{suffix}"
    for suffix in ("pdf", "png")
)
REQUIRED_COLUMNS = {
    "checkpoint_label",
    "locale",
    *METRICS,
    "anchor_blend_drop",
    "same_layer_random_blend_drop_mean",
    "same_layer_random_blend_drop_std",
}

INTERVAL_COLUMNS = {
    "gap_ci_lo",
    "gap_ci_hi",
    "source_question_count",
    "evaluation_row_count",
    "source_resample_seed",
}

LOCALES = {
    "kor": {
        "name": "Korean",
        "locale": "ko",
        "country_tag": "SouthKorea",
        "colour": LOCALE_COLOURS["ko"],
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.6,
        "zorder": 4,
    },
    "ara": {
        "name": "Arabic",
        "locale": "ar",
        "country_tag": "SaudiArabia",
        "colour": LOCALE_COLOURS["ar"],
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.6,
        "zorder": 2,
    },
    "fra": {
        "name": "French",
        "locale": "fr",
        "country_tag": "France",
        "colour": LOCALE_COLOURS["fr"],
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.6,
        "zorder": 3,
    },
}
PLOT_ORDER = ("kor", "fra", "ara")


def fail(message: str) -> None:
    """Raise a concise input-validation error."""

    raise RuntimeError(message)


def load_trajectories() -> dict[str, pd.DataFrame]:
    """Load and validate the three aligned seven-checkpoint trajectories."""

    trajectories: dict[str, pd.DataFrame] = {}
    for code, style in LOCALES.items():
        path = OUTPUT_ROOT / f"cp_summary_hplt2c-{code}-trace.csv"
        if not path.is_file():
            fail(f"required input is missing: {path}")

        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            fail(f"{path} is missing columns: {', '.join(missing)}")
        if len(frame) != len(EXPECTED_LABELS):
            fail(f"{path} has {len(frame)} rows; expected 7")
        if frame["checkpoint_label"].duplicated().any():
            fail(f"{path} contains duplicate checkpoint labels")

        frame = frame.set_index("checkpoint_label").reindex(EXPECTED_LABELS)
        if frame.index.tolist() != EXPECTED_LABELS or frame.isna().all(axis=1).any():
            fail(f"{path} does not contain the exact required checkpoints")
        observed_locale = set(frame["locale"].astype(str))
        if observed_locale != {style["locale"]}:
            fail(
                f"{path} has locale values {sorted(observed_locale)}; "
                f"expected {style['locale']!r}"
            )

        numeric_columns = METRICS + [
            "anchor_blend_drop",
            "same_layer_random_blend_drop_mean",
            "same_layer_random_blend_drop_std",
        ]
        frame[numeric_columns] = frame[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        values = frame[numeric_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            fail(f"{path} contains non-finite values in plotted/table columns")
        if (frame["same_layer_random_blend_drop_std"] < 0).any():
            fail(f"{path} contains a negative random-mask standard deviation")
        if not frame["blend_acc_non_neuron"].between(0, 1).all():
            fail(f"{path} contains accuracy outside [0, 1]")
        if not frame["anchor_rank_percentile_mean"].between(0, 100).all():
            fail(f"{path} contains rank percentile outside [0, 100]")

        trajectories[code] = frame.reset_index()

    checkpoint_sets = {
        tuple(frame["checkpoint_label"]) for frame in trajectories.values()
    }
    if checkpoint_sets != {tuple(EXPECTED_LABELS)}:
        fail("the three trajectories do not share the required checkpoints")
    return trajectories


def load_source_mapping() -> pd.DataFrame:
    """Load the frozen mapping from localised row ids to source questions."""

    if not EBLEND_PATH.is_file():
        fail(f"required source mapping is missing: {EBLEND_PATH}")
    mapping = pd.read_csv(
        EBLEND_PATH,
        usecols=["id", "org_id"],
        dtype={"id": str, "org_id": str},
    )
    if mapping[["id", "org_id"]].isna().any().any():
        fail(f"{EBLEND_PATH} contains missing id or org_id values")
    if mapping["id"].duplicated().any():
        fail(f"{EBLEND_PATH} contains duplicate localised row ids")
    return mapping


def correctness(values: pd.Series, *, path: Path) -> pd.Series:
    """Convert a saved correctness column to zero and one values."""

    normalised = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalised) - {"true", "false"})
    if unexpected:
        fail(f"{path} contains invalid correctness values: {unexpected}")
    return normalised.map({"true": 1.0, "false": 0.0}).astype(float)


def checkpoint_result_paths(
    code: str,
    checkpoint_label: str,
) -> tuple[Path, Path, list[Path]]:
    """Return the seven frozen result paths for one trace cell."""

    style = LOCALES[code]
    locale = style["locale"]
    country_tag = style["country_tag"]
    base = (
        OUTPUT_ROOT
        / f"hplt2c_{code}_checkpoints"
        / f"cp_{checkpoint_label}"
    )
    result_dir = f"blend_mcq_{locale}_non_neuron"
    unmasked = base / f"blend_mcq_{locale}" / "results.csv"
    selected = (
        base
        / "masked"
        / f"{country_tag}_neurons_blend_mcq"
        / result_dir
        / "results.csv"
    )
    controls = [
        base
        / "masked"
        / f"{country_tag}_all_neurons_random_layer{mask_seed}"
        / result_dir
        / "results.csv"
        for mask_seed in SLR_MASK_SEEDS
    ]
    return unmasked, selected, controls


def checkpoint_gap_source_clusters(
    code: str,
    checkpoint_label: str,
    expected_row: pd.Series,
    source_mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load one trace cell and return its paired source question blocks."""

    unmasked_path, selected_path, control_paths = checkpoint_result_paths(
        code, checkpoint_label
    )
    for path in (unmasked_path, selected_path, *control_paths):
        if not path.is_file():
            fail(f"required checkpoint result is missing: {path}")

    unmasked = pd.read_csv(
        unmasked_path,
        usecols=["id", "topic", "is_correct"],
        dtype={"id": str, "topic": str},
    )
    unmasked = unmasked.loc[
        unmasked["topic"].isin(NON_NEURON_TOPICS)
    ].copy()
    if unmasked["id"].duplicated().any():
        fail(f"{unmasked_path} contains duplicate evaluation ids")
    unmasked["unmasked_correct"] = correctness(
        unmasked["is_correct"], path=unmasked_path
    )

    selected = pd.read_csv(
        selected_path,
        usecols=["id", "is_correct"],
        dtype={"id": str},
    )
    if selected["id"].duplicated().any():
        fail(f"{selected_path} contains duplicate evaluation ids")
    selected["selected_correct"] = correctness(
        selected["is_correct"], path=selected_path
    )

    expected_ids = set(selected["id"])
    if set(unmasked["id"]) != expected_ids:
        fail(
            f"unmasked and selected ids differ for {code} at {checkpoint_label}"
        )
    paired = unmasked[["id", "unmasked_correct"]].merge(
        selected[["id", "selected_correct"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    control_columns: list[str] = []
    for mask_seed, path in zip(SLR_MASK_SEEDS, control_paths):
        control = pd.read_csv(
            path,
            usecols=["id", "is_correct"],
            dtype={"id": str},
        )
        if control["id"].duplicated().any():
            fail(f"{path} contains duplicate evaluation ids")
        if set(control["id"]) != expected_ids:
            fail(
                f"control ids differ for {code} at {checkpoint_label} "
                f"and mask {mask_seed}"
            )
        column = f"control_correct_{mask_seed}"
        control[column] = correctness(control["is_correct"], path=path)
        paired = paired.merge(
            control[["id", column]],
            on="id",
            how="inner",
            validate="one_to_one",
        )
        control_columns.append(column)

    expected_rows = EXPECTED_EVALUATION_ROWS[code]
    if len(paired) != expected_rows:
        fail(
            f"{code} at {checkpoint_label} has {len(paired)} evaluation rows; "
            f"expected {expected_rows}"
        )

    selected_delta = paired["unmasked_correct"] - paired["selected_correct"]
    control_delta = paired["unmasked_correct"].to_numpy(dtype=float)[:, None] - paired[
        control_columns
    ].to_numpy(dtype=float)
    paired["gap"] = selected_delta.to_numpy(dtype=float) - control_delta.mean(axis=1)

    raw_accuracy = float(paired["unmasked_correct"].mean())
    raw_selected_drop = float(selected_delta.mean())
    raw_control_drop = float(control_delta.mean(axis=0).mean())
    raw_gap = float(paired["gap"].mean())
    frozen_values = (
        ("unmasked accuracy", raw_accuracy, expected_row["blend_acc_non_neuron"]),
        ("selected drop", raw_selected_drop, expected_row["anchor_blend_drop"]),
        (
            "same layer random drop",
            raw_control_drop,
            expected_row["same_layer_random_blend_drop_mean"],
        ),
        ("SLR gap", raw_gap, expected_row["anchor_drop_over_slr_blend"]),
    )
    for label, observed, expected in frozen_values:
        if not np.isclose(observed, float(expected), atol=5.1e-5, rtol=0.0):
            fail(
                f"raw {label} differs from the frozen summary for {code} at "
                f"{checkpoint_label}. Raw value {observed:.8f}, frozen value "
                f"{float(expected):.8f}"
            )

    paired["source_id"] = paired["id"].str.replace(
        r"^eblnd-", "", regex=True
    )
    paired = paired.merge(
        source_mapping,
        left_on="source_id",
        right_on="id",
        how="left",
        validate="many_to_one",
        suffixes=("_result", "_source"),
    )
    if paired["org_id"].isna().any():
        missing = paired.loc[paired["org_id"].isna(), "source_id"].head()
        fail(f"checkpoint ids missing from eBLEnD mapping: {missing.tolist()}")

    clusters = paired.groupby("org_id", sort=True)["gap"].agg(["sum", "count"])
    expected_sources = EXPECTED_SOURCE_QUESTIONS[code]
    if len(clusters) != expected_sources:
        fail(
            f"{code} at {checkpoint_label} has {len(clusters)} source questions; "
            f"expected {expected_sources}"
        )
    return (
        clusters["sum"].to_numpy(dtype=float),
        clusters["count"].to_numpy(dtype=float),
        raw_gap,
    )


def source_question_interval(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    """Resample whole source questions in bounded memory."""

    if len(sums) == 0 or len(sums) != len(counts):
        fail("invalid source question blocks")
    rng = np.random.default_rng(seed)
    values = np.empty(SOURCE_RESAMPLE_DRAWS, dtype=float)
    batch_size = 512
    for start in range(0, SOURCE_RESAMPLE_DRAWS, batch_size):
        stop = min(start + batch_size, SOURCE_RESAMPLE_DRAWS)
        draw = rng.integers(0, len(sums), size=(stop - start, len(sums)))
        values[start:stop] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def add_gap_intervals(
    trajectories: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Attach paired source question intervals to all 21 trace cells."""

    source_mapping = load_source_mapping()
    completed: dict[str, pd.DataFrame] = {}
    for locale_index, (code, frame) in enumerate(trajectories.items()):
        frame = frame.copy()
        lows: list[float] = []
        highs: list[float] = []
        source_counts: list[int] = []
        row_counts: list[int] = []
        seeds: list[int] = []
        for checkpoint_index, (_, row) in enumerate(frame.iterrows()):
            label = str(row["checkpoint_label"])
            sums, counts, raw_gap = checkpoint_gap_source_clusters(
                code, label, row, source_mapping
            )
            seed = (
                SOURCE_RESAMPLE_BASE_SEED
                + locale_index * len(EXPECTED_LABELS)
                + checkpoint_index
            )
            low, high = source_question_interval(sums, counts, seed=seed)
            frozen_gap = float(row["anchor_drop_over_slr_blend"])
            if not low <= raw_gap <= high:
                fail(
                    f"source question interval excludes its raw estimate for "
                    f"{code} at {label}"
                )
            if not low - 5.1e-5 <= frozen_gap <= high + 5.1e-5:
                fail(
                    f"source question interval excludes its frozen estimate for "
                    f"{code} at {label}"
                )
            lows.append(low)
            highs.append(high)
            source_counts.append(len(sums))
            row_counts.append(int(counts.sum()))
            seeds.append(seed)
        frame["gap_ci_lo"] = lows
        frame["gap_ci_hi"] = highs
        frame["source_question_count"] = source_counts
        frame["evaluation_row_count"] = row_counts
        frame["source_resample_seed"] = seeds
        completed[code] = frame
    return completed


def build_checkpoint_panel(
    trajectories: dict[str, pd.DataFrame],
    metric: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Return one title-free checkpoint panel and its axis for inspection."""

    if metric not in _PANEL_AXES:
        raise ValueError(f"unsupported checkpoint metric: {metric}")

    required_columns = {"checkpoint_label", metric}
    if metric == "anchor_drop_over_slr_blend":
        required_columns.update({"gap_ci_lo", "gap_ci_hi"})
    for code in PLOT_ORDER:
        if code not in trajectories:
            fail(f"checkpoint trajectories are missing locale: {code}")
        frame = trajectories[code]
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            fail(f"{code} trajectory is missing panel columns: {missing}")
        if len(frame) != len(EXPECTED_LABELS):
            fail(f"{code} trajectory has {len(frame)} rows; expected 7")
        if frame["checkpoint_label"].astype(str).tolist() != EXPECTED_LABELS:
            fail(f"{code} trajectory does not contain the required checkpoint order")

    apply_plot_style()
    figure, axis = plt.subplots(figsize=(2.3, 2.18))
    x = np.arange(len(EXPECTED_LABELS), dtype=float)

    if metric == "anchor_drop_over_slr_blend":
        axis.axhline(0.0, color="#4A4A4A", linewidth=0.65, zorder=1)

    for code in PLOT_ORDER:
        style = LOCALES[code]
        frame = trajectories[code]
        values = frame[metric].to_numpy(dtype=float)
        plot_arguments = {
            "color": style["colour"],
            "linestyle": "-",
            "marker": "o",
            "linewidth": style["linewidth"],
            "markersize": 4.0,
            "markerfacecolor": style["colour"],
            "markeredgecolor": "white",
            "markeredgewidth": 0.8,
            "zorder": style["zorder"],
        }
        if metric == "anchor_drop_over_slr_blend":
            low = frame["gap_ci_lo"].to_numpy(dtype=float)
            high = frame["gap_ci_hi"].to_numpy(dtype=float)
            error = np.vstack(
                [np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)]
            )
            axis.errorbar(
                x,
                values,
                yerr=error,
                ecolor=style["colour"],
                elinewidth=0.55,
                capsize=1.6,
                capthick=0.55,
                **plot_arguments,
            )
        else:
            axis.plot(x, values, **plot_arguments)

    ylabel, ylim, yticks = _PANEL_AXES[metric]
    axis.set_ylabel(ylabel, fontweight="normal")
    axis.set_ylim(*ylim)
    axis.set_yticks(yticks)
    axis.set_xticks(x, CHECKPOINT_TICKS)
    axis.set_xlabel("Checkpoint", fontweight="normal")
    axis.xaxis.grid(False)
    axis.yaxis.grid(True)
    figure.tight_layout(pad=0.35)
    return figure, axis


def write_checkpoint_assets(
    trajectories: dict[str, pd.DataFrame],
    output_directory: Path = FIGURE_ROOT,
) -> None:
    """Write three checkpoint panels and their standalone shared legend."""

    for code, frame in trajectories.items():
        missing = sorted(INTERVAL_COLUMNS - set(frame.columns))
        if missing:
            fail(f"{code} trajectory is missing interval columns: {missing}")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for metric, output_name in PANEL_SPECS:
        figure, _axis = build_checkpoint_panel(trajectories, metric)
        save_pdf_png(figure, output_directory / output_name, dpi=300)
        plt.close(figure)

    handles = [
        Line2D(
            [0],
            [0],
            color=LOCALES[code]["colour"],
            linestyle="-",
            marker="o",
            linewidth=LOCALES[code]["linewidth"],
            markersize=4.0,
            markerfacecolor=LOCALES[code]["colour"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        for code in PLOT_ORDER
    ]
    labels = [LOCALES[code]["name"] for code in PLOT_ORDER]
    save_legend(
        handles,
        labels,
        output_directory / "fig_checkpoint_legend",
        columns=len(PLOT_ORDER),
    )


def checkpoint_step(label: str) -> str:
    """Convert a validated checkpoint label to the compact table form."""

    match = re.fullmatch(r"checkpoint_0*(\d+)", label)
    if match is None:
        fail(f"cannot parse checkpoint label: {label}")
    step = int(match.group(1))
    return f"{step // 1000}k"


def format_fixed(value: float, decimals: int) -> str:
    """Format a finite value while suppressing typographic negative zero."""

    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def build_table(trajectories: dict[str, pd.DataFrame]) -> None:
    """Write all three aligned trajectories as one compact full-width tabular."""

    for code, frame in trajectories.items():
        missing = sorted(INTERVAL_COLUMNS - set(frame.columns))
        if missing:
            fail(f"{code} trajectory is missing interval columns: {missing}")

    lines = [
        "% Generated by diss/analysis/checkpoint_effects.py. Do not edit.",
        "% Accuracy is on unseen Extended BLEnD topics. Rank traces the final selected set.",
        "% Gap intervals use 10,000 paired source question resamples with base seed 20760529.",
        "% They retain all row variants and condition on the five saved same layer random masks.",
        "% Source question counts are 41 for Korean, 36 for Arabic, and 31 for French.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrlrrrlrrrl@{}}",
        r"\toprule",
        (
            r"Checkpoint & \multicolumn{4}{c}{Korean} "
            r"& \multicolumn{4}{c}{Arabic} & \multicolumn{4}{c}{French} \\"
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        (
            r"& Acc. & Rank & Gap & 95\% interval "
            r"& Acc. & Rank & Gap & 95\% interval "
            r"& Acc. & Rank & Gap & 95\% interval \\"
        ),
        r"\midrule",
    ]

    by_code = {
        code: frame.set_index("checkpoint_label")
        for code, frame in trajectories.items()
    }
    for label in EXPECTED_LABELS:
        cells = [checkpoint_step(label)]
        for code in LOCALES:
            row = by_code[code].loc[label]
            cells.extend(
                [
                    format_fixed(row["blend_acc_non_neuron"], 3),
                    format_fixed(row["anchor_rank_percentile_mean"], 1),
                    format_fixed(row["anchor_drop_over_slr_blend"], 3),
                    (
                        "["
                        + format_fixed(row["gap_ci_lo"], 3)
                        + ", "
                        + format_fixed(row["gap_ci_hi"], 3)
                        + "]"
                    ),
                ]
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular*}", ""])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def validate_outputs() -> None:
    """Check that all requested assets exist and are non-empty."""

    for path in (*FIGURE_ASSET_PATHS, TABLE_PATH):
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"output was not created: {path}")
    table_text = TABLE_PATH.read_text(encoding="utf-8")
    row_count = len(re.findall(r"^\d+k &", table_text, flags=re.MULTILINE))
    if row_count != 7:
        fail(f"table has {row_count} checkpoint rows; expected 7")
    interval_count = len(
        re.findall(r"\[-?\d+\.\d{3},\s*-?\d+\.\d{3}\]", table_text)
    )
    if interval_count != 21:
        fail(f"table has {interval_count} checkpoint intervals; expected 21")
    if "bootstrap" in table_text.lower():
        fail("table must describe source question resampling directly")


def main() -> None:
    """Build and validate the frozen checkpoint assets."""

    trajectories = add_gap_intervals(load_trajectories())
    write_checkpoint_assets(trajectories, FIGURE_ROOT)
    build_table(trajectories)
    validate_outputs()
    print(
        "Built checkpoint assets with source question intervals from "
        "3 locales x 7 checkpoints:\n"
        + "\n".join(f"  {path}" for path in FIGURE_ASSET_PATHS)
        + "\n"
        f"  {TABLE_PATH}"
    )


if __name__ == "__main__":
    main()
