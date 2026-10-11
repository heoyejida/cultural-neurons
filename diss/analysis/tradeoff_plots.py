"""Plot cultural vs reasoning accuracy trade-off from behavior_summary.csv.

Emits five figures under diss/outputs/figures/:
  1. tradeoff_small.png        — cultural vs reasoning, <1B, all locales
  2. tradeoff_large.png        — cultural vs reasoning, ≥1B, ko / ar / fr / ja
                                 in the tradeoff_small format (locale-grouped
                                 bars coloured by regime) with the explicit
                                 per-locale model set in LARGE_SPEC
  3. scaling_curves.png        — XGLM family scaling (ko, ja, fr)
  4. regime_deltas_small.png   — mono − {bi, multi} delta, <1B, all locales
  5. regime_deltas_large.png   — mono − {bi, multi} delta, ≥1B, ko / ja / fr

The Qwen3 family is excluded from the multilingual regime by default; pass
``--include-qwen`` to put it back. Both cultural_accuracy and reasoning_accuracy
are the row-instance byte-norm accuracies from behavior_summary.csv.

Run from anywhere:
    python -m diss.analysis.tradeoff_plots
    python diss/analysis/tradeoff_plots.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, NullFormatter

from diss.core.registry import LOCALE_DISPLAY_ORDER, MODEL_PARAMS, fmt_params
from diss.analysis.figstyle import REGIME_COLOR, COLOR_GRID, apply_house_style

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "diss" / "outputs"
SUMMARY_CSV = OUTPUTS_DIR / "behavior_summary.csv"
FIG_DIR = OUTPUTS_DIR / "figures"

CANONICAL_LOCALES = LOCALE_DISPLAY_ORDER
CHANCE = 0.25
SMALL_PARAM_THRESHOLD = 1e9

# BCP-47-style locale tags used as x-axis / per-group labels in the trade-off
# figures (ko -> ko-KR, etc.).
LOCALE_TAG = {
    "ko": "ko-KR",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "fr": "fr-FR",
    "es": "es-ES",
    "ar": "ar-SA",
    "en": "en-US",
}

# Regime colours come from the shared palette (figstyle.REGIME_COLOR) so the
# trade-off figures match the cultural-neuron / masking / layer figures.
# C_BI_SMALL doubles as the "Cultural" metric accent; C_REASONING (teal) is the
# decoupled "Reasoning" metric accent.
C_MONO = REGIME_COLOR["monolingual"]  # muted blue — monolingual
C_BI_SMALL = REGIME_COLOR["bilingual"]  # muted orange — bilingual / cultural metric
C_XGLM = REGIME_COLOR["multilingual"]  # muted green — XGLM / multilingual
C_REGIME_MULTI = REGIME_COLOR["multilingual"]  # multilingual — "mono − multi" delta
C_BI_LARGE = "#c44e52"  # muted red — large bilingual / star marker accent
C_REASONING = "#17BEBB"  # teal — reasoning metric accent
C_GRID = COLOR_GRID

XGLM_FAMILY = ["xglm-564m", "xglm-1p7b", "xglm-2p9b", "xglm-7p5b"]

# Locales shown in scaling-curves and specialized-vs-general figures.
# ar is intentionally excluded — no compatible large mono/bi models available.
TARGET_LOCALES = ["ko", "ja", "fr"]

# Order within each panel: monolingual, bilingual, multilingual.
# Per-locale specialised picks are constrained to models that actually appear
# in behaviour_summary.csv — see diss/core/registry.MODEL_PARAMS for the
# canonical evaluated set.
SPECIALIZED_MATCHES = {
    "ko": {
        "bilingual": "kanana-1p5-2p1b-base",
        "monolingual": "polyglot-ko-1p3b",
        "multilingual": ["tiny-aya-base"],
    },
    "ja": {
        "bilingual": "rinna-bilingual-neox-4b",
        "monolingual": "rinna-monolingual-neox-3p6b",
        "multilingual": ["tiny-aya-base"],
    },
    "fr": {
        "bilingual": "croissantllm-base",
        "monolingual": "gpt-fr-cased-base",
        "multilingual": ["tiny-aya-base"],
    },
}

# Explicit ≥1B model set for tradeoff_large, per locale, as
# (behaviour_summary label, display name, regime). The locale-specialised
# monolingual / bilingual picks plus the shared multilingual references
# (qwen / eurollm / aya; ja uses Qwen3-4B). Bars are coloured by regime and
# annotated with the display name. qwen3 is excluded by load_summary by
# default, so tradeoff_large is fed a qwen-inclusive frame in main().
LARGE_LOCALE_ORDER = ["ko", "ja", "fr", "ar"]
LARGE_SPEC = {
    "ko": [
        ("hplt2c-kor", "HPLT2c-kor 2.1B", "monolingual"),
        ("kanana-1p5-2p1b-base", "Kanana1.5 2.1B", "bilingual"),
        ("qwen3-1p7b-base", "Qwen3 1.7B", "multilingual"),
        ("eurollm-1p7b", "EuroLLM 1.7B", "multilingual"),
        ("tiny-aya-base", "Tiny aya 3.35B", "multilingual"),
    ],
    "ar": [
        ("hplt2c-ara", "HPLT2c-ara 2.1B", "monolingual"),
        ("jais-family-1p3b", "Jais 1.3B", "bilingual"),
        ("qwen3-1p7b-base", "Qwen3 1.7B", "multilingual"),
        ("eurollm-1p7b", "EuroLLM 1.7B", "multilingual"),
        ("tiny-aya-base", "Tiny aya 3.35B", "multilingual"),
    ],
    "fr": [
        # fr monolingual is HPLT2c-fra (the spec said hplt2c_ara, a copy-paste
        # slip: the Arabic model has no French evaluation row).
        ("hplt2c-fra", "HPLT2c-fra 2.1B", "monolingual"),
        ("croissantllm-base", "CroissantLLM 1.3B", "bilingual"),
        ("qwen3-1p7b-base", "Qwen3 1.7B", "multilingual"),
        ("eurollm-1p7b", "EuroLLM 1.7B", "multilingual"),
        ("tiny-aya-base", "Tiny aya 3.35B", "multilingual"),
    ],
    "ja": [
        ("rinna-monolingual-neox-3p6b", "Rinna monolingual 3.6B", "monolingual"),
        ("rinna-bilingual-neox-4b", "Rinna bilingual 3.6B", "bilingual"),
        ("qwen3-4b-base", "Qwen3 4B", "multilingual"),
        ("eurollm-1p7b", "EuroLLM 1.7B", "multilingual"),
        ("tiny-aya-base", "Tiny aya 3.35B", "multilingual"),
    ],
}

# Scaling-curve x-axis: data-point-aligned ticks rather than powers of 10.
SCALING_XTICKS = [560e6, 1.7e9, 3e9, 7e9]


def load_summary(
    summary_csv: Path = SUMMARY_CSV,
    *,
    exclude_qwen: bool = True,
):
    df = pd.read_csv(summary_csv)
    if exclude_qwen:
        df = df[~df["label"].str.startswith("qwen3-")]
    df = df.reset_index(drop=True)
    df["params"] = df["label"].map(MODEL_PARAMS)
    missing = df[df["params"].isna()]["label"].unique()
    if len(missing) > 0:
        raise ValueError(f"Missing MODEL_PARAMS entries for: {missing.tolist()}")
    return df


def _multilingual_matches(locale: str) -> list[str]:
    """Per-locale multilingual reference labels from SPECIALIZED_MATCHES.
    Used by tradeoff_large and regime_deltas_large."""
    return list(SPECIALIZED_MATCHES[locale]["multilingual"])


def _params_tick_fmt(x, _pos):
    return fmt_params(x)


def _style_axes(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color=C_GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# -------------------------- Figure 1: small regimes --------------------------


def plot_small_regimes(df, fig_dir):
    small = df[df["params"] < SMALL_PARAM_THRESHOLD].copy()

    present = set(small["locale"].unique())
    locale_order = [loc for loc in CANONICAL_LOCALES if loc in present]
    locale_order += [
        loc for loc in small["locale"].unique() if loc not in CANONICAL_LOCALES
    ]
    locale_labels = {
        row["locale"]: LOCALE_TAG.get(row["locale"], row["locale"])
        for _, row in small.drop_duplicates("locale").iterrows()
    }

    def _key(row):
        if row["regime"] in ("monolingual", "bilingual"):
            return row["regime"]
        return f"multilingual ({row['label']})"

    small["model_key"] = small.apply(_key, axis=1)
    regime_rank = {"monolingual": 0, "bilingual": 1, "multilingual": 2}
    model_order = (
        small[["model_key", "regime"]]
        .drop_duplicates()
        .assign(rank=lambda d: d["regime"].map(regime_rank))
        .sort_values(["rank", "model_key"])["model_key"]
        .tolist()
    )

    # Semantic colors consistent with the other two figures.
    semantic_colors = {
        "monolingual": C_MONO,
        "bilingual": C_BI_SMALL,
        "multilingual (xglm-564m)": C_XGLM,
    }
    color_map = {
        mk: semantic_colors.get(mk, plt.cm.tab10.colors[i % 10])
        for i, mk in enumerate(model_order)
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    acc_types = [
        ("cultural_accuracy", "BLEnD Accuracy", axes[0]),
        ("reasoning_accuracy", "Belebele Accuracy", axes[1]),
    ]
    n_models = len(model_order)
    bar_w = 0.22
    group_gap = 0.28

    for acc_col, acc_label, ax in acc_types:
        x_ticks, x_tick_labels = [], []
        for i, locale in enumerate(locale_order):
            locale_df = small[small["locale"] == locale]
            group_center = i * (n_models * bar_w + group_gap)
            for j, mk in enumerate(model_order):
                row = locale_df[locale_df["model_key"] == mk]
                x_pos = group_center + j * bar_w
                if not row.empty:
                    val = row[acc_col].values[0]
                    ax.bar(
                        x_pos,
                        val,
                        width=bar_w * 0.9,
                        color=color_map[mk],
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=3,
                    )
                    ax.text(
                        x_pos,
                        val + 0.008,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        rotation=90,
                        color="#333333",
                    )
                else:
                    ax.text(
                        x_pos,
                        0.01,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        color="#aaaaaa",
                        rotation=90,
                    )
            x_ticks.append(group_center + (n_models - 1) * bar_w / 2)
            x_tick_labels.append(locale_labels[locale])

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=13)
        ax.set_ylabel(acc_label, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 0.80)
        _style_axes(ax)
        ax.axhline(
            CHANCE,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"~Chance ({CHANCE})",
        )
        ax.legend(loc="upper right", fontsize=12)

    # Legend shows regime names only: collapse "multilingual (xglm-564m)" to
    # "multilingual" (the model key keeps its identity for colour/grouping).
    legend_patches, seen = [], set()
    for mk in model_order:
        lbl = "multilingual" if mk.startswith("multilingual (") else mk
        if lbl in seen:
            continue
        seen.add(lbl)
        legend_patches.append(mpatches.Patch(color=color_map[mk], label=lbl))
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        title="Regime",
        title_fontsize=13,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    out = fig_dir / "tradeoff_small.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# -------------------------- Figure 2: scaling curves -------------------------


def plot_scaling_curves(df, fig_dir):
    """Single-row scaling curves: cultural and reasoning accuracy on the same
    y-axis per locale, distinguished by marker + colour. Each panel shows the
    two XGLM curves (cultural, reasoning)."""
    target_locales = TARGET_LOCALES
    metrics = [
        ("cultural_accuracy", "Cultural", C_BI_SMALL, "o"),
        ("reasoning_accuracy", "Reading comprehension", C_REASONING, "s"),
    ]
    families = [("XGLM", XGLM_FAMILY, "-")]

    n_locales = len(target_locales)
    fig, axes = plt.subplots(
        1, n_locales, figsize=(4.5 * n_locales + 1, 5), sharey=True
    )
    if n_locales == 1:
        axes = [axes]
    fig.suptitle(
        f"Scaling curves ({', '.join(target_locales)})",
        fontsize=14,
        y=0.99,
    )

    for c, locale in enumerate(target_locales):
        ax = axes[c]
        loc_df = df[df["locale"] == locale]

        for fam_name, fam_labels, fam_ls in families:
            for acc_col, metric_name, color, marker in metrics:
                pts = []
                for lbl in fam_labels:
                    row = loc_df[loc_df["label"] == lbl]
                    if row.empty:
                        continue
                    pts.append((MODEL_PARAMS[lbl], row[acc_col].values[0]))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                label = metric_name
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    linestyle=fam_ls,
                    linewidth=2,
                    markersize=8,
                    label=label,
                    zorder=3,
                )

        ax.axhline(CHANCE, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xticks(SCALING_XTICKS)
        ax.xaxis.set_major_formatter(FuncFormatter(_params_tick_fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(3e8, 1.2e10)

        ax.grid(True, which="major", linestyle="--", alpha=0.4, color=C_GRID, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(0.15, 0.75)

        ax.set_title(locale.upper(), fontsize=12)
        if c == 0:
            ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_xlabel("Parameters (log scale)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = fig_dir / "scaling_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------- Figure 3: tradeoff (≥1B) ----------------------------


def plot_tradeoff_large(df, fig_dir):
    """≥1B trade-off in the tradeoff_small format: two stacked panels (cultural
    top, reading-comprehension bottom), bars grouped by locale and coloured by
    regime, with the explicit per-locale model set in LARGE_SPEC. Each bar is
    annotated with its display name; locale headers sit above the top panel."""
    present = set(df["locale"])
    loc_order = [l for l in LARGE_LOCALE_ORDER if l in present]
    country = {
        row["locale"]: row["culture_country"]
        for _, row in df.drop_duplicates("locale").iterrows()
    }
    df_idx = df.set_index(["locale", "label"])
    regime_color = {
        r: REGIME_COLOR[r] for r in ("monolingual", "bilingual", "multilingual")
    }

    # One x-unit per bar, a gap between locale groups.
    GROUP_GAP = 1.4
    positions, bar_meta = [], []
    group_center, group_span = {}, {}
    x = 0.0
    for loc in loc_order:
        start = x
        for label, disp, regime in LARGE_SPEC[loc]:
            positions.append(x)
            bar_meta.append((loc, label, disp, regime))
            x += 1.0
        group_center[loc] = (start + x - 1.0) / 2
        group_span[loc] = (start - 0.5, x - 0.5)
        x += GROUP_GAP

    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)

    acc_types = [
        ("cultural_accuracy", "BLEnD Accuracy", axes[0]),
        ("reasoning_accuracy", "Belebele Accuracy", axes[1]),
    ]
    for acc_col, acc_label, ax in acc_types:
        # Faint alternating shading per locale group aids the grouping read.
        for gi, loc in enumerate(loc_order):
            if gi % 2 == 1:
                lo, hi = group_span[loc]
                ax.axvspan(lo, hi, color="0.5", alpha=0.05, zorder=0)
        for pos, (loc, label, disp, regime) in zip(positions, bar_meta):
            try:
                val = float(df_idx.loc[(loc, label), acc_col])
            except KeyError:
                ax.text(
                    pos,
                    0.01,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#aaaaaa",
                    rotation=90,
                )
                continue
            ax.bar(
                pos,
                val,
                width=0.82,
                color=regime_color[regime],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.text(
                pos,
                val + 0.008,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
                color="#333333",
            )
        ax.axhline(
            CHANCE,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"~Chance ({CHANCE})",
        )
        ax.set_ylim(0, 0.85)
        ax.set_ylabel(acc_label, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        _style_axes(ax)
        ax.legend(loc="upper right", fontsize=11)

    # Bottom panel: per-bar model display names. Top panel: locale headers.
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([m[2] for m in bar_meta], rotation=90, fontsize=13)
    axes[1].set_xlim(positions[0] - 1.0, positions[-1] + 1.0)
    trans = axes[0].get_xaxis_transform()
    for loc in loc_order:
        axes[0].text(
            group_center[loc],
            1.02,
            LOCALE_TAG.get(loc, loc),
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(color=regime_color[r], label=r)
        for r in ("monolingual", "bilingual", "multilingual")
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        title="Regime",
        title_fontsize=13,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    out = fig_dir / "tradeoff_large.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# -------------------- Figure 4: regime-gap delta plot ----------------------


def _plot_regime_deltas_core(
    df,
    fig_dir,
    *,
    param_filter,
    locale_filter,
    multi_labels_for_locale,
    subtitle_size_text,
    multi_legend_text,
    output_stem,
    mono_label_for_locale=None,
    bi_label_for_locale=None,
):
    """Shared core for regime-delta diverging-bar plots.

    Diverging bars: monolingual − {bilingual, multilingual} accuracy. Positive
    bar means monolingual wins.

    Args:
        param_filter: callable params -> bool. Selects which models to include.
        locale_filter: list of locales to plot in this order. None means all
            non-English locales found in the filtered data.
        multi_labels_for_locale: callable locale -> list of model labels whose
            mean accuracy serves as the multilingual regime baseline for that locale.
        subtitle_size_text: short text appended to the subtitle (e.g., "<1B
            scale-matched", "≥1B scale-matched (ko, ja, fr)").
        multi_legend_text: full text for the multi-baseline legend entry.
        output_stem: file stem (without extension) for the figure.
        mono_label_for_locale: optional callable locale -> model label pinning
            the monolingual representative. None = pick any row with
            regime=="monolingual" (unambiguous only when one such model exists).
        bi_label_for_locale: optional callable locale -> model label pinning the
            bilingual representative. None = pick any row with
            regime=="bilingual". Pin this for the ≥1B bucket, where a locale can
            have several bilingual models and row order alone is not meaningful.
    """
    bucket = df[df["params"].apply(param_filter)].copy()
    if locale_filter is not None:
        bucket = bucket[bucket["locale"].isin(locale_filter)]
        present = set(bucket["locale"])
        locale_order = [loc for loc in locale_filter if loc in present]
    else:
        present = set(bucket["locale"])
        locale_order = [
            loc for loc in CANONICAL_LOCALES if loc != "en" and loc in present
        ]

    metrics = [
        ("cultural_accuracy", "BLEnD"),
        ("reasoning_accuracy", "Belebele"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    bar_w = 0.38

    def _scalar(s):
        if s.empty:
            return None
        v = s.iloc[0]
        if pd.isna(v):
            return None
        return float(v)

    # First pass: compute deltas for both panels so we can set a shared ylim
    # symmetric around 0 (so the sign comparison is the visual focus).
    panel_deltas: list[tuple] = []  # (ax, acc_label, deltas_bi, deltas_multi)
    for ax, (acc_col, acc_label) in zip(axes, metrics):
        deltas_bi, deltas_multi = [], []
        for loc in locale_order:
            d = bucket[bucket["locale"] == loc]
            if mono_label_for_locale is not None:
                mono = _scalar(d[d["label"] == mono_label_for_locale(loc)][acc_col])
            else:
                mono = _scalar(d[d["regime"] == "monolingual"][acc_col])
            if bi_label_for_locale is not None:
                bi = _scalar(d[d["label"] == bi_label_for_locale(loc)][acc_col])
            else:
                bi = _scalar(d[d["regime"] == "bilingual"][acc_col])
            multi_vals = [
                _scalar(d[d["label"] == m][acc_col])
                for m in multi_labels_for_locale(loc)
            ]
            multi_vals = [v for v in multi_vals if v is not None]
            multi = sum(multi_vals) / len(multi_vals) if multi_vals else None

            deltas_bi.append(
                (mono - bi) if mono is not None and bi is not None else None
            )
            deltas_multi.append(
                (mono - multi) if mono is not None and multi is not None else None
            )
        panel_deltas.append((ax, acc_label, deltas_bi, deltas_multi))

    all_vals = [v for _, _, db, dm in panel_deltas for v in db + dm if v is not None]
    if all_vals:
        max_abs = max(abs(v) for v in all_vals)
        pad = max(0.04, max_abs * 0.18)
        ylim = (-(max_abs + pad), max_abs + pad)
    else:
        ylim = None

    x = np.arange(len(locale_order))

    for ax, acc_label, deltas_bi, deltas_multi in panel_deltas:

        def _draw(positions, values, color, axis=ax):
            for xpos, v in zip(positions, values):
                if v is None:
                    continue
                axis.bar(
                    xpos,
                    v,
                    width=bar_w,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                offset = 0.004 if v >= 0 else -0.004
                axis.text(
                    xpos,
                    v + offset,
                    f"{v:+.2f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8,
                    color="#222",
                )

        _draw(x - bar_w / 2, deltas_bi, C_BI_SMALL)
        _draw(x + bar_w / 2, deltas_multi, C_REGIME_MULTI)

        ax.axhline(0, color="black", linewidth=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LOCALE_TAG.get(loc, loc) for loc in locale_order], fontsize=11
        )
        ax.set_title(acc_label, fontsize=12)
        ax.set_ylabel("Δ accuracy (monolingual − other)", fontsize=10)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color=C_GRID, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_BI_SMALL, label="monolingual − bilingual"),
        mpatches.Patch(color=C_REGIME_MULTI, label=multi_legend_text),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    out = fig_dir / f"{output_stem}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_regime_deltas_small(df, fig_dir):
    """<1B scale-matched delta plot (goldfish vs *-en-intermediate vs xglm-564m)."""
    _plot_regime_deltas_core(
        df,
        fig_dir,
        param_filter=lambda p: p < SMALL_PARAM_THRESHOLD,
        locale_filter=None,
        multi_labels_for_locale=lambda _loc: ["xglm-564m"],
        subtitle_size_text="<1B",
        multi_legend_text="monolingual − multilingual  (xglm-564m)",
        output_stem="regime_deltas_small",
    )


def plot_regime_deltas_large(df, fig_dir):
    """≥1B scale-matched delta plot for ko / ja / fr.

    Representatives are selected per locale from SPECIALIZED_MATCHES.
    """
    _plot_regime_deltas_core(
        df,
        fig_dir,
        param_filter=lambda p: p >= SMALL_PARAM_THRESHOLD,
        locale_filter=TARGET_LOCALES,  # ["ko", "ja", "fr"]
        multi_labels_for_locale=_multilingual_matches,
        mono_label_for_locale=lambda loc: SPECIALIZED_MATCHES[loc]["monolingual"],
        bi_label_for_locale=lambda loc: SPECIALIZED_MATCHES[loc]["bilingual"],
        subtitle_size_text="≥1B",
        multi_legend_text="monolingual − multilingual  (Tiny Aya)",
        output_stem="regime_deltas_large",
    )


# -------------------------------- Driver -----------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot behavior trade-off figures")
    parser.add_argument(
        "--input",
        dest="summary_csv",
        type=Path,
        default=SUMMARY_CSV,
        help="Input behavior summary CSV",
    )
    parser.add_argument(
        "--figure-dir",
        "--output-dir",
        dest="figure_dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Include the Qwen3 family in plots (default: excluded)",
    )
    args = parser.parse_args()

    apply_house_style()
    exclude_qwen = not args.include_qwen
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary_csv, exclude_qwen=exclude_qwen)
    plot_small_regimes(df, args.figure_dir)
    # tradeoff_large shows the full multilingual set (xglm/qwen/eurollm/aya), so
    # it needs the qwen rows even when qwen is globally excluded elsewhere.
    df_large = (
        df if args.include_qwen else load_summary(args.summary_csv, exclude_qwen=False)
    )
    plot_tradeoff_large(df_large, args.figure_dir)
    plot_scaling_curves(df, args.figure_dir)
    plot_regime_deltas_small(df, args.figure_dir)
    plot_regime_deltas_large(df, args.figure_dir)


if __name__ == "__main__":
    main()
