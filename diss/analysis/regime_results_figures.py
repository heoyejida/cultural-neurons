"""Regime results by size tier: task accuracy bars and culture-neuron depth profiles.

Run: python -m diss.analysis.regime_results_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diss.analysis.plot_style import REGIME_COLOURS, apply_plot_style
from diss.analysis.rq1_per_model_table import GE1B_SELECT, SUB1B_LABELS, SUB1B_ORDER

OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG = OUT / "figures"
REGIMES = ["monolingual", "bilingual", "multilingual"]
TIERS = ["<1B parameters", "≥1B parameters"]
TASKS = {"cultural_accuracy": "Ext. BLEnD", "reasoning_accuracy": "Belebele"}
BINS = 10


def select() -> pd.DataFrame:
    df = pd.read_csv(OUT / "behavior_summary.csv")
    keep = {(loc, l) for loc, regs in GE1B_SELECT.items() for ls in regs.values() for l in ls}
    sub1b = df[df["label"].isin(SUB1B_LABELS) & df["locale"].isin(SUB1B_ORDER)]
    ge1b = df[[k in keep for k in zip(df["locale"], df["label"])]]
    return pd.concat([sub1b.assign(tier=TIERS[0]), ge1b.assign(tier=TIERS[1])])


def plot_accuracy(df: pd.DataFrame) -> None:
    cells = df.groupby(["tier", "locale", "regime"])[list(TASKS)].mean()
    means = cells.groupby(["tier", "regime"]).mean()
    print(means.round(3))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    width = 0.27
    for ax, tier in zip(axes, TIERS):
        for i, regime in enumerate(REGIMES):
            x = np.arange(len(TASKS)) + (i - 1) * width
            bars = ax.bar(x, means.loc[(tier, regime)], width, color=REGIME_COLOURS[regime], label=regime.capitalize(), zorder=3)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.axhline(0.25, color="#777777", ls=":", lw=0.8, zorder=2)
        ax.set_xticks(range(len(TASKS)), TASKS.values())
        ax.grid(axis="x", visible=False)
        ax.set_title(tier)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_ylabel("Accuracy")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_regime_accuracy.png")
    plt.close(fig)


def depth_share(layer_counts: dict) -> np.ndarray:
    counts = np.array([layer_counts[str(i)] for i in range(len(layer_counts))], float)
    cdf = np.concatenate([[0], np.cumsum(counts)]) / counts.sum()
    edges = np.interp(np.linspace(0, 1, BINS + 1), np.linspace(0, 1, len(counts) + 1), cdf)
    return np.diff(edges) * 100


def plot_neurons(df: pd.DataFrame) -> None:
    rows = []
    for r in df.itertuples():
        country = r.culture_country.replace(" ", "")
        path = OUT / r.model_name.split("/")[-1] / "cultural_neuron" / f"{country}_neurons_blend_mcq.json"
        if not path.exists():
            print(f"skip {r.tier} {r.regime} {r.label} {r.locale}: no neuron selection")
            continue
        sel = json.loads(path.read_text())
        rows.append((r.tier, r.regime, depth_share(sel["layer_distribution"])))
    centers = (np.arange(BINS) + 0.5) / BINS
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.4), sharex=True, sharey=True)
    for (i, tier), (j, regime) in [(t, g) for t in enumerate(TIERS) for g in enumerate(REGIMES)]:
        ax = axes[i, j]
        shares = np.array([s for t, g, s in rows if (t, g) == (tier, regime)])
        se = shares.std(axis=0, ddof=1) / np.sqrt(len(shares)) if len(shares) > 1 else None
        ax.bar(centers, shares.mean(axis=0), 0.9 / BINS, yerr=se, color=REGIME_COLOURS[regime],
               error_kw={"lw": 0.6, "ecolor": "#555555"}, zorder=3)
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.set_title(regime.capitalize())
        if j == 0:
            ax.set_ylabel(f"{tier}\nCulture neurons (%)")
        if i == 1:
            ax.set_xlabel("Relative layer depth")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    fig.tight_layout()
    fig.savefig(FIG / "fig_regime_neuron_depth.png")
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    FIG.mkdir(parents=True, exist_ok=True)
    df = select()
    plot_accuracy(df)
    plot_neurons(df)


if __name__ == "__main__":
    main()
