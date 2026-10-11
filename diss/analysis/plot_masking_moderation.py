"""RQ2 moderation figures — cultural vs random mask drop by moderator level.

Reads the per-cell significance CSV (``diss/outputs/masking_significance_per_cell.csv``,
produced by ``diss.analysis.masking_significance``) and emits three figures under
``diss/outputs/figures/``, one per moderator axis:

  fig_cultural_drop_by_param.png    grouped by parameter-size tier
  fig_cultural_drop_by_regime.png   grouped by language composition
  fig_cultural_drop_by_tokens.png   grouped by per-language token tier

Each figure has two panels (Extended BLEnD, Belebele), and within each panel a
pair of bars per group level: the cultural-neuron mask drop (Cultural mask) and
the same-layer-random mask drop (SLR mask, the most stringent null, which keeps
the cultural set's module group and per-layer depth profile). Bars are the group
mean accuracy drop with a ±1 standard-error error bar. The token figure covers
only the cells with a documented per-language token budget.

Inputs are local CSV, but per the results-figures workflow this still runs on
flamingo (Python execution is flamingo-only); rsync the PNGs back and stage into
``report/figures/`` afterwards.

Run::
    python -m diss.analysis.plot_masking_moderation
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diss.core.registry import MODEL_PARAMS, target_lang_tokens
from diss.analysis.figstyle import SEABORN_PASTEL, COLOR_RANDOM, apply_house_style

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "outputs"
FIG = OUT / "figures"

DATASETS = ("blend_mcq", "belebele")
DS_TITLE = {"blend_mcq": "BLEnD", "belebele": "Belebele"}

# Descriptive tier bins (figure x-axis groups). Kept in step with the tier
# definitions in ``masking_moderation.py``; the labels here use the ≥ glyph.
PARAM_TIERS = (
    ("<200M", 0.0, 200e6),
    ("200M-1B", 200e6, 1e9),
    ("1-3B", 1e9, 3e9),
    ("≥3B", 3e9, float("inf")),
)
TOKEN_TIERS = (
    ("<1B", 0.0, 1e9),
    ("1-10B", 1e9, 1e10),
    ("10-100B", 1e10, 1e11),
    ("≥100B", 1e11, float("inf")),
)
REGIMES = ("monolingual", "bilingual", "multilingual")


def _tier(value: float, tiers) -> str:
    for label, lo, hi in tiers:
        if lo <= value < hi:
            return label
    return tiers[-1][0]


def _f(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _sem(xs: list[float]) -> float:
    """Standard error of the mean (0 for a single observation)."""
    n = len(xs)
    if n <= 1:
        return 0.0
    return float(np.std(xs, ddof=1) / np.sqrt(n))


def load_cells(csv_path: Path, *, exclude_locale: str | None = None) -> list[dict]:
    """Per-(model, locale, dataset) records with the moderator levels joined.

    Keeps the Extended BLEnD and Belebele cells and joins each cell's parameter
    tier and per-language token tier so the bars can be grouped by either axis.
    """
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    cells: list[dict] = []
    for r in rows:
        if r["dataset"] not in DATASETS:
            continue
        if exclude_locale and r["locale"] == exclude_locale:
            continue
        model_id, locale = r["model_id"], r["locale"]
        cult = _f(r, "cultural_drop")
        slr = _f(r, "slr_drop_mean")
        params = MODEL_PARAMS.get(model_id)
        if cult is None or slr is None or params is None:
            LOGGER.warning(
                "skip %s/%s/%s (missing drop/params)", model_id, locale, r["dataset"]
            )
            continue
        tokens = target_lang_tokens(model_id, locale)
        cells.append(
            {
                "model_id": model_id,
                "locale": locale,
                "dataset": r["dataset"],
                "regime": r["regime"],
                "cultural_drop": cult,
                "slr_drop": slr,
                "param_tier": _tier(params, PARAM_TIERS),
                "token_tier": _tier(tokens, TOKEN_TIERS) if tokens else None,
            }
        )
    return cells


def _panel(
    ax,
    cells: list[dict],
    dataset: str,
    key: str,
    levels: list[str],
    *,
    show_ylabel: bool,
) -> None:
    """One BLEnD/Belebele panel: paired cultural/random bars per group level."""
    sub = [c for c in cells if c["dataset"] == dataset and c.get(key) is not None]
    cult = {lvl: [c["cultural_drop"] for c in sub if c[key] == lvl] for lvl in levels}
    slr = {lvl: [c["slr_drop"] for c in sub if c[key] == lvl] for lvl in levels}
    present = [lvl for lvl in levels if cult[lvl]]

    x = np.arange(len(present))
    w = 0.38
    # seaborn pastel: pastel red for the cultural mask, pastel grey for the
    # same-layer-random (SLR) baseline, white edges.
    ax.bar(
        x - w / 2,
        [float(np.mean(cult[l])) for l in present],
        w,
        yerr=[_sem(cult[l]) for l in present],
        label="Cultural mask",
        color=SEABORN_PASTEL[3],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
    )
    ax.bar(
        x + w / 2,
        [float(np.mean(slr[l])) for l in present],
        w,
        yerr=[_sem(slr[l]) for l in present],
        label="SLR mask",
        color=COLOR_RANDOM,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=12)
    ax.set_title(DS_TITLE[dataset], fontsize=15)
    if show_ylabel:
        ax.set_ylabel("Accuracy drop", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)


def plot_axis(cells: list[dict], key: str, levels: list[str], stem: str) -> Path:
    # Slightly shorter than fig_cultural_drop_by_regime so the three stacked
    # full-width panels fit one report float page.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), sharey=True)
    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        _panel(ax, cells, ds, key, levels, show_ylabel=(i == 0))
    fig.tight_layout(pad=0.5, w_pad=0.8)
    out = FIG / f"{stem}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out)
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    apply_house_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cells-in", type=Path, default=OUT / "masking_significance_per_cell.csv"
    )
    ap.add_argument(
        "--exclude-locale", default=None, help="drop cells for this locale (e.g. 'ja')"
    )
    args = ap.parse_args()

    if not args.cells_in.exists():
        raise SystemExit(f"missing {args.cells_in} — run masking_significance first")

    cells = load_cells(args.cells_in, exclude_locale=args.exclude_locale)
    if not cells:
        raise SystemExit("no BLEnD/Belebele cells after load")
    n_tok = len(
        {(c["model_id"], c["locale"]) for c in cells if c["token_tier"] is not None}
    )
    LOGGER.info(
        "loaded %d cells; %d model-locale pairs with a token budget", len(cells), n_tok
    )

    FIG.mkdir(parents=True, exist_ok=True)
    plot_axis(
        cells, "param_tier", [t[0] for t in PARAM_TIERS], "fig_cultural_drop_by_param"
    )
    plot_axis(cells, "regime", list(REGIMES), "fig_cultural_drop_by_regime")
    plot_axis(
        cells, "token_tier", [t[0] for t in TOKEN_TIERS], "fig_cultural_drop_by_tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
