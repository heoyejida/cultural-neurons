"""Trajectory PNGs for one anchor-based checkpoint-trace run.

Reads ``diss/outputs/cp_summary_{run_id}.csv`` and produces three standalone
figures (one metric each), each a per-locale colour-coded line over steps:

  {stem}_rank.png — Emergence
        anchor_rank_percentile_mean vs training step (log x).
        50 reference line = random-distribution expectation; 100 = anchor cp.

  {stem}_blend_drop.png — Cultural mask drop
        anchor_blend_drop vs training step (log x). This is the raw cultural
        accuracy drop when the anchor set is masked (unmasked − masked), NOT
        the anchor-minus-random gap.

  {stem}_belebele_drop.png — Reading control
        anchor_belebele_drop vs training step (log x), on the same y-scale as
        the cultural drop so the reading control's flatness is directly visible.

One line per locale, colour-coded by locale.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diss.analysis.figstyle import (
    LOCALE_COLORS as _LOCALE_COLOURS,
    LOCALE_CODE as _LOCALE_CODE,
    apply_house_style,
)

LOGGER = logging.getLogger(__name__)

# Legend / draw order for locales (overrides alphabetical).
_LEGEND_ORDER = ["ko", "fr", "es", "ar", "zh", "ja", "en"]


def _loc_rank(locale: str) -> int:
    return (
        _LEGEND_ORDER.index(locale) if locale in _LEGEND_ORDER else len(_LEGEND_ORDER)
    )


# Fixed y-axis ranges so every run's trajectory figures are directly comparable
# rather than each autoscaling to its own data. The raw-drop range contains all
# HPLT2c runs' values (incl. the kor BLEnD drop ~0.212) plus a small margin, and
# the cultural and reading drops share it so the reading control's flatness reads
# directly against the cultural rise.
YLIM_RANK_PCT = (0, 105)
YLIM_RAW_DROP = (-0.05, 0.23)


def _read_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _maybe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _step_from_label(label: str) -> int:
    """Extract the integer training step from a checkpoint label.

    Handles both `"step-12345"` (Beetle/Pico) and `"0000012345"` (Trillion).
    """
    m = re.search(r"\d+", label)
    return int(m.group(0)) if m else 0


def _xpos_log(step: int) -> float:
    """Place a step on a pseudo-log x axis (log10(step+1)).

    Using `step+1` keeps step-0 at x=0 instead of -inf, and gives a sensible
    geometric spacing for the rest.
    """
    return float(np.log10(step + 1))


def _format_step(step: int) -> str:
    """Human-friendly step label: 1024 -> '1k', 32768 -> '33k'."""
    if step == 0:
        return "0"
    if step < 1000:
        return str(step)
    if step < 10_000:
        return f"{step / 1000:.1f}k".rstrip("0").rstrip(".")
    return f"{int(round(step / 1000))}k"


def _draw_locale_line(ax, x, y, locale: str, *, marker: str = "o") -> None:
    color = _LOCALE_COLOURS.get(locale, "C0")
    nan = float("nan")
    ax.plot(
        x,
        [v if v is not None else nan for v in y],
        marker=marker,
        color=color,
        linewidth=2.2,
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=_LOCALE_CODE.get(locale, locale),
    )


def _set_log_step_ticks(ax, steps: list[int]) -> None:
    """Tick at every cp with human-readable step labels, log-spaced."""
    xs = [_xpos_log(s) for s in steps]
    ax.set_xticks(xs)
    ax.set_xticklabels([_format_step(s) for s in steps], rotation=0, fontsize=10)
    ax.set_xlabel("training step", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)


def plot_trajectory(*, run_id: str, summary_csv: Path, out_png: Path) -> None:
    rows = _read_summary(summary_csv)
    if not rows:
        LOGGER.warning("Empty summary %s — skipping plot.", summary_csv)
        return
    _plot_rows(rows, out_dir=out_png.parent, stem=out_png.stem)


def plot_trajectory_combined(
    *, run_ids: list[str], summary_csvs: list[Path], out_png: Path, title: str
) -> None:
    """Overlay several single-locale runs into the three per-metric figures.

    Each HPLT2c run is a single locale, so concatenating their summary rows and
    grouping by locale yields one colour-coded line per locale on shared axes,
    with the legend distinguishing locales. Runs whose summary CSV is missing or
    empty are skipped with a warning. All runs must share the same step grid for
    the overlay to align (true for the HPLT2c traces after the 0007000 unify).
    """
    rows: list[dict] = []
    for run_id, csv_path in zip(run_ids, summary_csvs):
        if not csv_path.exists():
            LOGGER.warning(
                "Combined: summary %s missing — skipping %s", csv_path, run_id
            )
            continue
        run_rows = _read_summary(csv_path)
        if not run_rows:
            LOGGER.warning("Combined: summary %s empty — skipping %s", csv_path, run_id)
            continue
        rows.extend(run_rows)
    if not rows:
        LOGGER.warning("Combined: no rows across %s — skipping plot.", run_ids)
        return
    stem = out_png.stem
    if stem.endswith("_combined"):
        stem = stem[: -len("_combined")]
    _plot_rows(rows, out_dir=out_png.parent, stem=stem)


def _plot_rows(rows: list[dict], *, out_dir: Path, stem: str) -> None:
    """Emit three standalone figures (rank, BLEnD drop, Belebele drop).

    The two drop panels use the *raw* anchor mask drop (anchor_blend_drop /
    anchor_belebele_drop), not the anchor-minus-random gap, so the y-axis is the
    accuracy lost when the anchor set is masked. No star markers and no noise
    band: each figure is one colour-coded line per locale over training steps.
    """
    locales = sorted({r["locale"] for r in rows}, key=_loc_rank)
    by_locale: dict[str, dict] = {}
    all_steps: list[int] = []
    for loc in locales:
        loc_rows = [r for r in rows if r["locale"] == loc]
        loc_rows.sort(key=lambda r: _step_from_label(r["checkpoint_label"]))
        steps = [_step_from_label(r["checkpoint_label"]) for r in loc_rows]
        all_steps = sorted(set(all_steps + steps))
        by_locale[loc] = {
            "steps": steps,
            "rank_pct": [
                _maybe_float(r["anchor_rank_percentile_mean"]) for r in loc_rows
            ],
            "blend_drop": [_maybe_float(r["anchor_blend_drop"]) for r in loc_rows],
            "belebele_drop": [
                _maybe_float(r["anchor_belebele_drop"]) for r in loc_rows
            ],
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, ax, name: str) -> None:
        _set_log_step_ticks(ax, all_steps)
        fig.tight_layout()
        out_path = out_dir / name
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        LOGGER.info("Saved %s", out_path)

    def _lines(ax, key: str) -> None:
        for loc in locales:
            d = by_locale[loc]
            xs = [_xpos_log(s) for s in d["steps"]]
            _draw_locale_line(ax, xs, d[key], loc)

    # (1) Emergence — anchor rank percentile.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(100, color="0.85", linestyle=":", linewidth=0.9)
    _lines(ax, "rank_pct")
    ax.set_ylim(*YLIM_RANK_PCT)
    ax.set_ylabel("Rank percentile", fontsize=11)
    ax.legend(loc="lower right", fontsize=11, markerscale=1.3, frameon=False)
    _save(fig, ax, f"{stem}_rank.png")

    # (2) Cultural mask drop — raw anchor BLEnD drop (no random subtraction).
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0, color="0.4", linestyle="-", linewidth=0.8)
    _lines(ax, "blend_drop")
    ax.set_ylim(*YLIM_RAW_DROP)
    ax.set_ylabel("BLEnD Accuracy Drop", fontsize=11)
    ax.legend(loc="upper left", fontsize=11, markerscale=1.3, frameon=False)
    _save(fig, ax, f"{stem}_blend_drop.png")

    # (3) Reading control — raw anchor Belebele drop, same y-scale as (2).
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0, color="0.4", linestyle="-", linewidth=0.8)
    _lines(ax, "belebele_drop")
    ax.set_ylim(*YLIM_RAW_DROP)
    ax.set_ylabel("Belebele Accuracy Drop", fontsize=11)
    ax.legend(loc="upper left", fontsize=11, markerscale=1.3, frameon=False)
    _save(fig, ax, f"{stem}_belebele_drop.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot checkpoint trajectory PNG")
    parser.add_argument(
        "--run-id", default=None, help="Single-run mode: plot one run's trajectory."
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Defaults to diss/outputs/cp_summary_{run_id}.csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Defaults to diss/outputs/figures/cp_trajectory_{run_id}.png",
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        default=None,
        help="Combined mode: overlay these run-ids (one locale each) "
        "on a single shared figure, colour-coded by locale.",
    )
    parser.add_argument(
        "--combine-out",
        default="diss/outputs/figures/cp_trajectory_hplt2c_combined.png",
        help="Output path for --combine mode.",
    )
    parser.add_argument(
        "--title",
        default="Anchor checkpoint trajectory — HPLT2c",
        help="Suptitle for --combine mode.",
    )
    args = parser.parse_args()

    apply_house_style()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.combine:
        csvs = [Path(f"diss/outputs/cp_summary_{rid}.csv") for rid in args.combine]
        plot_trajectory_combined(
            run_ids=args.combine,
            summary_csvs=csvs,
            out_png=Path(args.combine_out),
            title=args.title,
        )
        return

    if not args.run_id:
        parser.error("either --run-id or --combine is required")
    summary = Path(args.summary or f"diss/outputs/cp_summary_{args.run_id}.csv")
    out = Path(args.out or f"diss/outputs/figures/cp_trajectory_{args.run_id}.png")
    plot_trajectory(run_id=args.run_id, summary_csv=summary, out_png=out)


if __name__ == "__main__":
    main()
