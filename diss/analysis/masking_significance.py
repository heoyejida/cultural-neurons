"""Significance of cultural-neuron mask drops against the same-layer-random null.

Reviewer concern this module answers: the masking baseline is an *N-seed
average* (5 seeds for the 564 M multilinguals, 10 for the Goldfish/
intermediate cohort), and the claim "cultural drop exceeds same-layer-random
(SLR)" was previously judged by a point estimate only. With N seeds there is a
genuine null distribution, so we attach, for every (model, locale, dataset)
cell:

  1. an **item-level paired bootstrap CI** for the cultural drop itself
     (resampling the held-out items the masked eval ran on);
  2. a **same-layer-random null** built from the per-seed SLR drops, giving
       - a parametric z-score  z = (cult_drop - mean_slr) / sd_slr  and its
         one-sided p (H1: cultural drops MORE than SLR), and
       - an exact one-sided empirical / permutation p
         p = (1 + #{slr_drop >= cult_drop}) / (n_slr + 1);
  3. the same two quantities against the module-type-matched **random** null.

Aggregating across cells (overall, per dataset, per regime) we add:
  - **Stouffer** combination of the per-cell one-sided z (the across-cell test
    that survives even when each cell's 5–10-seed null is individually
    underpowered);
  - a **cell-level bootstrap CI** on the mean (cultural - SLR) drop gap;
  - the paired **Wilcoxon** cult-vs-SLR / cult-vs-random with a matched-pairs
    **rank-biserial** effect size (the earlier table reported p only);
  - **Benjamini-Hochberg FDR** over the per-cell p-values (how many cells
    survive multiple-comparison control).

Data source: the per-seed masked evals already on disk —
  masked/{Country}_neurons_blend_mcq/{ds_subdir}/results.csv       (cultural)
  masked/{Country}_all_neurons_random{s}/{ds_subdir}/results.csv    (random)
  masked/{Country}_all_neurons_random_layer{s}/{ds_subdir}/results.csv (SLR)
with the unmasked baseline matched on the same item ids (mirrors
cultural_neuron_masking.py and build_consolidated_mask_compare.py).

Outputs (diss/outputs/):
  masking_significance_per_cell.csv
  masking_significance.json
  masking_significance.md
  tables/tab_masking_significance.tex

Run:
    python -m diss.analysis.masking_significance
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from diss.core.io import (
    masked_output_dir,
    model_slug,
    output_dir,
    read_results_csv,
)
from diss.core.registry import CULTURE_COUNTRY_BY_LOCALE, load_manifest
from diss.prep.blend import NON_NEURON_TOPICS

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "outputs"
TABLES_OUT = OUT / "tables"

DATASETS = ("blend_mcq", "belebele", "countryrc_localized")
DS_LABEL = {
    "blend_mcq": "BLEnD",
    "belebele": "Belebele",
    "countryrc_localized": "CountryRC",
}
REGIMES = ("monolingual", "bilingual", "multilingual")

# Bootstrap configuration (fixed seed -> reproducible CIs).
B_ITEM = 10000  # item-level resamples for the per-cell cultural-drop CI
B_CELL = 10000  # cell-level resamples for the aggregate gap CIs
RNG_SEED = 20260529
ALPHA = 0.05
MAX_SEEDS = 16  # probe seeds 0..MAX_SEEDS-1; in practice 5 (564M) or 10 exist


# ─────────────────────────────────────────────────────────────────────────────
#  Per-item readers
# ─────────────────────────────────────────────────────────────────────────────


def _correct_map(
    path: Path, *, topics: set[str] | None = None
) -> dict[str, int] | None:
    """{id: 0/1} on the row-instance byte-norm ``is_correct`` field.

    ``topics`` filters to the BLEnD non-neuron held-out split; raises if the
    CSV pre-dates the topic column (same contract as core.results_accuracy)."""
    if not path.exists():
        return None
    rows = read_results_csv(path)
    if topics is not None:
        if rows and "topic" not in rows[0]:
            raise RuntimeError(f"{path} has no 'topic' column; re-run the eval")
        rows = [r for r in rows if r["topic"] in topics]
    if not rows:
        return None
    return {
        r["id"]: (1 if str(r.get("is_correct", "")).lower() == "true" else 0)
        for r in rows
    }


def _ds_subdir(ds: str, locale: str) -> str:
    if ds == "belebele":
        return f"belebele_{locale}"
    return f"{ds}_{locale}_non_neuron"


def _unmasked_map(base: Path, ds: str, locale: str) -> dict[str, int] | None:
    """Unmasked per-item correctness on the slice the masked eval ran on."""
    if ds == "blend_mcq":
        return _correct_map(
            base / f"blend_mcq_{locale}" / "results.csv", topics=NON_NEURON_TOPICS
        )
    if ds == "countryrc_localized":
        return _correct_map(
            base / f"countryrc_localized_{locale}_non_neuron" / "results.csv"
        )
    return _correct_map(base / f"belebele_{locale}" / "results.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────


def _matched_rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial r = (R+ - R-)/(R+ + R-) on nonzero diffs.

    +1 -> every pair has cultural drop > baseline drop; -1 the reverse."""
    d = diff[diff != 0]
    if d.size == 0:
        return float("nan")
    r = stats.rankdata(np.abs(d))
    r_pos = r[d > 0].sum()
    r_neg = r[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total else float("nan")


def _bh_fdr_significant(pvals: list[float], q: float = ALPHA) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of rejections at FDR<=q."""
    p = np.asarray([x for x in pvals], dtype=float)
    valid = ~np.isnan(p)
    out = np.zeros(p.shape, dtype=bool)
    pv = p[valid]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return out
    kmax = np.max(np.nonzero(below)[0])
    cut = ranked[kmax]
    rej = pv <= cut
    out[np.nonzero(valid)[0]] = rej
    return out


def _percentile_ci(
    samples: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def _stouffer(zs: list[float]) -> dict:
    """Stouffer combination of one-sided z-scores (H1: cult drop > null)."""
    z = np.asarray([v for v in zs if v is not None and np.isfinite(v)], dtype=float)
    if z.size == 0:
        return {"k": 0, "Z": None, "p_one_sided": None, "p_two_sided": None}
    Z = float(z.sum() / math.sqrt(z.size))
    return {
        "k": int(z.size),
        "Z": Z,
        "p_one_sided": float(stats.norm.sf(Z)),
        "p_two_sided": float(2 * stats.norm.sf(abs(Z))),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-cell computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_cell(
    slug: str,
    country: str,
    locale: str,
    ds: str,
    regime: str,
    model_id: str,
    revision: str | None,
    rng: np.random.Generator,
    *,
    b_item: int,
) -> dict | None:
    base = output_dir(slug, "", revision=revision, root=OUT)
    ctag = country.replace(" ", "")
    sub = _ds_subdir(ds, locale)

    cult_map = _correct_map(
        masked_output_dir(
            slug, f"{ctag}_neurons_blend_mcq", sub, revision=revision, root=OUT
        )
        / "results.csv"
    )
    unmasked_map = _unmasked_map(base, ds, locale)
    if cult_map is None or unmasked_map is None:
        return None

    shared = sorted(set(cult_map) & set(unmasked_map))
    if len(shared) < 5:
        return None
    u = np.array([unmasked_map[i] for i in shared], dtype=float)
    c = np.array([cult_map[i] for i in shared], dtype=float)
    unmasked_acc = float(u.mean())
    cult_acc = float(c.mean())
    cult_drop = unmasked_acc - cult_acc

    # Item-level paired bootstrap of the cultural drop on the shared items.
    n = len(shared)
    idx = rng.integers(0, n, size=(b_item, n))
    boot = u[idx].mean(axis=1) - c[idx].mean(axis=1)
    drop_lo, drop_hi = _percentile_ci(boot)
    boot_p_drop_le_0 = float((boot <= 0).mean())  # P(bootstrap drop <= 0)

    def _seed_drops(prefix: str) -> list[float]:
        """Per-seed drops over a fixed seed range (seeds are 0..9 in practice;
        we probe MAX_SEEDS and keep whichever exist — no contiguity assumed,
        mirroring build_consolidated_mask_compare.py)."""
        drops = []
        for seed in range(MAX_SEEDS):
            m = _correct_map(
                masked_output_dir(
                    slug, f"{ctag}_{prefix}{seed}", sub, revision=revision, root=OUT
                )
                / "results.csv"
            )
            if m is None:
                continue
            vals = [m[i] for i in shared if i in m]
            if vals:
                drops.append(unmasked_acc - float(np.mean(vals)))
        return drops

    slr_drops = _seed_drops("all_neurons_random_layer")
    rand_drops = _seed_drops("all_neurons_random")

    def _null_stats(drops: list[float]) -> dict:
        arr = np.asarray(drops, dtype=float)
        k = arr.size
        if k == 0:
            return {
                "n_seeds": 0,
                "mean": None,
                "sd": None,
                "z": None,
                "p_z_one_sided": None,
                "p_emp_one_sided": None,
            }
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if k > 1 else 0.0
        z = (cult_drop - mean) / sd if sd > 0 else None
        return {
            "n_seeds": int(k),
            "mean": mean,
            "sd": sd,
            "z": float(z) if z is not None else None,
            "p_z_one_sided": float(stats.norm.sf(z)) if z is not None else None,
            "p_emp_one_sided": float((1 + int((arr >= cult_drop).sum())) / (k + 1)),
        }

    slr = _null_stats(slr_drops)
    rand = _null_stats(rand_drops)

    return {
        "model_id": model_id,
        "slug": slug,
        "regime": regime,
        "locale": locale,
        "country": country,
        "dataset": ds,
        "revision": revision or "",
        "n_items": n,
        "unmasked_acc": unmasked_acc,
        "cultural_masked_acc": cult_acc,
        "cultural_drop": cult_drop,
        "cultural_drop_ci_lo": drop_lo,
        "cultural_drop_ci_hi": drop_hi,
        "boot_p_drop_le_0": boot_p_drop_le_0,
        "slr": slr,
        "random": rand,
        # convenience flag: drop CI strictly above the SLR null mean
        "exceeds_slr_null": (slr["mean"] is not None and drop_lo > slr["mean"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def aggregate(cells: list[dict], rng: np.random.Generator) -> dict:
    out: dict = {"by_dataset": {}, "by_regime_blend": {}}

    def _agg_group(group: list[dict], baseline: str) -> dict:
        rows = [c for c in group if c[baseline]["z"] is not None]
        cult = np.array([c["cultural_drop"] for c in group], dtype=float)
        base_mean = np.array(
            [c[baseline]["mean"] for c in group if c[baseline]["mean"] is not None],
            dtype=float,
        )
        paired = [
            (c["cultural_drop"], c[baseline]["mean"])
            for c in group
            if c[baseline]["mean"] is not None
        ]
        res: dict = {
            "n_cells": len(group),
            "n_cells_with_null": len(rows),
            "mean_cultural_drop": float(cult.mean()) if cult.size else None,
            "mean_baseline_drop": float(base_mean.mean()) if base_mean.size else None,
        }
        # Stouffer over per-cell one-sided z
        res["stouffer"] = _stouffer([c[baseline]["z"] for c in rows])
        # Cell-level bootstrap CI on mean(cult - baseline) gap
        if paired:
            gap = np.array([a - b for a, b in paired], dtype=float)
            k = gap.size
            bidx = rng.integers(0, k, size=(B_CELL, k))
            boot = gap[bidx].mean(axis=1)
            lo, hi = _percentile_ci(boot)
            res["mean_gap"] = float(gap.mean())
            res["mean_gap_ci"] = [lo, hi]
            res["gap_p_le_0"] = float((boot <= 0).mean())
            # Paired Wilcoxon + rank-biserial effect size on the gap
            if k >= 2 and np.any(gap != 0):
                a = np.array([p[0] for p in paired])
                b = np.array([p[1] for p in paired])
                w, p = stats.wilcoxon(a, b, zero_method="zsplit")
                res["wilcoxon"] = {
                    "stat": float(w),
                    "p": float(p),
                    "rank_biserial": _matched_rank_biserial(gap),
                }
        # per-cell p tables + BH-FDR
        emp_p = [c[baseline]["p_emp_one_sided"] for c in group]
        z_p = [c[baseline]["p_z_one_sided"] for c in group]
        res["n_sig_emp_raw"] = int(sum(1 for p in emp_p if p is not None and p < ALPHA))
        res["n_sig_z_raw"] = int(sum(1 for p in z_p if p is not None and p < ALPHA))
        res["n_sig_emp_bh"] = int(_bh_fdr_significant(emp_p).sum())
        res["n_sig_z_bh"] = int(_bh_fdr_significant(z_p).sum())
        res["n_drop_ci_excludes_0"] = int(
            sum(1 for c in group if c["cultural_drop_ci_lo"] > 0)
        )
        res["n_drop_ci_above_slr"] = int(sum(1 for c in group if c["exceeds_slr_null"]))
        return res

    for ds in DATASETS:
        group = [c for c in cells if c["dataset"] == ds]
        if not group:
            continue
        out["by_dataset"][ds] = {
            "vs_slr": _agg_group(group, "slr"),
            "vs_random": _agg_group(group, "random"),
        }

    blend = [c for c in cells if c["dataset"] == "blend_mcq"]
    for regime in REGIMES:
        group = [c for c in blend if c["regime"] == regime]
        if not group:
            continue
        out["by_regime_blend"][regime] = {
            "vs_slr": _agg_group(group, "slr"),
            "vs_random": _agg_group(group, "random"),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Writers
# ─────────────────────────────────────────────────────────────────────────────


def write_per_cell_csv(cells: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
        "cultural_drop_ci_lo",
        "cultural_drop_ci_hi",
        "boot_p_drop_le_0",
        "slr_n_seeds",
        "slr_drop_mean",
        "slr_drop_sd",
        "slr_z",
        "slr_p_z_one_sided",
        "slr_p_emp_one_sided",
        "random_n_seeds",
        "random_drop_mean",
        "random_drop_sd",
        "random_z",
        "random_p_z_one_sided",
        "random_p_emp_one_sided",
        "exceeds_slr_null",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in cells:

            def f(x):
                return "" if x is None else (f"{x:.4f}" if isinstance(x, float) else x)

            w.writerow(
                {
                    "model_id": c["model_id"],
                    "slug": c["slug"],
                    "regime": c["regime"],
                    "locale": c["locale"],
                    "country": c["country"],
                    "dataset": c["dataset"],
                    "revision": c["revision"],
                    "n_items": c["n_items"],
                    "unmasked_acc": f(c["unmasked_acc"]),
                    "cultural_masked_acc": f(c["cultural_masked_acc"]),
                    "cultural_drop": f(c["cultural_drop"]),
                    "cultural_drop_ci_lo": f(c["cultural_drop_ci_lo"]),
                    "cultural_drop_ci_hi": f(c["cultural_drop_ci_hi"]),
                    "boot_p_drop_le_0": f(c["boot_p_drop_le_0"]),
                    "slr_n_seeds": c["slr"]["n_seeds"],
                    "slr_drop_mean": f(c["slr"]["mean"]),
                    "slr_drop_sd": f(c["slr"]["sd"]),
                    "slr_z": f(c["slr"]["z"]),
                    "slr_p_z_one_sided": f(c["slr"]["p_z_one_sided"]),
                    "slr_p_emp_one_sided": f(c["slr"]["p_emp_one_sided"]),
                    "random_n_seeds": c["random"]["n_seeds"],
                    "random_drop_mean": f(c["random"]["mean"]),
                    "random_drop_sd": f(c["random"]["sd"]),
                    "random_z": f(c["random"]["z"]),
                    "random_p_z_one_sided": f(c["random"]["p_z_one_sided"]),
                    "random_p_emp_one_sided": f(c["random"]["p_emp_one_sided"]),
                    "exceeds_slr_null": c["exceeds_slr_null"],
                }
            )


def _fmt(x, p=4, dash="--"):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    if isinstance(x, float) and abs(x) < 10**-p and x != 0:
        return f"<10^-{p}"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def _fmt_tex(x, p=4, dash="--"):
    """LaTeX-safe variant of _fmt: render the tiny-value sentinel in math mode.

    ``_fmt`` returns the bare string ``<10^-4`` for very small magnitudes, which
    is valid Markdown but invalid LaTeX (a bare ``^`` outside math mode aborts the
    build). Wrap that one case in inline math; everything else passes through.
    """
    s = _fmt(x, p, dash)
    return f"$<\\!10^{{-{p}}}$" if s.startswith("<10^-") else s


def write_markdown(cells: list[dict], agg: dict, path: Path) -> None:
    L: list[str] = []
    L.append("# Cultural-mask significance vs same-layer-random null\n")
    L.append(
        f"Cells: {len(cells)} (model, locale, dataset). "
        f"Item bootstrap B={B_ITEM}, cell bootstrap B={B_CELL}, "
        f"RNG seed {RNG_SEED}.\n"
    )
    L.append("\n## Method\n")
    L.append(
        "- Per cell: cultural drop = unmasked - cultural-masked accuracy on "
        "the shared held-out items; **item-level paired bootstrap** 95% CI.\n"
    )
    L.append(
        "- **Same-layer-random null**: per-seed SLR drops give mean/sd; "
        "z = (cult_drop - mean_slr)/sd_slr (one-sided H1: cult > SLR) and an "
        "exact empirical p = (1 + #{slr>=cult})/(n_slr+1).\n"
    )
    L.append(
        "- Aggregates: Stouffer combined z across cells; cell-level bootstrap "
        "CI on the mean (cult - SLR) gap; paired Wilcoxon + rank-biserial; "
        "Benjamini-Hochberg FDR over per-cell p.\n"
    )

    L.append("\n## Aggregate by dataset\n")
    L.append(
        "| dataset | baseline | n | mean cult drop | mean base drop | "
        "mean gap [95% CI] | Stouffer Z (p1) | Wilcoxon p (r) | "
        "#cells z-sig raw/BH | #drop CI>0 | #drop CI>SLR |"
    )
    L.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: |")
    for ds in DATASETS:
        d = agg["by_dataset"].get(ds)
        if not d:
            continue
        for base_key, base_lbl in (("vs_slr", "SLR"), ("vs_random", "random")):
            r = d[base_key]
            gap_ci = (
                f"{_fmt(r.get('mean_gap'))} "
                f"[{_fmt(r.get('mean_gap_ci',[None,None])[0])}, "
                f"{_fmt(r.get('mean_gap_ci',[None,None])[1])}]"
            )
            st = r["stouffer"]
            st_s = f"{_fmt(st['Z'],2)} (p={_fmt(st['p_one_sided'])})"
            w = r.get("wilcoxon", {})
            w_s = (
                f"{_fmt(w.get('p'))} (r={_fmt(w.get('rank_biserial'),2)})"
                if w
                else "--"
            )
            L.append(
                f"| {DS_LABEL[ds]} | {base_lbl} | {r['n_cells']} | "
                f"{_fmt(r['mean_cultural_drop'])} | {_fmt(r['mean_baseline_drop'])} | "
                f"{gap_ci} | {st_s} | {w_s} | "
                f"{r['n_sig_z_raw']}/{r['n_sig_z_bh']} | "
                f"{r['n_drop_ci_excludes_0']} | {r['n_drop_ci_above_slr']} |"
            )

    L.append("\n## BLEnD by regime (vs SLR)\n")
    L.append(
        "| regime | n | mean cult drop | mean SLR drop | mean gap [95% CI] | "
        "Stouffer Z (p1) | #cells z-sig raw/BH |"
    )
    L.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
    for regime in REGIMES:
        d = agg["by_regime_blend"].get(regime)
        if not d:
            continue
        r = d["vs_slr"]
        gap_ci = (
            f"{_fmt(r.get('mean_gap'))} "
            f"[{_fmt(r.get('mean_gap_ci',[None,None])[0])}, "
            f"{_fmt(r.get('mean_gap_ci',[None,None])[1])}]"
        )
        st = r["stouffer"]
        L.append(
            f"| {regime} | {r['n_cells']} | {_fmt(r['mean_cultural_drop'])} | "
            f"{_fmt(r['mean_baseline_drop'])} | {gap_ci} | "
            f"{_fmt(st['Z'],2)} (p={_fmt(st['p_one_sided'])}) | "
            f"{r['n_sig_z_raw']}/{r['n_sig_z_bh']} |"
        )

    L.append("\n## Per-cell BLEnD detail (cultural drop vs SLR null)\n")
    L.append(
        "| model | locale | n | cult drop [95% CI] | SLR drop mean±sd | "
        "z | p(z) | p(emp) | >SLR null |"
    )
    L.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | :-: |")
    for c in sorted(
        [c for c in cells if c["dataset"] == "blend_mcq"],
        key=lambda c: (REGIMES.index(c["regime"]), c["model_id"], c["locale"]),
    ):
        s = c["slr"]
        ci = f"{_fmt(c['cultural_drop'],3)} [{_fmt(c['cultural_drop_ci_lo'],3)}, {_fmt(c['cultural_drop_ci_hi'],3)}]"
        nulls = (
            f"{_fmt(s['mean'],3)}±{_fmt(s['sd'],3)}" if s["mean"] is not None else "--"
        )
        L.append(
            f"| {c['model_id']} | {c['locale']} | {c['n_items']} | {ci} | {nulls} | "
            f"{_fmt(s['z'],2)} | {_fmt(s['p_z_one_sided'])} | "
            f"{_fmt(s['p_emp_one_sided'])} | {'Y' if c['exceeds_slr_null'] else ''} |"
        )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_latex(agg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "% Auto-generated by diss.analysis.masking_significance — do not edit.",
        r"\begin{tabular}{ll rrr r r}",
        r"\toprule",
        r"Dataset & Null & $N$ & Mean cult.\ drop & Mean gap [95\% CI] "
        r"& Stouffer $Z$ ($p$) & Wilcoxon $r$ \\",
        r"\midrule",
    ]
    for ds in DATASETS:
        d = agg["by_dataset"].get(ds)
        if not d:
            continue
        for base_key, base_lbl in (("vs_slr", "SLR"), ("vs_random", "Random")):
            r = d[base_key]
            ci = r.get("mean_gap_ci", [None, None])
            gap = (
                f"{_fmt_tex(r.get('mean_gap'),3)} "
                f"[{_fmt_tex(ci[0],3)}, {_fmt_tex(ci[1],3)}]"
            )
            st = r["stouffer"]
            w = r.get("wilcoxon", {})
            L.append(
                f"{DS_LABEL[ds]} & {base_lbl} & {r['n_cells']} & "
                f"{_fmt_tex(r['mean_cultural_drop'],3)} & {gap} & "
                f"{_fmt_tex(st['Z'],2)} ({_fmt_tex(st['p_one_sided'])}) & "
                f"{_fmt_tex(w.get('rank_biserial'),2)} \\\\"
            )
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────

# Dropped from the masking cohort: their >=1B monolingual role is now filled by
# the architecture-matched HPLT2c suite (see the RQ1 regime comparison).
EXCLUDE_MODEL_IDS = {
    "polyglot-ko-1p3b",
    "gpt-fr-cased-base",
    "aragpt2-mega",
    "beetle-zh-en-50-b2",
    # gemma-3 discarded from the analysis (2026-06): adding gemma-1b to
    # the base cohort tipped the (already unstable) multilingual term to
    # significance without changing the scale result, and the cross-arch
    # role was not retained. Eval outputs are kept on disk but out of cohort.
    "gemma-3-270m",
    "gemma-3-1b",
    # HPLT2c es/zh monolinguals dropped from the masking cohort (2026-06):
    # the >=1B monolingual slot is restricted to ko/ar/fr, matching the
    # RQ3 checkpoint-trace set and the RQ2 scope (HPLT2c mono = ko/ar/fr).
    # es and zh remain represented through the multilingual models.
    # Eval outputs stay on disk but out of the reported cohort.
    "hplt2c-spa",
    "hplt2c-zhos",
}
# Multilingual size-probe models kept out of the base regime cohort and reported
# only in the within-family parameter-size comparison.
SCALE_ONLY_IDS = {"xglm-1p7b", "qwen3-0p6b-base"}
# Japanese is excluded from RQ2 entirely (byte-norm degeneracy, Section sec_rq2_japanese).
EXCLUDE_LOCALES = {"ja"}


def collect_cells(
    rng: np.random.Generator, *, b_item: int, with_scale: bool = False
) -> list[dict]:
    manifest = load_manifest()
    excluded = EXCLUDE_MODEL_IDS if with_scale else (EXCLUDE_MODEL_IDS | SCALE_ONLY_IDS)
    cells: list[dict] = []
    for spec in manifest.models:
        if not getattr(spec, "neuron_enabled", False):
            continue
        if spec.model_id in excluded:
            continue
        slug = model_slug(spec.name)
        for locale in spec.eval_locales:
            if locale in EXCLUDE_LOCALES:
                continue
            country = CULTURE_COUNTRY_BY_LOCALE.get(locale)
            if country is None:
                continue
            for ds in DATASETS:
                cell = compute_cell(
                    slug,
                    country,
                    locale,
                    ds,
                    spec.role,
                    spec.model_id,
                    getattr(spec, "revision", None),
                    rng,
                    b_item=b_item,
                )
                if cell is not None:
                    cells.append(cell)
                    LOGGER.info(
                        "cell %s/%s/%s: cult_drop=%.4f slr_z=%s",
                        spec.model_id,
                        locale,
                        ds,
                        cell["cultural_drop"],
                        _fmt(cell["slr"]["z"], 2),
                    )
    return cells


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b-item", type=int, default=B_ITEM)
    ap.add_argument(
        "--csv-out", type=Path, default=OUT / "masking_significance_per_cell.csv"
    )
    ap.add_argument("--json-out", type=Path, default=OUT / "masking_significance.json")
    ap.add_argument("--md-out", type=Path, default=OUT / "masking_significance.md")
    ap.add_argument(
        "--tex-out", type=Path, default=TABLES_OUT / "tab_masking_significance.tex"
    )
    ap.add_argument(
        "--with-scale",
        action="store_true",
        help="Include the multilingual size-probe models (xglm-1.7B, "
        "qwen3-0.6B) for the within-family parameter-size comparison.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    cells = collect_cells(rng, b_item=args.b_item, with_scale=args.with_scale)
    LOGGER.info("collected %d (model, locale, dataset) cells", len(cells))
    if not cells:
        LOGGER.error("no cells found — did summarize / run-attribution finish?")
        return 1

    agg = aggregate(cells, rng)

    write_per_cell_csv(cells, args.csv_out)
    write_markdown(cells, agg, args.md_out)
    write_latex(agg, args.tex_out)
    args.json_out.write_text(
        json.dumps(
            {
                "n_cells": len(cells),
                "b_item": args.b_item,
                "b_cell": B_CELL,
                "rng_seed": RNG_SEED,
                "aggregate": agg,
            },
            indent=2,
            default=str,
        )
    )

    for ds in DATASETS:
        d = agg["by_dataset"].get(ds)
        if d:
            s = d["vs_slr"]["stouffer"]
            print(
                f"[{ds}] vs SLR: mean cult drop={d['vs_slr']['mean_cultural_drop']:.4f}, "
                f"Stouffer Z={s['Z']:.2f} (p1={s['p_one_sided']:.3g}), "
                f"gap CI={d['vs_slr'].get('mean_gap_ci')}"
            )
    print(f"[done] wrote {args.csv_out.name}, .json, .md, {args.tex_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
