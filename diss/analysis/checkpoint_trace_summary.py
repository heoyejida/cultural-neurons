"""Build trajectory summary CSV from anchor-based checkpoint outputs.

The anchor cp is ``labels[-1]`` (the final/most-trained checkpoint). At that
cp, the cultural-neuron set is fixed (the *anchor*); every other cp re-runs
masking with the same anchor neuron file (and the same matched random /
same-layer-random baselines generated at the anchor cp).

Per (run, checkpoint, locale, country) row, ``cp_summary_{run_id}.csv``:

  Behaviour (unmasked accuracy at this cp):
    blend_acc_all, blend_acc_non_neuron, belebele_acc

  Anchor attribution (this cp's full attribution scores, restricted to the
  fixed anchor coords for the target country):
    anchor_size              - count of anchor neurons (constant per locale)
    anchor_attr_sum          - sum of attribution scores
    anchor_attr_mean         - mean of attribution scores
    anchor_rank_percentile_mean
                             - mean of rank-percentile-in-distribution
                               (100 = top of full attribution distribution)

  Anchor masking (drop = unmasked - masked, this cp):
    anchor_blend_drop, anchor_belebele_drop
    anchor_cross_blend_drop_max  - max off-diagonal anchor mask drop on this
                                   locale's blend non_neuron data

  Anchor-random / anchor-same-layer-random baselines (final-cp baselines
  applied at this cp; aggregated over seeds):
    random_blend_drop_mean, random_blend_drop_std
    random_belebele_drop_mean, random_belebele_drop_std
    same_layer_random_blend_drop_mean, same_layer_random_blend_drop_std
    same_layer_random_belebele_drop_mean, same_layer_random_belebele_drop_std

  Anchor effect over baselines:
    anchor_drop_over_random_blend, anchor_drop_over_random_belebele
    anchor_drop_over_slr_blend, anchor_drop_over_slr_belebele

  Stability of the full attribution ranking between adjacent cps:
    spearman_attr_with_prev_cp
"""

from __future__ import annotations

import bisect
import csv
import logging
import statistics
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from diss.core.io import (
    load_json,
    masked_output_dir,
    model_slug,
    output_dir,
    read_results_csv,
    results_accuracy,
)
from diss.core.registry import CULTURE_COUNTRY_BY_LOCALE
from diss.prep.blend import NON_NEURON_TOPICS

LOGGER = logging.getLogger(__name__)


def _read_acc(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    correct = sum(1 for r in rows if str(r.get("is_correct", "")).lower() == "true")
    return correct / len(rows)


def _coord_key(family: str, logical_module_name: str, layer_idx, neuron_idx) -> str:
    return f"{family}|{logical_module_name}|{layer_idx}|{neuron_idx}"


def _selected_neuron_keys(neuron_file: Path) -> set[str]:
    if not neuron_file.exists():
        return set()
    payload = load_json(neuron_file)
    return {
        _coord_key(
            n["family"], n["logical_module_name"], n["layer_idx"], n["neuron_idx"]
        )
        for n in payload.get("top_neurons", [])
    }


def _iter_labels(run: dict) -> list[str]:
    src = run["source"]
    if src["kind"] == "hf_revisions":
        return [str(r) for r in src["revisions"]]
    return [cp["label"] for cp in src["checkpoints"]]


def _country_score_vector(
    scores_path: Path, country: str
) -> tuple[list[str], list[float]]:
    """Aligned (neuron_keys, target-country score) from a max_scores.json."""
    if not scores_path.exists():
        return [], []
    payload = load_json(scores_path)
    keys: list[str] = []
    vals: list[float] = []
    for rec in payload.get("neuron_scores", []):
        keys.append(
            _coord_key(
                rec.get("family", ""),
                rec.get("logical_module_name", ""),
                rec.get("layer_idx", ""),
                rec.get("neuron_idx", ""),
            )
        )
        vals.append(float(rec.get("scores", {}).get(country, 0.0)))
    return keys, vals


def _spearman_with_prev(curr_path: Path, prev_path: Path, country: str) -> float | None:
    curr_keys, curr_vals = _country_score_vector(curr_path, country)
    prev_keys, prev_vals = _country_score_vector(prev_path, country)
    if not curr_keys or not prev_keys:
        return None
    prev_lookup = dict(zip(prev_keys, prev_vals))
    aligned_curr: list[float] = []
    aligned_prev: list[float] = []
    for k, v in zip(curr_keys, curr_vals):
        if k in prev_lookup:
            aligned_curr.append(v)
            aligned_prev.append(prev_lookup[k])
    if len(aligned_curr) < 2:
        return None
    rho, _ = scipy_stats.spearmanr(aligned_curr, aligned_prev)
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return None
    return float(rho)


def _anchor_attribution_stats(
    scores_path: Path,
    anchor_keys: set[str],
    country: str,
) -> tuple[float | None, float | None, float | None]:
    """Sum, mean, and mean rank-percentile of anchor neurons at this cp.

    Rank-percentile uses the full attribution distribution at this cp (target
    country), with 100 = top, 0 = bottom.
    """
    if not scores_path.exists() or not anchor_keys:
        return None, None, None
    payload = load_json(scores_path)
    records = payload.get("neuron_scores", [])
    if not records:
        return None, None, None

    score_by_key: dict[str, float] = {}
    all_scores: list[float] = []
    for rec in records:
        key = _coord_key(
            rec.get("family", ""),
            rec.get("logical_module_name", ""),
            rec.get("layer_idx", ""),
            rec.get("neuron_idx", ""),
        )
        s = float(rec.get("scores", {}).get(country, 0.0))
        score_by_key[key] = s
        all_scores.append(s)
    n = len(all_scores)
    if n == 0:
        return None, None, None
    sorted_asc = sorted(all_scores)

    anchor_scores: list[float] = []
    rank_pcts: list[float] = []
    for key in anchor_keys:
        s = score_by_key.get(key)
        if s is None:
            continue
        anchor_scores.append(s)
        rank_below = bisect.bisect_left(sorted_asc, s)
        rank_pcts.append(100.0 * rank_below / n)

    if not anchor_scores:
        return None, None, None
    return (
        sum(anchor_scores),
        statistics.fmean(anchor_scores),
        statistics.fmean(rank_pcts),
    )


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def build_trajectory_summary(
    *,
    run: dict,
    slug: str,
    out_csv: Path,
    random_seeds: list[int],
) -> list[dict]:
    labels = _iter_labels(run)
    if not labels:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("", encoding="utf-8")
        return []

    anchor_label = labels[-1]
    locales = run["locales"]
    source = run["source"]
    checkpoint_slugs = (
        {
            cp["label"]: model_slug(str(Path(source["base_path"]) / cp["path"]))
            for cp in source["checkpoints"]
        }
        if source["kind"] == "local_checkpoints"
        else {label: slug for label in labels}
    )

    # Pre-resolve anchor neuron files and their key sets, per (locale, country).
    anchor_files: dict[tuple[str, str], Path] = {}
    anchor_key_sets: dict[tuple[str, str], set[str]] = {}
    for loc in locales:
        country = CULTURE_COUNTRY_BY_LOCALE[loc]
        ctag = country.replace(" ", "")
        nf = (
            output_dir(
                checkpoint_slugs[anchor_label], "cultural_neuron", revision=anchor_label
            )
            / f"{ctag}_neurons_blend_mcq.json"
        )
        anchor_files[(loc, country)] = nf
        anchor_key_sets[(loc, country)] = _selected_neuron_keys(nf)

    rows: list[dict] = []
    for label_idx, label in enumerate(labels):
        checkpoint_slug = checkpoint_slugs[label]
        prev_label = labels[label_idx - 1] if label_idx > 0 else None
        for loc in locales:
            country = CULTURE_COUNTRY_BY_LOCALE[loc]
            ctag = country.replace(" ", "")
            anchor_keys = anchor_key_sets[(loc, country)]
            anchor_stem = anchor_files[(loc, country)].stem

            # Unmasked behaviour at this cp
            blend_path = (
                output_dir(checkpoint_slug, f"blend_mcq_{loc}", revision=label)
                / "results.csv"
            )
            blend_rows = read_results_csv(blend_path) if blend_path.exists() else []
            blend_all = results_accuracy(blend_rows)[0] if blend_rows else None
            blend_nn = (
                results_accuracy(blend_rows, topics=NON_NEURON_TOPICS)[0]
                if blend_rows
                else None
            )
            belebele = _read_acc(
                output_dir(checkpoint_slug, f"belebele_{loc}", revision=label)
                / "results.csv"
            )

            # Anchor attribution stats at this cp
            scores_path = (
                output_dir(checkpoint_slug, "cultural_neuron", revision=label)
                / "blend_mcq_max_scores.json"
            )
            attr_sum, attr_mean, rank_pct_mean = _anchor_attribution_stats(
                scores_path,
                anchor_keys,
                country,
            )

            # Anchor diagonal masking drops at this cp
            anchor_blend_masked = _read_acc(
                masked_output_dir(
                    checkpoint_slug,
                    anchor_stem,
                    f"blend_mcq_{loc}_non_neuron",
                    revision=label,
                )
                / "results.csv"
            )
            anchor_belebele_masked = _read_acc(
                masked_output_dir(
                    checkpoint_slug, anchor_stem, f"belebele_{loc}", revision=label
                )
                / "results.csv"
            )
            anchor_blend_drop = _diff(blend_nn, anchor_blend_masked)
            anchor_belebele_drop = _diff(belebele, anchor_belebele_masked)

            # Anchor cross-locale mask: other countries' anchor sets on this loc's data
            cross_drops: list[float] = []
            for other_loc in locales:
                if other_loc == loc:
                    continue
                other_country = CULTURE_COUNTRY_BY_LOCALE[other_loc]
                other_stem = anchor_files[(other_loc, other_country)].stem
                cross_masked = _read_acc(
                    masked_output_dir(
                        checkpoint_slug,
                        other_stem,
                        f"blend_mcq_{loc}_non_neuron",
                        revision=label,
                    )
                    / "results.csv"
                )
                cd = _diff(blend_nn, cross_masked)
                if cd is not None:
                    cross_drops.append(cd)
            anchor_cross_blend_drop_max = max(cross_drops) if cross_drops else None

            # Anchor random / SLR baseline drops at this cp
            random_blend_drops: list[float] = []
            random_belebele_drops: list[float] = []
            slr_blend_drops: list[float] = []
            slr_belebele_drops: list[float] = []
            for seed in random_seeds:
                rb = _read_acc(
                    masked_output_dir(
                        checkpoint_slug,
                        f"{ctag}_all_neurons_random{seed}",
                        f"blend_mcq_{loc}_non_neuron",
                        revision=label,
                    )
                    / "results.csv"
                )
                rl = _read_acc(
                    masked_output_dir(
                        checkpoint_slug,
                        f"{ctag}_all_neurons_random{seed}",
                        f"belebele_{loc}",
                        revision=label,
                    )
                    / "results.csv"
                )
                slr_b = _read_acc(
                    masked_output_dir(
                        checkpoint_slug,
                        f"{ctag}_all_neurons_random_layer{seed}",
                        f"blend_mcq_{loc}_non_neuron",
                        revision=label,
                    )
                    / "results.csv"
                )
                slr_l = _read_acc(
                    masked_output_dir(
                        checkpoint_slug,
                        f"{ctag}_all_neurons_random_layer{seed}",
                        f"belebele_{loc}",
                        revision=label,
                    )
                    / "results.csv"
                )
                if (d := _diff(blend_nn, rb)) is not None:
                    random_blend_drops.append(d)
                if (d := _diff(belebele, rl)) is not None:
                    random_belebele_drops.append(d)
                if (d := _diff(blend_nn, slr_b)) is not None:
                    slr_blend_drops.append(d)
                if (d := _diff(belebele, slr_l)) is not None:
                    slr_belebele_drops.append(d)

            rb_mean, rb_std = _mean_std(random_blend_drops)
            rl_mean, rl_std = _mean_std(random_belebele_drops)
            slr_b_mean, slr_b_std = _mean_std(slr_blend_drops)
            slr_l_mean, slr_l_std = _mean_std(slr_belebele_drops)

            # Spearman of full attribution rank between adjacent cps
            spearman_prev: float | None = None
            if prev_label is not None:
                curr_scores = (
                    output_dir(checkpoint_slug, "cultural_neuron", revision=label)
                    / "blend_mcq_max_scores.json"
                )
                prev_scores = (
                    output_dir(
                        checkpoint_slugs[prev_label], "cultural_neuron", revision=prev_label
                    )
                    / "blend_mcq_max_scores.json"
                )
                spearman_prev = _spearman_with_prev(curr_scores, prev_scores, country)

            rows.append(
                {
                    "run_id": run["id"],
                    "checkpoint_label": label,
                    "is_anchor_cp": label == anchor_label,
                    "source_kind": run["source"]["kind"],
                    "source_value": run["source"].get("repo")
                    or run["source"].get("base_path"),
                    "locale": loc,
                    "country": country,
                    "anchor_size": len(anchor_keys),
                    "blend_acc_all": _fmt(blend_all),
                    "blend_acc_non_neuron": _fmt(blend_nn),
                    "belebele_acc": _fmt(belebele),
                    "anchor_attr_sum": _fmt(attr_sum),
                    "anchor_attr_mean": _fmt(attr_mean),
                    "anchor_rank_percentile_mean": _fmt(rank_pct_mean),
                    "anchor_blend_drop": _fmt(anchor_blend_drop),
                    "anchor_belebele_drop": _fmt(anchor_belebele_drop),
                    "anchor_cross_blend_drop_max": _fmt(anchor_cross_blend_drop_max),
                    "random_blend_drop_mean": _fmt(rb_mean),
                    "random_blend_drop_std": _fmt(rb_std),
                    "random_belebele_drop_mean": _fmt(rl_mean),
                    "random_belebele_drop_std": _fmt(rl_std),
                    "same_layer_random_blend_drop_mean": _fmt(slr_b_mean),
                    "same_layer_random_blend_drop_std": _fmt(slr_b_std),
                    "same_layer_random_belebele_drop_mean": _fmt(slr_l_mean),
                    "same_layer_random_belebele_drop_std": _fmt(slr_l_std),
                    "anchor_drop_over_random_blend": _fmt(
                        _diff(anchor_blend_drop, rb_mean)
                    ),
                    "anchor_drop_over_random_belebele": _fmt(
                        _diff(anchor_belebele_drop, rl_mean)
                    ),
                    "anchor_drop_over_slr_blend": _fmt(
                        _diff(anchor_blend_drop, slr_b_mean)
                    ),
                    "anchor_drop_over_slr_belebele": _fmt(
                        _diff(anchor_belebele_drop, slr_l_mean)
                    ),
                    "spearman_attr_with_prev_cp": _fmt(spearman_prev),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")
    LOGGER.info("Saved %s (%d rows)", out_csv, len(rows))
    return rows


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    return v
