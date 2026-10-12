"""Per-(model, locale) layer distribution of selected cultural neurons.

For every (model, locale) cell with a non-empty selection JSON, this script
reads ``top_neurons`` from
``diss/outputs/<slug>/[cp_<rev>/]cultural_neuron/<Country>_neurons_blend_mcq.json``
and plots the *layer distribution*: the count of the cell's selected neurons at
each absolute layer index (0 = first layer). Bars are coloured by locale. Each
panel covers a single model, so its absolute-layer x-axis is exact; depth bands
are read within a model, not across models with different layer counts.

Three figures are emitted, one per regime, so the report can present them as
sub-figures and a 12-layer model's x-axis is not visually conflated with a
28-layer model's:

  - ``diss/outputs/figures/cultural_neuron_layer_dist_monolingual.png``
  - ``diss/outputs/figures/cultural_neuron_layer_dist_bilingual.png``
  - ``diss/outputs/figures/cultural_neuron_layer_dist_multilingual.png``

Each panel is one (model, locale) cell. Rows = models in the regime
(manifest order), cols = locales in display order.

Run::

    python -m diss.analysis.cultural_neuron_layer_distribution
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diss.core.io import load_json, model_slug, output_dir
from diss.core.registry import (
    CULTURE_COUNTRY_BY_LOCALE,
    LOCALE_DISPLAY_ORDER,
    MODEL_PARAMS,
    load_manifest,
)
from diss.analysis.figstyle import (
    REGIME_ORDER,
    COLOR_MLP,
    LOCALE_COLORS,
    LOCALE_CODE,
    apply_house_style,
)

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve()
OUTPUTS_DIR = HERE.parents[1] / "outputs"

# Target unit is MLP-only; attention counts are always zero and the attn bar /
# legend are omitted. COLOR_ATTN kept for backward-compat with any caller.
COLOR_ATTN = "#dd8452"

# Restrict the layer-distribution figures to the explicit per-regime model set
# (model_id == behaviour_summary label): the locale-specialised mono/bi picks
# plus the ≥1B multilingual references, matching tradeoff_large / the overlap
# figure. ja-only models (rinna-*, qwen3-4b-base) drop out because ja is
# excluded below; qwen3-4b-base is also neuron_enabled: false.
DEFAULT_MODEL_FAMILIES: tuple[str, ...] = (
    # monolingual
    "hplt2c-kor",
    "hplt2c-ara",
    "hplt2c-fra",
    "rinna-monolingual-neox-3p6b",
    # bilingual
    "kanana-1p5-2p1b-base",
    "jais-family-1p3b",
    "croissantllm-base",
    "rinna-bilingual-neox-4b",
    # multilingual
    "qwen3-1p7b-base",
    "qwen3-4b-base",
    "eurollm-1p7b",
    "tiny-aya-base",
)

# Display names for the panel labels (model_id -> shown label).
DISPLAY_NAMES: dict[str, str] = {
    "hplt2c-kor": "HPLT 2.0-kor",
    "hplt2c-fra": "HPLT 2.0-fra",
    "hplt2c-ara": "HPLT 2.0-ara",
    "kanana-1p5-2p1b-base": "Kanana 1.5",
    "croissantllm-base": "CroissantLLM",
    "jais-family-1p3b": "Jais",
    "qwen3-1p7b-base": "Qwen3",
    "eurollm-1p7b": "EuroLLM",
    "tiny-aya-base": "Tiny Aya",
}


def _disp(model_id: str) -> str:
    return DISPLAY_NAMES.get(model_id, model_id)


# Locales omitted from the per-regime layer-distribution figures. ``ja`` is
# dropped here so its panel does not appear in any of the three regime grids.
EXCLUDE_LOCALES: tuple[str, ...] = ("ja",)


def _model_is_in_families(model_id: str, families: tuple[str, ...]) -> bool:
    return any(model_id == f or model_id.startswith(f) for f in families)


def _module_kind(logical_module_name: str) -> str:
    """Return 'mlp' or 'attn' from a logical_module_name string.

    Covers the four naming conventions in use:
      - llama-like: ``mlp.gate_proj`` / ``self_attn.q_proj``
      - GPT-2-like: ``mlp.c_fc`` / ``attn.c_attn``
      - XGLM/marian: bare ``fc1`` / ``fc2`` (MLP), no attention attribution
      - HF GPT-NeoX-style: ``mlp.dense_4h_to_h`` / ``attention.*``
    """
    name = logical_module_name.lower()
    if name.startswith("mlp.") or name in {"fc1", "fc2"} or "ffn" in name:
        return "mlp"
    if "attn" in name or "attention" in name:
        return "attn"
    return "mlp"  # default — most pre-LLaMA architectures only attribute MLP


def _num_layers(cn_dir: Path) -> int | None:
    """Read total layer count from the attribution sidecar (lists every layer
    even if not selected). Falls back to layer_distribution max + 1 if absent."""
    scores_path = cn_dir / "blend_mcq_max_scores.json"
    if scores_path.exists():
        payload = load_json(scores_path)
        ls = payload.get("layer_summary")
        if ls:
            return len(ls)
    return None


def collect_cells(
    output_root: Path,
    families: tuple[str, ...] = DEFAULT_MODEL_FAMILIES,
) -> list[dict]:
    """Each cell = a (regime, model_id, locale) with a non-empty selection.

    The cell holds two arrays of length num_layers: counts for MLP and attn.
    Only models whose ``model_id`` matches one of ``families`` are included
    (default: the five families covered by the priority-2 sweep).
    """
    manifest = load_manifest()
    cells: list[dict] = []
    for spec in manifest.models:
        if not spec.neuron_enabled:
            continue
        if not _model_is_in_families(spec.model_id, families):
            continue
        slug = model_slug(spec.name)
        for locale in spec.eval_locales:
            country = CULTURE_COUNTRY_BY_LOCALE.get(locale)
            if country is None:
                continue
            ctag = country.replace(" ", "")
            cn_dir = output_dir(
                slug, "cultural_neuron", revision=spec.revision, root=output_root
            )
            nf = cn_dir / f"{ctag}_neurons_blend_mcq.json"
            if not nf.exists():
                continue
            payload = load_json(nf)
            top = payload.get("top_neurons", [])
            if not top:
                continue
            n_layers = _num_layers(cn_dir)
            if n_layers is None:
                # derive from observed layer indices
                n_layers = max(int(n["layer_idx"]) for n in top) + 1
            mlp_counts = np.zeros(n_layers, dtype=int)
            attn_counts = np.zeros(n_layers, dtype=int)
            for n in top:
                li = int(n["layer_idx"])
                if li >= n_layers:
                    continue
                kind = _module_kind(n["logical_module_name"])
                if kind == "mlp":
                    mlp_counts[li] += 1
                else:
                    attn_counts[li] += 1
            cells.append(
                {
                    "regime": spec.role,
                    "model_id": spec.model_id,
                    "model_slug": slug,
                    "locale": locale,
                    "country": country,
                    "num_layers": n_layers,
                    "mlp": mlp_counts,
                    "attn": attn_counts,
                }
            )
    return cells


def _draw_cell(
    ax,
    cell: dict,
    *,
    top_title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    max_count: float | None = None,
) -> None:
    # Layer distribution: count of the cell's selected neurons at each absolute
    # layer index (0 = first layer), with bars coloured by the cell's locale.
    counts = np.asarray(cell["mlp"], dtype=float) + np.asarray(
        cell["attn"], dtype=float
    )
    n_layers = int(cell["num_layers"])
    layers = np.arange(n_layers)
    color = LOCALE_COLORS.get(cell["locale"], COLOR_MLP)
    ax.bar(layers, counts[:n_layers], color=color, width=0.9, zorder=3)
    upper = (
        max_count if max_count is not None else (float(counts.max()) or 1.0)
    ) * 1.15
    ax.set_ylim(0, max(upper, 1.0))
    ax.set_xlim(-0.5, n_layers - 0.5)
    # Sparse integer ticks so dense-layer models stay readable.
    step = max(1, n_layers // 6)
    ticks = list(range(0, n_layers, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7)
    if top_title is not None:
        ax.set_title(top_title, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def _ordered_models(cells: list[dict], regime: str) -> list[str]:
    """Model IDs for a regime, ascending model size (<1B before ≥1B) then
    manifest order for ties — consistent with the rest of the figures.
    MODEL_PARAMS is keyed by model_id (= behavior_summary "label")."""
    seen: list[str] = []
    for c in cells:
        if c["regime"] != regime:
            continue
        if c["model_id"] not in seen:
            seen.append(c["model_id"])
    manifest_rank = {mid: i for i, mid in enumerate(seen)}
    seen.sort(key=lambda mid: (MODEL_PARAMS.get(mid, float("inf")), manifest_rank[mid]))
    return seen


def plot_regime(
    cells: list[dict],
    regime: str,
    out_path: Path,
    locale_order: list[str] = LOCALE_DISPLAY_ORDER,
) -> None:
    """Grid layout depends on whether models in this regime span multiple locales:

    - "wide" regime (multilingual): rows = models, cols = locales — full grid.
    - "tall" regime (mono / bi): each model evaluates a single locale, so the
      models-x-locales grid would be mostly empty. Collapse to a single row,
      one panel per (model, locale) cell, sorted by locale display order.
    """
    locales_all = [l for l in locale_order if l != "en" and l not in EXCLUDE_LOCALES]
    models = _ordered_models(cells, regime)
    if not models:
        LOGGER.warning("no cells for regime=%s — skipping %s", regime, out_path.name)
        return

    by_key: dict[tuple[str, str], dict] = {
        (c["model_id"], c["locale"]): c for c in cells if c["regime"] == regime
    }
    # Detect "tall" regime: every model covers exactly one locale.
    cells_per_model = {m: [l for l in locales_all if (m, l) in by_key] for m in models}
    is_tall = all(len(v) <= 1 for v in cells_per_model.values())

    rank_locale = {l: i for i, l in enumerate(locales_all)}
    if is_tall:
        # one panel per (model, locale) cell, sorted by locale
        panels: list[tuple[str, str]] = sorted(
            ((m, l) for m, ls in cells_per_model.items() for l in ls),
            key=lambda ml: (rank_locale.get(ml[1], 99), ml[0]),
        )
        nrows, ncols = 1, max(1, len(panels))
    else:
        panels = [
            (m, l) for m in models for l in locales_all if (m, l) in by_key or True
        ]  # iterate full grid; missing cells become axis off
        nrows, ncols = len(models), len(locales_all)

    fig_w = max(2.2 * ncols, 8.0)
    fig_h = max(2.2 * nrows, 2.6)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharey=False
    )

    if is_tall:
        # Each panel is a different model; let every panel autoscale to its own
        # neuron-count peak so the depth band is legible regardless of model size.
        for j, (model_id, locale) in enumerate(panels):
            ax = axes[0, j]
            c = by_key[(model_id, locale)]
            _draw_cell(
                ax,
                c,
                top_title=f"{LOCALE_CODE.get(locale, locale)}\n{_disp(model_id)}",
                ylabel="Neuron count" if j == 0 else None,
                xlabel="Layer",
                max_count=None,
            )
    else:
        for i, model_id in enumerate(models):
            row_cells = [by_key.get((model_id, l)) for l in locales_all]
            present = [c for c in row_cells if c is not None]
            if not present:
                for j in range(ncols):
                    axes[i, j].axis("off")
                continue
            # One model per row, so share a neuron-count scale across its
            # locale columns (same model, hence comparable counts).
            max_count = (
                max(
                    (
                        float(
                            (
                                np.asarray(c["mlp"], dtype=float)
                                + np.asarray(c["attn"], dtype=float)
                            ).max()
                        )
                        for c in present
                    ),
                    default=1.0,
                )
                or 1.0
            )
            for j, locale in enumerate(locales_all):
                ax = axes[i, j]
                c = by_key.get((model_id, locale))
                if c is None:
                    ax.axis("off")
                    continue
                _draw_cell(
                    ax,
                    c,
                    top_title=(LOCALE_CODE.get(locale, locale) if i == 0 else None),
                    ylabel=(_disp(model_id) if j == 0 else None),
                    xlabel=("Layer" if i == nrows - 1 else None),
                    max_count=max_count,
                )

    # Bars are coloured by locale, and every panel is already locale-labelled
    # (column header for the grid, panel title for the single-row regimes), so a
    # separate colour legend would be redundant.
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    apply_house_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=None)
    args = parser.parse_args()

    fig_dir = args.fig_dir or (args.output_root / "figures")
    cells = collect_cells(args.output_root)
    LOGGER.info("collected %d cells", len(cells))

    for regime in REGIME_ORDER:
        out_path = fig_dir / f"cultural_neuron_layer_dist_{regime}.png"
        plot_regime(cells, regime, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
