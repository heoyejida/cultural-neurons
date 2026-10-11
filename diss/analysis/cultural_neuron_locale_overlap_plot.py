"""Pairwise Jaccard heatmap of selected cultural-neuron sets across locales,
for every multilingual model with ≥3 selected locales.

For each such model, build the symmetric matrix
``J[a, b] = |S_a ∩ S_b| / |S_a ∪ S_b|`` over its selected locales
(intersected with the six-locale display set, English excluded), and render
one heatmap per model in a single 1×N PNG.

Output:
    diss/outputs/figures/cultural_neuron_locale_overlap.png

The CSV companion (full per-pair statistics) is produced by
``diss.analysis.cross_locale_neuron_overlap`` and is not duplicated here.

Run::

    python -m diss.analysis.cultural_neuron_locale_overlap_plot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Pastel sequential colourmap (white -> pastel blue -> muted blue) so the
# Jaccard heatmap matches the seaborn-pastel palette used by the other figures
# instead of the stand-out viridis.
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    "pastel_seq", ["#ffffff", "#a1c9f4", "#5b7db1"]
)

from diss.core.io import model_slug
from diss.core.registry import (
    CULTURE_COUNTRY_BY_LOCALE,
    LOCALE_DISPLAY_ORDER,
    MODEL_PARAMS,
    load_manifest,
)
from diss.analysis.figstyle import LOCALE_CODE, apply_house_style
from diss.analysis._common import NEURON_FILE_RE, load_neuron_selection

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve()
OUTPUTS_DIR = HERE.parents[1] / "outputs"

# Restrict the overlap figure to this multilingual set (model_id ==
# behaviour_summary label). The smaller xglm / qwen variants are omitted so the
# panel set matches the ≥1B multilingual models used elsewhere in the report.
OVERLAP_MODELS = ["qwen3-1p7b-base", "eurollm-1p7b", "tiny-aya-base"]

# Display names for the panel titles (model_id -> shown label).
OVERLAP_DISPLAY = {
    "qwen3-1p7b-base": "Qwen3",
    "eurollm-1p7b": "EuroLLM",
    "tiny-aya-base": "Tiny Aya",
}

# Locales excluded from the overlap matrices: English (never a target) and
# Japanese (dropped from the masking analysis over the byte-length issue, so the
# overlap panels match the ja-excluded RQ2 cohort).
OVERLAP_EXCLUDE_LOCALES = {"en", "ja"}

COUNTRY_TO_LOCALE = {
    v.replace(" ", ""): k for k, v in CULTURE_COUNTRY_BY_LOCALE.items()
}


def collect_multilingual_models(output_root: Path) -> list[dict]:
    """Return [{model_id, slug, regime, sels: {locale: set_of_keys}}, ...] for
    every manifest-enabled multilingual model whose selection directory yields
    ≥3 non-empty locale selections (excluding English)."""
    manifest = load_manifest()
    out: list[dict] = []
    for spec in manifest.models:
        if spec.role != "multilingual" or not spec.neuron_enabled:
            continue
        if spec.model_id not in OVERLAP_MODELS:
            continue
        slug = model_slug(spec.name)
        cn_dir = output_root / slug
        if spec.revision:
            cn_dir = cn_dir / f"cp_{spec.revision}"
        cn_dir = cn_dir / "cultural_neuron"
        if not cn_dir.is_dir():
            continue
        sels: dict[str, set[tuple]] = {}
        for path in sorted(cn_dir.iterdir()):
            m = NEURON_FILE_RE.match(path.name)
            if not m:
                continue
            loc = COUNTRY_TO_LOCALE.get(m.group("country"))
            if loc is None or loc == "en":
                continue
            s = load_neuron_selection(path)
            if s:
                sels[loc] = s
        if len(sels) >= 3:
            out.append(
                {
                    "model_id": spec.model_id,
                    "slug": slug,
                    "regime": spec.role,
                    "sels": sels,
                }
            )
    return out


def jaccard_matrix(
    sels: dict[str, set[tuple]], locales: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Symmetric J matrix over the subset of locales that have selections."""
    present = [l for l in locales if l in sels]
    n = len(present)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(present):
        sa = sels[a]
        for j, b in enumerate(present):
            sb = sels[b]
            union = sa | sb
            M[i, j] = (len(sa & sb) / len(union)) if union else 0.0
    return M, present


def plot_overlap_grid(
    models: list[dict], out_path: Path, locale_order: list[str] = LOCALE_DISPLAY_ORDER
) -> None:
    if not models:
        LOGGER.warning("no multilingual models with ≥3 locales — nothing to plot")
        return
    locales = [l for l in locale_order if l not in OVERLAP_EXCLUDE_LOCALES]

    n = len(models)
    fig_w = max(3.0 * n, 7.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.4), squeeze=False)
    axes = axes[0]

    # Shared colour scale so panels are directly comparable.
    matrices = [jaccard_matrix(m["sels"], locales) for m in models]
    vmax = max(
        (
            np.nanmax(np.where(np.eye(M.shape[0], dtype=bool), np.nan, M))
            for M, _ in matrices
            if M.size > 0
        ),
        default=0.2,
    )
    vmax = max(vmax, 0.05)  # avoid washed-out heatmaps when all Jaccards ~0

    im = None
    for ax, model, (M, labels) in zip(axes, models, matrices):
        im = ax.imshow(M, vmin=0, vmax=vmax, cmap=PASTEL_CMAP, aspect="equal")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels([LOCALE_CODE.get(l, l) for l in labels], fontsize=8)
        ax.set_yticklabels([LOCALE_CODE.get(l, l) for l in labels], fontsize=8)
        ax.set_title(
            OVERLAP_DISPLAY.get(model["model_id"], model["model_id"]), fontsize=10
        )
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    continue  # diagonal is always 1; suppress to reduce noise
                v = M[i, j]
                # Light pastel cells -> dark text; only the deepest cells flip.
                txt_color = "white" if v > vmax * 0.65 else "#222222"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )
        ax.tick_params(axis="both", which="major", length=0)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02, fraction=0.04)
        cbar.set_label("Jaccard $|A \\cap B| / |A \\cup B|$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

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
    parser.add_argument("--fig-out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.fig_out or (
        args.output_root / "figures" / "cultural_neuron_locale_overlap.png"
    )
    models = collect_multilingual_models(args.output_root)
    # Panels in ascending model size (<1B before ≥1B) for cross-figure consistency.
    # MODEL_PARAMS is keyed by model_id (= behavior_summary "label"), not slug.
    models.sort(key=lambda m: MODEL_PARAMS.get(m.get("model_id", ""), float("inf")))
    LOGGER.info("found %d multilingual model(s) with ≥3 locales", len(models))
    for m in models:
        LOGGER.info(
            "  %s: %d locales (%s)",
            m["model_id"],
            len(m["sels"]),
            ",".join(sorted(m["sels"].keys())),
        )
    plot_overlap_grid(models, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
