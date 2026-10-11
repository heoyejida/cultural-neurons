"""Shared figure style for ``diss.analysis`` plots.

Single source of truth for colour palettes, locale/regime ordering, and the
matplotlib rcParams that give every dissertation figure one visual tone.
Import these constants/helpers instead of re-defining local palettes so that
the regime colours, locale colours, fonts and grid match across all figures.

Locale *order* is owned by :data:`diss.core.locale_meta.LOCALE_DISPLAY_ORDER`
(the single source feeding figures, summary CSVs and tables alike); this module
re-exports ordering helpers built on top of it.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib as mpl

from diss.core.locale_meta import LOCALE_DISPLAY_ORDER

# ─── regime ordering ─────────────────────────────────────────────────────────
REGIME_ORDER = ["monolingual", "bilingual", "multilingual"]
REGIME_RANK = {r: i for i, r in enumerate(REGIME_ORDER)}

# ─── seaborn "pastel" palette (10 colours, exact hex) ────────────────────────
# seaborn is not installed in the flamingo env, so the seaborn aesthetic is
# reproduced dependency-free: matplotlib's bundled seaborn style for the canvas
# (see apply_house_style) plus this verbatim seaborn pastel palette for colours.
SEABORN_PASTEL = [
    "#a1c9f4",  # blue
    "#ffb482",  # orange
    "#8de5a1",  # green
    "#ff9f9b",  # red
    "#d0bbff",  # purple
    "#debb9b",  # brown
    "#fab0e4",  # pink
    "#cfcfcf",  # grey
    "#fffea3",  # yellow
    "#b9f2f0",  # cyan
]

# ─── regime palette (seaborn "pastel" blue / orange / green) ─────────────────
# Standardised across every figure that distinguishes regimes.
REGIME_COLOR = {
    "monolingual": "#a1c9f4",  # pastel blue
    "bilingual": "#ffb482",  # pastel orange
    "multilingual": "#8de5a1",  # pastel green
}

# ─── locale palette (seaborn "pastel", keyed in display order) ───────────────
# Used by per-locale line / trajectory plots. Distinct pastel hues, same tonal
# family as the regime palette. Figures use either regime *or* locale colours,
# never both at once, so a shared blue/orange/green between the two is harmless.
LOCALE_COLORS = {
    "ko": "#a1c9f4",  # pastel blue
    "ja": "#ffb482",  # pastel orange
    "zh": "#8de5a1",  # pastel green
    "fr": "#ff9f9b",  # pastel red
    "es": "#d0bbff",  # pastel purple
    "ar": "#debb9b",  # pastel brown
    "en": "#cfcfcf",  # pastel grey
}

# ─── locale codes (BCP-47 country-language tags) for axis / legend labels ─────
LOCALE_CODE = {
    "ko": "ko-KR",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "fr": "fr-FR",
    "es": "es-ES",
    "ar": "ar-SA",
    "en": "en-US",
}

# ─── stable accents (kept identical across figures) ──────────────────────────
COLOR_MLP = "#a1c9f4"  # MLP target unit (target unit is MLP-only)
COLOR_RANDOM = "#cfcfcf"  # random / null baselines
COLOR_GRID = "#cccccc"


def locale_rank(locale: str) -> int:
    """Index of ``locale`` in the canonical display order (unknown → last)."""
    try:
        return LOCALE_DISPLAY_ORDER.index(locale)
    except ValueError:
        return len(LOCALE_DISPLAY_ORDER)


def order_locales(locales: Iterable[str], *, drop_en: bool = True) -> list[str]:
    """Filter+sort ``locales`` into canonical display order.

    ``drop_en`` removes English (excluded from the core regime comparison and
    from every per-locale figure)."""
    present = set(locales)
    out = [l for l in LOCALE_DISPLAY_ORDER if l in present]
    if drop_en:
        out = [l for l in out if l != "en"]
    return out


def model_sort_key(regime: str, param_M: float | int | None):
    """Sort key for listing models: mono → bi → multi, then ascending size
    within a regime (so <1B models precede ≥1B models)."""
    return (REGIME_RANK.get(regime, len(REGIME_ORDER)), float(param_M or 0.0))


def apply_house_style() -> None:
    """Apply the shared style. Call once at the top of each plot script's
    ``main`` (after ``matplotlib.use('Agg')`` if set) so the canvas, palette,
    fonts, grid and title weights match across figures.

    The canvas uses matplotlib's bundled seaborn theme (the ``seaborn-v0_8``
    family ships with matplotlib, so no seaborn dependency is needed), overlaid
    with the seaborn pastel colour cycle and the project's rcParams."""
    try:
        mpl.style.use("seaborn-v0_8-whitegrid")
    except OSError:  # older matplotlib names the bundled style differently
        mpl.style.use("seaborn-whitegrid")
    mpl.rcParams.update(
        {
            "axes.prop_cycle": mpl.cycler(color=SEABORN_PASTEL),
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "regular",
            "axes.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.titlesize": 12,
            "figure.titleweight": "regular",
        }
    )
