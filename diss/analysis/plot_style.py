"""Shared visual style and save helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PASTEL_COLOURS = (
    "#a1c9f4",
    "#ffb482",
    "#8de5a1",
    "#ff9f9b",
    "#d0bbff",
    "#debb9b",
    "#fab0e4",
    "#cfcfcf",
    "#fffea3",
    "#b9f2f0",
)

LOCALE_COLOURS = {
    "ko": "#a1c9f4",
    "fr": "#ff9f9b",
    "ar": "#debb9b",
}

REGIME_COLOURS = {
    "monolingual": "#a1c9f4",
    "bilingual": "#ffb482",
    "multilingual": "#8de5a1",
}

GRID_COLOUR = "#cccccc"


def apply_plot_style() -> None:
    """Apply the shared white-grid plot style."""

    try:
        mpl.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        mpl.style.use("seaborn-whitegrid")
    mpl.rcParams.update(
        {
            "axes.prop_cycle": mpl.cycler(color=PASTEL_COLOURS),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.size": 9,
            "font.weight": "normal",
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOUR,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_pdf_png(
    fig,
    output_stem: Path,
    *,
    dpi: int,
    transparent: bool,
) -> tuple[Path, Path]:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    save_options = {
        "bbox_inches": "tight",
        "pad_inches": 0.04,
        "transparent": transparent,
    }
    fig.savefig(pdf_path, **save_options)
    fig.savefig(png_path, dpi=dpi, **save_options)
    return pdf_path, png_path


def save_pdf_png(
    fig,
    output_stem: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save ``fig`` as tightly cropped PDF and PNG files and return both paths."""

    return _save_pdf_png(
        fig,
        output_stem,
        dpi=dpi,
        transparent=False,
    )


def save_legend(
    handles,
    labels,
    output_stem: Path,
    *,
    columns: int,
) -> tuple[Path, Path]:
    """Save a transparent standalone legend and return its PDF/PNG paths."""

    apply_plot_style()
    figure = plt.figure(figsize=(3.15, 0.28))
    figure.legend(
        handles,
        labels,
        loc="center",
        ncol=columns,
        frameon=False,
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.4,
    )
    figure.patch.set_alpha(0.0)
    try:
        return _save_pdf_png(
            figure,
            Path(output_stem),
            dpi=300,
            transparent=True,
        )
    finally:
        plt.close(figure)
