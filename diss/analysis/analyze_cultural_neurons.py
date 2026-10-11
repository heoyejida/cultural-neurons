"""Cultural-neuron cross-cutting analysis across the masking cohort (cohort size
read from the consolidated mask-compare; currently 30 models / 50 cells).

Produces statistical tests, layer-distribution stats, cross-locale overlap
matrices, and LaTeX table snippets for the dissertation results section.

Inputs:
  diss/outputs/cultural_neuron_mask_compare.csv   — consolidated mask-compare results
  diss/outputs/{slug}/cultural_neuron/{tag}_neurons_blend_mcq.json   — neuron coords + layer

Outputs (diss/outputs/):
  tables/tab_mask_summary_by_regime.tex
  tables/tab_significance_tests.tex
  tables/tab_layer_concentration.tex
  tables/tab_overlap_summary.tex
  figures/.../*.png  (handled by plot_cultural_neurons.py)
  cultural_neuron_analysis.md
  cultural_neuron_layer_distribution.csv
  cultural_neuron_overlap_matrix.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from diss.core.registry import MODEL_PARAMS

OUT = PROJECT / "diss" / "outputs"
TABLES_OUT = OUT / "tables"
TABLES_OUT.mkdir(parents=True, exist_ok=True)

LOCALES = ["ko", "es", "zh", "ar", "ja", "fr"]
COUNTRY_OF = {
    "ko": "South Korea",
    "es": "Spain",
    "zh": "China",
    "ar": "Saudi Arabia",
    "ja": "Japan",
    "fr": "France",
}
TAG_OF = {loc: COUNTRY_OF[loc].replace(" ", "") for loc in LOCALES}

# Cross-locale overlap (Jaccard) is restricted to the >=1B multilingual models
# and excludes Japanese (unreliable CountryRC control), matching the locale-
# overlap figure (cultural_neuron_locale_overlap_plot.py). Other analyses in
# this module keep the full LOCALES / model set.
OVERLAP_MODELS = ["Qwen3-1.7B-Base", "EuroLLM-1.7B", "tiny-aya-base"]
OVERLAP_LOCALES = [loc for loc in LOCALES if loc != "ja"]

# (slug, regime, architecture_family, param_M)
MODEL_INFO = {
    "kor_hang_1000mb": ("monolingual", "gpt2_like", 125),
    "spa_latn_1000mb": ("monolingual", "gpt2_like", 125),
    "zho_hans_1000mb": ("monolingual", "gpt2_like", 125),
    "arb_arab_1000mb": ("monolingual", "gpt2_like", 125),
    "jpn_jpan_1000mb": ("monolingual", "gpt2_like", 125),
    "fra_latn_1000mb": ("monolingual", "gpt2_like", 125),
    "ko-en-intermediate": ("bilingual", "gpt2_like", MODEL_PARAMS["ko-en-intermediate"] / 1e6),
    "es-en-intermediate": ("bilingual", "gpt2_like", MODEL_PARAMS["es-en-intermediate"] / 1e6),
    "zh-en-intermediate": ("bilingual", "gpt2_like", MODEL_PARAMS["zh-en-intermediate"] / 1e6),
    "ar-en-intermediate": ("bilingual", "gpt2_like", MODEL_PARAMS["ar-en-intermediate"] / 1e6),
    "fr-en-intermediate": ("bilingual", "gpt2_like", MODEL_PARAMS["fr-en-intermediate"] / 1e6),
    "ja-en-intermediate": ("bilingual", "gpt2_like", MODEL_PARAMS["ja-en-intermediate"] / 1e6),
    "polyglot-ko-1.3b": ("monolingual", "gpt_neox_like", 1300),
    "kanana-1.5-2.1b-base": ("bilingual", "llama_like", 2100),
    "rinna-japanese-gpt-neox-3.6b": ("monolingual", "gpt_neox_like", 3600),
    "bilingual-gpt-neox-4b": ("bilingual", "gpt_neox_like", 3800),
    "Qwen3-4B-Base": ("multilingual", "llama_like", 4000),
    "gpt-fr-cased-base": ("monolingual", "gpt2_like", 110),
    "CroissantLLMBase": ("bilingual", "llama_like", 1300),
    "aragpt2-mega": ("monolingual", "gpt2_like", 1500),
    "jais-family-1p3b": ("bilingual", "gpt2_like", 1300),
    # HPLT 2.0 monolinguals (~2.15B, LlamaForCausalLM)
    "hplt2c_kor_checkpoints": ("monolingual", "llama_like", 2150),
    "hplt2c_ara_checkpoints": ("monolingual", "llama_like", 2150),
    "hplt2c_fra_checkpoints": ("monolingual", "llama_like", 2150),
    "hplt2c_zhos_checkpoints": ("monolingual", "llama_like", 2150),
    "hplt2c_spa_checkpoints": ("monolingual", "llama_like", 2150),
    "xglm-564M": ("multilingual", "xglm_like", 564),
    "Qwen3-1.7B-Base": ("multilingual", "llama_like", 1700),
    "EuroLLM-1.7B": ("multilingual", "llama_like", 1700),
    "tiny-aya-base": ("multilingual", "llama_like", 3350),
}


def _slug(name: str) -> str:
    return name.split("/")[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  Load consolidated mask-compare CSV + enrich with regime/architecture/size
# ─────────────────────────────────────────────────────────────────────────────


def load_mask_compare() -> pd.DataFrame:
    csv_path = OUT / "cultural_neuron_mask_compare.csv"
    df = pd.read_csv(csv_path)
    df["slug"] = df["model"].apply(_slug)
    df["regime"] = df["slug"].map(lambda s: MODEL_INFO[s][0])
    df["arch"] = df["slug"].map(lambda s: MODEL_INFO[s][1])
    df["param_M"] = df["slug"].map(lambda s: MODEL_INFO[s][2])
    for col in [
        "unmasked_acc",
        "cultural_masked_acc",
        "cultural_drop",
        "random_masked_mean",
        "random_masked_std",
        "random_drop_mean",
        "slr_masked_mean",
        "slr_masked_std",
        "slr_drop_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical tests
# ─────────────────────────────────────────────────────────────────────────────


def per_row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of cultural_drop against the random-baseline distribution.

    z = (cultural_drop - random_drop_mean) / random_masked_std
    Sign convention: positive z means cultural mask drops accuracy MORE than
    random masking, i.e. the selected neurons are above the random null.
    """
    df = df.copy()
    df["z_vs_random"] = (df["cultural_drop"] - df["random_drop_mean"]) / df[
        "random_masked_std"
    ]
    # Two-tailed p from z
    df["p_vs_random"] = 2 * (1 - stats.norm.cdf(df["z_vs_random"].abs()))
    return df


def regime_significance(df: pd.DataFrame, dataset: str) -> dict:
    """Mann-Whitney U comparing cultural_drop distributions between regimes,
    plus per-regime descriptive stats."""
    sub = df[df["dataset"] == dataset].dropna(subset=["cultural_drop"])
    groups = {
        r: sub[sub["regime"] == r]["cultural_drop"].values
        for r in ("monolingual", "bilingual", "multilingual")
    }
    descriptives = {
        r: {
            "n": len(v),
            "mean": float(np.mean(v)) if len(v) else float("nan"),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            "median": float(np.median(v)) if len(v) else float("nan"),
        }
        for r, v in groups.items()
    }
    pairs = [
        ("monolingual", "bilingual"),
        ("monolingual", "multilingual"),
        ("bilingual", "multilingual"),
    ]
    pair_tests = {}
    for a, b in pairs:
        if len(groups[a]) >= 2 and len(groups[b]) >= 2:
            u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            pair_tests[f"{a}_vs_{b}"] = {"U": float(u), "p": float(p)}
        else:
            pair_tests[f"{a}_vs_{b}"] = {"U": float("nan"), "p": float("nan")}
    return {"descriptives": descriptives, "pairs": pair_tests}


def overall_paired_tests(df: pd.DataFrame) -> dict:
    """Cultural drop vs random drop, paired across all (model, locale, dataset)
    rows. Wilcoxon (non-parametric) plus paired t-test as a sanity check.
    Done per dataset because effect sizes differ by domain."""
    out = {}
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds].dropna(
            subset=["cultural_drop", "random_drop_mean"]
        )
        cult = sub["cultural_drop"].values
        rand = sub["random_drop_mean"].values
        slr = sub["slr_drop_mean"].values
        out[ds] = {
            "n": len(sub),
            "cultural_mean_drop": float(np.mean(cult)),
            "random_mean_drop": float(np.mean(rand)),
            "slr_mean_drop": float(np.mean(slr)),
            "wilcoxon_cult_vs_random": dict(
                zip(("stat", "p"), stats.wilcoxon(cult, rand, zero_method="zsplit"))
            ),
            "wilcoxon_cult_vs_slr": dict(
                zip(("stat", "p"), stats.wilcoxon(cult, slr, zero_method="zsplit"))
            ),
            "t_cult_vs_random_p": float(stats.ttest_rel(cult, rand).pvalue),
        }
        for k, v in out[ds].items():
            if isinstance(v, dict):
                out[ds][k] = {kk: float(vv) for kk, vv in v.items()}
    return out


def size_correlation(df: pd.DataFrame, dataset: str = "blend_mcq") -> dict:
    sub = df[df["dataset"] == dataset].dropna(subset=["cultural_drop", "param_M"])
    log_size = np.log10(sub["param_M"].values)
    drop = sub["cultural_drop"].values
    pear = stats.pearsonr(log_size, drop)
    spear = stats.spearmanr(log_size, drop)
    return {
        "n": len(sub),
        "x": "log10(param_M)",
        "y": "cultural_drop",
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Layer distribution
# ─────────────────────────────────────────────────────────────────────────────


def load_neuron_layers(slug: str, country: str) -> list[int] | None:
    tag = country.replace(" ", "")
    p = OUT / slug / "cultural_neuron" / f"{tag}_neurons_blend_mcq.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    return [int(n["layer_idx"]) for n in payload.get("top_neurons", [])]


def collect_layer_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen = set()
    for _, r in df[df["dataset"] == "blend_mcq"].iterrows():
        key = (r["slug"], r["locale"])
        if key in seen:
            continue
        seen.add(key)
        layers = load_neuron_layers(r["slug"], r["target_country"])
        if not layers:
            continue
        counts = Counter(layers)
        n_total = sum(counts.values())
        n_layers = max(counts.keys()) + 1
        for li, cnt in counts.items():
            rows.append(
                {
                    "slug": r["slug"],
                    "regime": r["regime"],
                    "arch": r["arch"],
                    "param_M": r["param_M"],
                    "locale": r["locale"],
                    "layer_idx": li,
                    "rel_depth": li / max(n_layers - 1, 1),
                    "neuron_count": cnt,
                    "neuron_share": cnt / n_total,
                    "total_selected": n_total,
                    "n_layers": n_layers,
                }
            )
    return pd.DataFrame(rows)


def layer_concentration(layer_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, locale): top-3 layers by neuron count, % of selected total."""
    rows = []
    for (slug, locale), group in layer_df.groupby(["slug", "locale"]):
        total = group["total_selected"].iloc[0]
        top3 = group.nlargest(3, "neuron_count")
        top3_share = top3["neuron_count"].sum() / total if total else float("nan")
        rows.append(
            {
                "slug": slug,
                "locale": locale,
                "n_layers": int(group["n_layers"].iloc[0]),
                "total_selected": int(total),
                "top3_share": top3_share,
                "top3_layers": ",".join(
                    str(int(li)) for li in top3["layer_idx"].values
                ),
                "regime": group["regime"].iloc[0],
                "param_M": int(group["param_M"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["regime", "param_M", "slug", "locale"])


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-locale neuron overlap
# ─────────────────────────────────────────────────────────────────────────────


def neuron_set(slug: str, country: str) -> set[tuple[int, str, int]] | None:
    tag = country.replace(" ", "")
    p = OUT / slug / "cultural_neuron" / f"{tag}_neurons_blend_mcq.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    return {
        (int(n["layer_idx"]), str(n["logical_module_name"]), int(n["neuron_idx"]))
        for n in payload.get("top_neurons", [])
    }


def overlap_matrix(slug: str) -> pd.DataFrame | None:
    sets = {}
    for loc in OVERLAP_LOCALES:
        s = neuron_set(slug, COUNTRY_OF[loc])
        if s is None:
            continue
        sets[loc] = s
    if len(sets) < 2:
        return None
    mat = pd.DataFrame(index=list(sets), columns=list(sets), dtype=float)
    for a in sets:
        for b in sets:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            mat.loc[a, b] = inter / union if union else float("nan")
    return mat


def collect_overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    multi_slugs = [
        s
        for s in df[df["regime"] == "multilingual"]["slug"].unique()
        if s in OVERLAP_MODELS
    ]
    for slug in multi_slugs:
        mat = overlap_matrix(slug)
        if mat is None:
            continue
        # Off-diagonal mean jaccard
        m = mat.values
        n = m.shape[0]
        off = m[np.triu_indices(n, k=1)]
        rows.append(
            {
                "slug": slug,
                "n_locales": n,
                "mean_pairwise_jaccard": float(np.nanmean(off)),
                "min_pairwise_jaccard": float(np.nanmin(off)),
                "max_pairwise_jaccard": float(np.nanmax(off)),
            }
        )
        # Save matrix to per-model CSV for plot
        mat.to_csv(OUT / f"cultural_neuron_overlap_{slug}.csv")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX table generators
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(x, p=3, dash="--"):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    if abs(x) < 10**-p and x != 0:
        return f"$<10^{{-{p}}}$"
    return f"{x:.{p}f}"


def write_summary_by_regime(df: pd.DataFrame, regime_stats: dict) -> Path:
    """3-column (regime) × dataset summary table."""
    lines = [
        "% Auto-generated by analyze_cultural_neurons.py — do not edit by hand.",
        r"\begin{tabular}{l l rrrr}",
        r"\toprule",
        r"Dataset & Regime & $N$ & Cultural drop (mean) & Random drop (mean) & Cultural $-$ Random \\",
        r"\midrule",
    ]
    for ds in ("blend_mcq", "belebele", "countryrc_localized"):
        for regime in ("monolingual", "bilingual", "multilingual"):
            sub = df[(df["dataset"] == ds) & (df["regime"] == regime)].dropna(
                subset=["cultural_drop"]
            )
            n = len(sub)
            cult_mean = sub["cultural_drop"].mean() if n else float("nan")
            rand_mean = sub["random_drop_mean"].mean() if n else float("nan")
            diff = cult_mean - rand_mean if n else float("nan")
            ds_label = {
                "blend_mcq": "BLEnD",
                "belebele": "Belebele",
                "countryrc_localized": "CountryRC",
            }[ds]
            lines.append(
                f"{ds_label} & {regime} & {n} & {_fmt(cult_mean)} & {_fmt(rand_mean)} & {_fmt(diff)} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    p = TABLES_OUT / "tab_mask_summary_by_regime.tex"
    p.write_text("\n".join(lines))
    return p


def write_significance(
    df: pd.DataFrame, paired: dict, regime_tests: dict, size_corr: dict
) -> Path:
    lines = [
        "% Auto-generated by analyze_cultural_neurons.py.",
        r"\begin{tabular}{l l rr l}",
        r"\toprule",
        r"Test & Dataset & Stat & $p$ & Interpretation \\",
        r"\midrule",
    ]
    for ds in ("blend_mcq", "belebele", "countryrc_localized"):
        r = paired[ds]
        ds_label = {
            "blend_mcq": "BLEnD",
            "belebele": "Belebele",
            "countryrc_localized": "CountryRC",
        }[ds]
        w = r["wilcoxon_cult_vs_random"]
        sign = ">" if r["cultural_mean_drop"] > r["random_mean_drop"] else "$\\le$"
        lines.append(
            f"Wilcoxon cult.\\ vs.\\ random & {ds_label} & "
            f"{_fmt(w['stat'], p=1)} & {_fmt(w['p'])} & "
            f"$\\Delta_{{\\mathrm{{cult}}}}$ {sign} $\\Delta_{{\\mathrm{{rand}}}}$ ($n=${r['n']}) \\\\"
        )
    lines.append(r"\midrule")
    for ds in ("blend_mcq", "belebele", "countryrc_localized"):
        ds_label = {
            "blend_mcq": "BLEnD",
            "belebele": "Belebele",
            "countryrc_localized": "CountryRC",
        }[ds]
        rs = regime_tests[ds]["pairs"]
        for pair_key, label_short in [
            ("monolingual_vs_bilingual", "mono vs bi"),
            ("monolingual_vs_multilingual", "mono vs multi"),
            ("bilingual_vs_multilingual", "bi vs multi"),
        ]:
            t = rs[pair_key]
            lines.append(
                f"Mann-Whitney {label_short} & {ds_label} & "
                f"{_fmt(t['U'], p=1)} & {_fmt(t['p'])} & "
                f"regime difference in cultural drop \\\\"
            )
        lines.append(r"\midrule")
    lines.append(
        f"Spearman $\\log_{{10}}(\\text{{param\\_M}})$ vs cultural drop & BLEnD & "
        f"{_fmt(size_corr['spearman_r'])} & {_fmt(size_corr['spearman_p'])} & "
        f"scale effect ($n=${size_corr['n']}) \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    p = TABLES_OUT / "tab_significance_tests.tex"
    p.write_text("\n".join(lines))
    return p


def _tex_safe(s: str) -> str:
    """Escape underscores in model slugs for LaTeX text mode."""
    return s.replace("_", r"\_")


def write_layer_concentration(layer_conc: pd.DataFrame) -> Path:
    lines = [
        "% Auto-generated by analyze_cultural_neurons.py.",
        r"\begin{tabular}{l l rr l}",
        r"\toprule",
        r"Model & Locale & $L$ & Top-3 share (\%) & Top-3 layers \\",
        r"\midrule",
    ]
    last_regime = None
    for _, r in layer_conc.iterrows():
        if r["regime"] != last_regime:
            if last_regime is not None:
                lines.append(r"\midrule")
            regime_label = str(r["regime"]).capitalize()
            lines.append(
                rf"\multicolumn{{5}}{{l}}{{\textit{{{regime_label}}}}} \\"
            )
        last_regime = r["regime"]
        lines.append(
            f"\\texttt{{{_tex_safe(r['slug'])}}} & {r['locale']} & {r['n_layers']} & "
            f"{r['top3_share']*100:.1f} & {r['top3_layers']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    p = TABLES_OUT / "tab_layer_concentration.tex"
    p.write_text("\n".join(lines))
    return p


def write_overlap_summary(overlap_df: pd.DataFrame) -> Path:
    lines = [
        "% Auto-generated by analyze_cultural_neurons.py.",
        r"\begin{tabular}{l r rrr}",
        r"\toprule",
        r"Model & $L$ & Mean Jaccard & Min Jaccard & Max Jaccard \\",
        r"\midrule",
    ]
    for _, r in overlap_df.iterrows():
        lines.append(
            f"\\texttt{{{_tex_safe(r['slug'])}}} & {int(r['n_locales'])} & "
            f"{_fmt(r['mean_pairwise_jaccard'])} & "
            f"{_fmt(r['min_pairwise_jaccard'])} & "
            f"{_fmt(r['max_pairwise_jaccard'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    p = TABLES_OUT / "tab_overlap_summary.tex"
    p.write_text("\n".join(lines))
    return p


def write_markdown_report(
    df, paired, regime_tests, size_corr, overlap_df, layer_conc, zdf
) -> Path:
    md = []
    n_cells = df.groupby(["slug", "locale"]).ngroups
    n_models = df["slug"].nunique()
    md.append(
        f"# Cultural-neuron analysis — {n_models} models, {n_cells} (model, locale) tuples\n"
    )
    md.append("## Headline numbers\n")
    for ds in ("blend_mcq", "belebele", "countryrc_localized"):
        r = paired[ds]
        md.append(
            f"- **{ds}**: cultural drop mean = {r['cultural_mean_drop']:.4f}, "
            f"random drop mean = {r['random_mean_drop']:.4f}, "
            f"Wilcoxon p = {r['wilcoxon_cult_vs_random']['p']:.4g} (n={r['n']})"
        )
    md.append("\n## Per-regime descriptives (BLEnD)")
    desc = regime_tests["blend_mcq"]["descriptives"]
    for regime, d in desc.items():
        md.append(
            f"- {regime}: n={d['n']}, mean cult drop = {d['mean']:.4f} ± {d['std']:.4f}, median = {d['median']:.4f}"
        )
    md.append("\n## Cross-locale Jaccard overlap (multilingual models)")
    for _, r in overlap_df.iterrows():
        md.append(
            f"- {r['slug']} ({int(r['n_locales'])} locales): "
            f"mean pairwise Jaccard = {r['mean_pairwise_jaccard']:.3f}, "
            f"range [{r['min_pairwise_jaccard']:.3f}, {r['max_pairwise_jaccard']:.3f}]"
        )
    md.append("\n## Size scaling (Spearman)")
    md.append(
        f"- log10(param_M) vs cultural_drop (BLEnD): "
        f"ρ = {size_corr['spearman_r']:.3f}, p = {size_corr['spearman_p']:.4g} (n={size_corr['n']})"
    )
    md.append("\n## Per-row reliability (z-score vs random null)")
    n_sig = (zdf["p_vs_random"] < 0.05).sum()
    md.append(
        f"- {n_sig}/{len(zdf)} (model, locale, dataset) rows reject the random null at α=0.05 (two-tailed z)"
    )
    p = OUT / "cultural_neuron_analysis.md"
    p.write_text("\n".join(md))
    return p


def main() -> int:
    df = load_mask_compare()
    print(
        f"loaded {len(df)} rows ({df['slug'].nunique()} models, "
        f"{df.groupby(['slug', 'locale']).ngroups} (model, locale) pairs)"
    )

    zdf = per_row_zscore(df)
    zdf.to_csv(OUT / "cultural_neuron_per_row_zscore.csv", index=False)

    paired = overall_paired_tests(df)
    regime_tests = {
        ds: regime_significance(df, ds)
        for ds in ("blend_mcq", "belebele", "countryrc_localized")
    }
    size_corr = size_correlation(df)

    layer_df = collect_layer_distribution(df)
    layer_df.to_csv(OUT / "cultural_neuron_layer_distribution.csv", index=False)
    layer_conc = layer_concentration(layer_df)
    layer_conc.to_csv(OUT / "cultural_neuron_layer_concentration.csv", index=False)

    overlap_df = collect_overlap_summary(df)
    overlap_df.to_csv(OUT / "cultural_neuron_overlap_summary.csv", index=False)

    write_summary_by_regime(df, regime_tests)
    write_significance(df, paired, regime_tests, size_corr)
    write_layer_concentration(layer_conc)
    write_overlap_summary(overlap_df)
    write_markdown_report(
        df, paired, regime_tests, size_corr, overlap_df, layer_conc, zdf
    )

    # Drop the full analysis bundle as a JSON for reproducibility
    bundle = {
        "paired_tests": paired,
        "regime_tests": regime_tests,
        "size_correlation": size_corr,
        "n_significant_rows_alpha_0.05": int((zdf["p_vs_random"] < 0.05).sum()),
        "total_rows": int(len(zdf)),
    }
    (OUT / "cultural_neuron_analysis.json").write_text(json.dumps(bundle, indent=2))
    print("[done] tables + analysis written under diss/outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
