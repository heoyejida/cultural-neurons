"""RQ1 per-model behavioural tables (body): sub-1B and >=1B matched triples.

Two regime-matched comparisons, one model per regime per locale, each with a
regime-mean block:

  sub-1B : Goldfish (mono) / xx-en intermediate (bi) / XGLM-564M (multi), 6 locales.
  >=1B   : HPLT-2.0 (mono) / per-language bilingual / Qwen3 (multi, param-matched:
           4B for Japanese, 1.7B elsewhere), for ko/ja/fr/ar only. Chinese and
           Spanish are dropped because they have no >=1B bilingual model. Japanese
           has no HPLT checkpoint, so its monolingual cell is rinna-monolingual.

Each cell is row-instance accuracy on Extended BLEnD (cultural) and Belebele
(reading). The best model value within each locale is bolded, and the best
regime mean is bolded in the mean block. Parameter counts come from HF
safetensors metadata, with a nominal fallback for non-safetensors repos.

Writes diss/outputs/tables/tab_rq1_per_model_{sub1b,ge1b}.tex.

CPU-only. Run on flamingo:
  cd /local/scratch/yh614 && conda activate diss
  python -m diss.analysis.rq1_per_model_table
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUMMARY = Path("diss/outputs/behavior_summary.csv")
TABLE_DIR = Path("diss/outputs/tables")

LOCALE_NAME = {
    "ko": "ko-KR",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "fr": "fr-FR",
    "es": "es-ES",
    "ar": "ar-SA",
}
REGIME_ORDER = {"monolingual": 0, "bilingual": 1, "multilingual": 2}

SUB1B_ORDER = ["ko", "ja", "zh", "fr", "es", "ar"]
GE1B_ORDER = ["ko", "ja", "fr", "ar"]  # zh, es have no >=1B bilingual model

# sub-1B core triple: one label per regime per locale.
SUB1B_LABELS = (
    {f"goldfish-{l}" for l in SUB1B_ORDER}
    | {f"{l}-en-intermediate" for l in SUB1B_ORDER}
    | {"xglm-564m"}
)

# >=1B curated selection: explicit (locale -> regime -> [labels]).
GE1B_SELECT = {
    "ko": {
        "monolingual": ["hplt2c-kor"],
        "bilingual": ["kanana-1p5-2p1b-base"],
        "multilingual": ["qwen3-1p7b-base"],
    },
    "ja": {
        "monolingual": ["rinna-monolingual-neox-3p6b"],
        "bilingual": ["rinna-bilingual-neox-4b"],
        "multilingual": ["qwen3-4b-base"],
    },
    "fr": {
        "monolingual": ["hplt2c-fra"],
        "bilingual": ["croissantllm-base"],
        "multilingual": ["qwen3-1p7b-base"],
    },
    "ar": {
        "monolingual": ["hplt2c-ara"],
        "bilingual": ["jais-family-1p3b"],
        "multilingual": ["qwen3-1p7b-base"],
    },
}

MODEL_DISPLAY = {
    **{f"goldfish-{l}": f"Goldfish-{l}" for l in SUB1B_ORDER},
    **{f"{l}-en-intermediate": f"{l}-en intermediate" for l in SUB1B_ORDER},
    "xglm-564m": "XGLM",
    "xglm-1p7b": "XGLM",
    "polyglot-ko-1p3b": "polyglot-ko",
    "rinna-monolingual-neox-3p6b": "rinna-monolingual",
    "gpt-fr-cased-base": "GPT-fr",
    "aragpt2-mega": "AraGPT2-mega",
    "hplt2c-kor": "HPLT-2.0",
    "hplt2c-ara": "HPLT-2.0",
    "hplt2c-fra": "HPLT-2.0",
    "hplt2c-spa": "HPLT-2.0",
    "hplt2c-zhos": "HPLT-2.0",
    "kanana-1p5-2p1b-base": "Kanana-1.5",
    "rinna-bilingual-neox-4b": "rinna-bilingual",
    "croissantllm-base": "CroissantLLM",
    "jais-family-1p3b": "Jais",
    "qwen3-0p6b-base": "Qwen3",
    "qwen3-1p7b-base": "Qwen3",
    "qwen3-4b-base": "Qwen3",
    "eurollm-1p7b": "EuroLLM",
    "tiny-aya-base": "tiny-aya",
}

NOMINAL = {
    "goldfish-models/kor_hang_1000mb": 125e6,
    "goldfish-models/spa_latn_1000mb": 125e6,
    "goldfish-models/zho_hans_1000mb": 125e6,
    "goldfish-models/arb_arab_1000mb": 125e6,
    "goldfish-models/jpn_jpan_1000mb": 125e6,
    "goldfish-models/fra_latn_1000mb": 125e6,
    "facebook/xglm-564M": 564e6,
    "facebook/xglm-1.7B": 1.7e9,
    "asi/gpt-fr-cased-base": 1.0e9,
    "croissantllm/CroissantLLMBase": 1.3e9,
    "utter-project/EuroLLM-1.7B": 1.7e9,
    "HPLT/hplt2c_kor_checkpoints": 2.15e9,
    "HPLT/hplt2c_ara_checkpoints": 2.15e9,
    "HPLT/hplt2c_fra_checkpoints": 2.15e9,
    "HPLT/hplt2c_zhos_checkpoints": 2.15e9,
    "HPLT/hplt2c_spa_checkpoints": 2.15e9,
    # Curated table models — nominal fallback so a flaky HF metadata fetch never
    # blanks a Params cell (values match the safetensors-derived counts).
    "inceptionai/jais-family-1p3b": 1.6e9,
    "kakaocorp/kanana-1.5-2.1b-base": 2.1e9,
    "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b": 3.8e9,
    "rinna/bilingual-gpt-neox-4b": 3.9e9,
    "Qwen/Qwen3-1.7B-Base": 1.7e9,
    "Qwen/Qwen3-4B-Base": 4.0e9,
    "yejiheo/ko-en-intermediate": 277e6,
    "yejiheo/ja-en-intermediate": 277e6,
    "RA-ALTA/es-en-intermediate": 277e6,
    "RA-ALTA/zh-en-intermediate": 277e6,
    "RA-ALTA/ar-en-intermediate": 277e6,
    "RA-ALTA/fr-en-intermediate": 277e6,
}

HEADER = r"Locale & Regime & Model & Params & BLEnD & Belebele \\"


def param_count(repo: str) -> float | None:
    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(repo)
        return float(sum(meta.parameter_count.values()))
    except Exception:  # noqa: BLE001
        return NOMINAL.get(repo)


def fmt_params(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "--"
    return f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.1f}B"


def emit_table(df: pd.DataFrame, order: list[str], label: str) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\refstepcounter{table}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        HEADER,
        r"\midrule",
    ]
    first = True
    for loc in order:
        sub = df[df["locale"] == loc].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["regime_ord", "params"])
        best_b, best_r = sub["blend"].max(), sub["belebele"].max()
        if not first:
            lines.append(r"\midrule")
        first = False
        for _, r in sub.iterrows():
            b = (
                f"\\textbf{{{r['blend']:.3f}}}"
                if r["blend"] == best_b
                else f"{r['blend']:.3f}"
            )
            v = (
                f"\\textbf{{{r['belebele']:.3f}}}"
                if r["belebele"] == best_r
                else f"{r['belebele']:.3f}"
            )
            lines.append(
                f"{LOCALE_NAME[loc]} & {r['regime']} & {r['model_disp']} & "
                f"{r['params_str']} & {b} & {v} \\\\"
            )
    # Regime-mean block: average per (locale, regime) cell first, then across
    # locales, so a regime with more models in one locale does not over-weight
    # that locale. Matches the Friedman cell-means in regime_tradeoff_stats.
    cells = (
        df.groupby(["locale", "regime"])
        .agg(blend=("blend", "mean"), belebele=("belebele", "mean"))
        .reset_index()
    )
    means = (
        cells.groupby("regime")
        .agg(blend=("blend", "mean"), belebele=("belebele", "mean"))
        .reset_index()
    )
    means["regime_ord"] = means["regime"].map(REGIME_ORDER)
    means = means.sort_values("regime_ord")
    mb, mr = means["blend"].max(), means["belebele"].max()
    lines.append(r"\midrule")
    for i, (_, r) in enumerate(means.iterrows()):
        loc_cell = "Mean" if i == 0 else ""
        b = f"\\textbf{{{r['blend']:.3f}}}" if r["blend"] == mb else f"{r['blend']:.3f}"
        v = (
            f"\\textbf{{{r['belebele']:.3f}}}"
            if r["belebele"] == mr
            else f"{r['belebele']:.3f}"
        )
        lines.append(f"{loc_cell} & {r['regime']} & & & {b} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(SUMMARY).rename(
        columns={"cultural_accuracy": "blend", "reasoning_accuracy": "belebele"}
    )

    pmap = {repo: param_count(repo) for repo in df["model_name"].unique()}
    df["params"] = df["model_name"].map(pmap)
    df["params_str"] = df["params"].map(fmt_params)
    df["model_disp"] = df["label"].map(lambda l: MODEL_DISPLAY.get(l, l))
    df["regime_ord"] = df["regime"].map(REGIME_ORDER)

    sub1b = df[df["label"].isin(SUB1B_LABELS) & df["locale"].isin(SUB1B_ORDER)].copy()

    keep = []
    for loc, by_regime in GE1B_SELECT.items():
        for lbls in by_regime.values():
            for lbl in lbls:
                keep.append((loc, lbl))
    ge1b = df[df.apply(lambda r: (r["locale"], r["label"]) in keep, axis=1)].copy()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "tab_rq1_per_model_sub1b.tex").write_text(
        emit_table(sub1b, SUB1B_ORDER, "tab:regime_sub1b")
    )
    (TABLE_DIR / "tab_rq1_per_model_ge1b.tex").write_text(
        emit_table(ge1b, GE1B_ORDER, "tab:regime_ge1b")
    )

    print(f"sub-1B rows: {len(sub1b)} (locales {sorted(sub1b['locale'].unique())})")
    print(f">=1B rows: {len(ge1b)} (locales {sorted(ge1b['locale'].unique())})")
    print(
        "\n===== sub1b =====\n"
        + (TABLE_DIR / "tab_rq1_per_model_sub1b.tex").read_text()
    )
    print("===== ge1b =====\n" + (TABLE_DIR / "tab_rq1_per_model_ge1b.tex").read_text())


if __name__ == "__main__":
    main()
