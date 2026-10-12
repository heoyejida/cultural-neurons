"""Option-length evidence for the byte-normalised continuation scoring choice.

For Extended BLEnD, Belebele, and CountryRC, measure the byte length and the
token length of each answer option (gold vs distractor) across the seven
locales. The token length uses one representative multilingual tokeniser
(facebook/xglm-564M) because token counts are tokeniser-specific, whereas byte
length is model-independent. The Japanese CountryRC case is where byte-length
normalisation is expected to strain (Kanji/Katakana country names).

Outputs (under diss/outputs/):
  - option_length_raw.csv        one row per option (dataset, locale, role, lengths)
  - option_length_summary.csv    per (dataset, role) and per (dataset, locale)
  - figures/option_byte_length_by_locale.png   3-panel boxplot of option byte length
  - figures/japanese_countryrc_byte_length.png Japanese CountryRC gold/distractor plot

CPU-only. Run on flamingo:
  cd /local/scratch/yh614 && conda activate diss
  python -m diss.analysis.option_length_stats
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from diss.core.locale_meta import ALL_TARGET_LOCALES, LOCALE_TO_COUNTRY
from diss.prep.belebele import load_belebele_examples
from diss.prep.blend import _EBLND_CSV
from diss.prep.countryrc import load_countryrc_examples

OUT = Path("diss/outputs")
EBLEND_CSV = Path("diss/data") / _EBLND_CSV
TOKENIZER_NAME = "facebook/xglm-564M"

# Presentation order: Latin-script locales grouped, then CJK, then Arabic, then
# Japanese last so the byte-length outlier reads cleanly left-to-right.
LOCALE_ORDER = ["en", "es", "fr", "ko", "zh", "ar", "ja"]
DATASETS = ["BLEnD", "Belebele", "CountryRC"]


def _records_from_examples(examples, dataset: str) -> list[dict]:
    rows = []
    for ex in examples:
        gold = ex.gold_label
        for label, text in ex.option_text_by_label.items():
            t = str(text)
            rows.append(
                {
                    "dataset": dataset,
                    "locale": ex.locale,
                    "role": "gold" if label == gold else "distractor",
                    "byte_len": len(t.encode("utf-8")),
                    "char_len": len(t),
                    "text": t,
                }
            )
    return rows


def _blend_records() -> list[dict]:
    df = pd.read_csv(EBLEND_CSV, encoding="utf-8")
    rows = []
    for _, r in df.iterrows():
        gold = str(r["answer"]).strip()
        for label in ["A", "B", "C", "D"]:
            t = str(r[f"option {label}"])
            rows.append(
                {
                    "dataset": "BLEnD",
                    "locale": str(r["lang"]),
                    "role": "gold" if label == gold else "distractor",
                    "byte_len": len(t.encode("utf-8")),
                    "char_len": len(t),
                    "text": t,
                }
            )
    return rows


def main() -> None:
    locales = list(ALL_TARGET_LOCALES)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    rows: list[dict] = []
    rows += _blend_records()
    rows += _records_from_examples(
        load_belebele_examples(target_locales=locales), "Belebele"
    )
    rows += _records_from_examples(
        load_countryrc_examples(target_locales=locales), "CountryRC"
    )

    df = pd.DataFrame(rows)
    # token length under the representative tokeniser (batch for speed)
    df["tok_len"] = [
        len(ids)
        for ids in tok(df["text"].tolist(), add_special_tokens=False)["input_ids"]
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["text"]).to_csv(OUT / "option_length_raw.csv", index=False)

    def agg(g):
        return pd.Series(
            {
                "n": len(g),
                "byte_mean": g["byte_len"].mean(),
                "byte_std": g["byte_len"].std(),
                "tok_mean": g["tok_len"].mean(),
                "tok_std": g["tok_len"].std(),
            }
        )

    print("=== PER (dataset, role): byte vs token length ===")
    by_role = df.groupby(["dataset", "role"]).apply(agg)
    print(by_role.to_string())

    print("\n=== PER (dataset, locale): byte vs token length (all options) ===")
    by_loc = df.groupby(["dataset", "locale"]).apply(agg)
    by_loc["tok_per_byte"] = by_loc["tok_mean"] / by_loc["byte_mean"]
    print(by_loc.to_string())

    print("\n=== PER (dataset, locale, role): byte length (gold vs distractor gap) ===")
    by_lr = df.groupby(["dataset", "locale", "role"])["byte_len"].mean().unstack("role")
    by_lr["gold_minus_distr"] = by_lr["gold"] - by_lr["distractor"]
    print(by_lr.to_string())

    summary = pd.concat({"by_role": by_role, "by_locale": by_loc}, names=["view"])
    summary.to_csv(OUT / "option_length_summary.csv")
    by_lr.to_csv(OUT / "option_length_by_locale_role.csv")

    # --- figure: 3-panel boxplot of option byte length, gold vs distractor ---
    order = [l for l in LOCALE_ORDER if l in df["locale"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        positions, gold_data, distr_data = [], [], []
        for i, l in enumerate(order):
            gold_data.append(
                sub[(sub["locale"] == l) & (sub["role"] == "gold")]["byte_len"].values
            )
            distr_data.append(
                sub[(sub["locale"] == l) & (sub["role"] == "distractor")][
                    "byte_len"
                ].values
            )
            positions.append(i)
        w = 0.34
        bg = ax.boxplot(
            gold_data,
            positions=[p - w / 1.6 for p in positions],
            widths=w,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        bd = ax.boxplot(
            distr_data,
            positions=[p + w / 1.6 for p in positions],
            widths=w,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        for patch in bg["boxes"]:
            patch.set_facecolor("#4393c3")
        for patch in bd["boxes"]:
            patch.set_facecolor("#f4a582")
        ax.set_xticks(positions)
        ax.set_xticklabels(order)
        ax.set_title(ds)
        ax.set_xlabel("locale")
        if ax is axes[0]:
            ax.set_ylabel("answer option byte length")
        ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#4393c3"),
        plt.Rectangle((0, 0), 1, 1, fc="#f4a582"),
    ]
    fig.legend(handles, ["gold", "distractor"], loc="upper right", ncol=2)
    fig.suptitle("Answer-option byte length by locale, gold versus distractor")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "option_byte_length_by_locale.png", dpi=200, bbox_inches="tight")

    # --- focused appendix figure: Japanese CountryRC gold vs distractor bytes ---
    jp = df[(df["dataset"] == "CountryRC") & (df["locale"] == "ja")]
    gold_b = jp[jp["role"] == "gold"]["byte_len"].values
    distr_b = jp[jp["role"] == "distractor"]["byte_len"].values
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.0))
    bp = ax2.boxplot(
        [gold_b, distr_b],
        showfliers=True,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black"),
    )
    bp["boxes"][0].set_facecolor("#4393c3")
    bp["boxes"][1].set_facecolor("#f4a582")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["gold\n(Japan, Kanji)", "distractors\n(other countries)"])
    ax2.set_ylabel("Byte length")
    ax2.axhline(6, color="#4393c3", ls="--", lw=1, alpha=0.7)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(
        OUT / "figures" / "japanese_countryrc_byte_length.png", dpi=200, bbox_inches="tight"
    )
    print(f"\nwrote {OUT/'figures'/'option_byte_length_by_locale.png'}")
    print(f"wrote {OUT/'figures'/'japanese_countryrc_byte_length.png'}")
    print(
        f"  ja CountryRC gold bytes: mean={gold_b.mean():.1f} std={gold_b.std():.2f} "
        f"min={gold_b.min()} max={gold_b.max()}"
    )
    print(
        f"  ja CountryRC distractor bytes: mean={distr_b.mean():.1f} std={distr_b.std():.2f} "
        f"min={distr_b.min()} max={distr_b.max()}"
    )
    print(f"wrote {OUT/'option_length_raw.csv'} ({len(df)} options)")


if __name__ == "__main__":
    main()
