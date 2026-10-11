"""Attach original-BLEnD Topic/Source to extended-BLEnD MCQs by matching
country-masked question templates; join the per-id MCQ answer key.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
BLEND_QUESTIONS = BASE.parent / "BLEnD" / "questions" / "South_Korea_questions.csv"
EBLEND_INPUT = BASE / "track_2_mcq_input-2.tsv"
MCQ_ANSWERS = BASE / "final_2_mcq_reference.tsv"
OUTPUT = BASE / "eblend_topic.csv"

# Surface forms masked on both BLEnD and extended-BLEnD sides; order parallels
# TARGET_LOCALES (ko-KR, es-ES, zh-CN, ar-SA, en-US, ja-JP, fr-FR).
COUNTRY_NAMES = [
    "South Korea",
    "Spain",
    "China",
    "Saudi Arabia",
    "the US",
    "Japan",
    "France",
]
TARGET_LOCALES = ["ko-KR", "es-ES", "zh-CN", "ar-SA", "en-US", "ja-JP", "fr-FR"]
OPTION_COLUMNS = ["option A", "option B", "option C", "option D"]

# Cap MCQ variants per (locale, org_id) so high-variant questions don't
# dominate aggregates; see culnig_gate_proj_split_decisions.md §6.
VARIANTS_CAP = 5

_COUNTRY_RE = re.compile("|".join(re.escape(name) for name in COUNTRY_NAMES))
_PLACEHOLDER = "######"


def mask_country(text: str) -> str:
    return _COUNTRY_RE.sub(_PLACEHOLDER, text)


def load_topic_templates() -> pd.DataFrame:
    """question_template -> (ID, Topic, Source) lookup from BLEnD.

    One country file suffices: templates are identical across all 16 countries.
    Dedup on question_template keeps the downstream merge many-to-one.
    """
    df = pd.read_csv(BLEND_QUESTIONS, index_col=0)
    df["question_template"] = df["Translation"].map(mask_country)
    df = df.drop_duplicates(subset="question_template", keep="first")
    return df[["ID", "question_template", "Topic", "Source"]]


def main() -> None:
    templates = load_topic_templates()

    eblend = pd.read_csv(EBLEND_INPUT, sep="\t")
    locale = eblend["id"].str.split("_").str[0]
    eblend.insert(0, "lang", locale.str.split("-").str[0])
    eblend.insert(1, "locale", locale)
    eblend = eblend[eblend["locale"].isin(TARGET_LOCALES)].copy()

    # Drop all-empty-option rows (upstream eBLEnD defect).
    all_blank = (
        eblend[OPTION_COLUMNS]
        .fillna("")
        .apply(lambda c: c.str.strip())
        .eq("")
        .all(axis=1)
    )
    if all_blank.any():
        print(
            f"dropped {int(all_blank.sum())} row(s) with all-empty options: "
            f"{', '.join(eblend.loc[all_blank, 'id'])}"
        )
    eblend = eblend[~all_blank].copy()

    eblend["question_template"] = eblend["question"].map(mask_country)

    merged = eblend.merge(templates, on="question_template", how="left")
    matched = merged[merged["Topic"].notna()].copy()
    matched = matched.rename(
        columns={"ID": "org_id", "Topic": "topic", "Source": "source"}
    )

    answers = pd.read_csv(MCQ_ANSWERS, sep="\t")[["id", "answer"]]
    matched = matched.merge(answers, on="id", how="left")
    missing_answer = int(matched["answer"].isna().sum())

    n_topic_matched = len(matched)

    # Drop rows whose 4-option multiset duplicates another of the same
    # (locale, org_id): reorderings of one distractor set, no new signal.
    matched = matched.sort_values(["locale", "org_id", "id"], kind="mergesort")
    combo = matched[OPTION_COLUMNS].apply(
        lambda r: tuple(sorted(str(x) for x in r)), axis=1
    )
    dup_mask = combo.groupby([matched["locale"], matched["org_id"]]).transform(
        lambda s: s.duplicated(keep="first")
    )
    n_dedup_dropped = int(dup_mask.sum())
    matched = matched[~dup_mask].copy()

    rank = matched.groupby(["locale", "org_id"]).cumcount()
    n_pre_cap = len(matched)
    matched = matched[rank < VARIANTS_CAP].copy()
    n_cap_dropped = n_pre_cap - len(matched)

    print(f"extended-BLEnD rows (target locales): {len(eblend)}")
    print(f"  unique question templates : {eblend['question_template'].nunique()}")
    print(f"  topic matched             : {n_topic_matched}")
    print(f"  unmatched (dropped)       : {len(merged) - n_topic_matched}")
    print(f"  missing answer key        : {missing_answer}")
    print(f"  distractor-dedup dropped  : {n_dedup_dropped}")
    print(f"  variant-cap{VARIANTS_CAP} dropped       : {n_cap_dropped}")
    print(f"  retained                  : {len(matched)}")
    print("per-locale retained counts:")
    print(matched["locale"].value_counts().sort_values(ascending=False).to_string())

    matched.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} ({len(matched)} rows)")


if __name__ == "__main__":
    main()
