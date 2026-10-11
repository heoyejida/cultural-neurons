from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from diss.core.text import build_continuation_context
from diss.core.types import PreparedExample

LOGGER = logging.getLogger(__name__)

from diss.core.locale_meta import (
    COUNTRY_TO_LOCALE,
    LOCALE_TO_COUNTRY,
)

# CULNIG (Yamamoto et al. 2025) topic split: neuron-identification vs held-out eval.
# Exhaustive over the six eBLEnD topics.
NEURON_TOPICS = {"Food", "Sport", "Work life"}
NON_NEURON_TOPICS = {
    "Education",
    "Family",
    "Holidays/Celebration/Leisure",
}

_EBLND_CSV = "eBLEnD/eblend_final.csv"


def _filter_by_target_data(df: pd.DataFrame, target_data: str) -> pd.DataFrame:
    if target_data == "all":
        return df
    if "topic" not in df.columns:
        raise ValueError(
            "Extended BLEnD CSV is missing the 'topic' column required for "
            "target_data filtering. Use eblend_final.csv."
        )
    if target_data == "neuron":
        return df[df["topic"].isin(NEURON_TOPICS)]
    if target_data == "non_neuron":
        return df[df["topic"].isin(NON_NEURON_TOPICS)]
    raise ValueError(f"Unsupported target_data: {target_data!r}")


def load_blend_examples(
    *,
    target_countries: list[str],
    data_root: Path,
    target_data: str,
) -> list[PreparedExample]:
    csv_path = data_root / _EBLND_CSV
    df = pd.read_csv(csv_path, encoding="utf-8")

    target_langs = {
        COUNTRY_TO_LOCALE[c] for c in target_countries if c in COUNTRY_TO_LOCALE
    }
    df = df[df["lang"].isin(target_langs)]
    df = _filter_by_target_data(df, target_data)

    examples: list[PreparedExample] = []
    for _, row in df.iterrows():
        locale = str(row["lang"])
        country = LOCALE_TO_COUNTRY.get(locale, locale)
        question = str(row["question"])
        choices = {
            "A": str(row["option A"]),
            "B": str(row["option B"]),
            "C": str(row["option C"]),
            "D": str(row["option D"]),
        }
        gold_label = str(row["answer"])
        example_id = f"eblnd-{row['id']}"

        examples.append(
            PreparedExample(
                example_id=example_id,
                country=country,
                locale=locale,
                labels=["A", "B", "C", "D"],
                gold_label=gold_label,
                prompt_prefix=build_continuation_context(question, locale),
                option_text_by_label=choices,
                metadata={"_topic": str(row.get("topic", ""))},
            )
        )
    return examples
