from __future__ import annotations

import logging

from datasets import load_dataset

from diss.core.text import build_continuation_context
from diss.core.types import PreparedExample

LOGGER = logging.getLogger(__name__)

from diss.core.locale_meta import (
    LOCALE_TO_BELEBELE_CONFIG,
    SUPPORTED_LOCALES,
)

_HF_REPO = "facebook/belebele"
_LABEL_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


def load_belebele_examples(
    *,
    target_locales: list[str],
) -> list[PreparedExample]:
    """Load Belebele (FLORES-200 parallel 4-way MCQ) examples as PreparedExamples."""
    examples: list[PreparedExample] = []
    for locale in target_locales:
        if locale not in SUPPORTED_LOCALES:
            LOGGER.warning("Locale %s not supported for Belebele, skipping.", locale)
            continue
        config = LOCALE_TO_BELEBELE_CONFIG[locale]
        LOGGER.info("Loading Belebele config %s for locale %s", config, locale)
        dataset = load_dataset(_HF_REPO, config, split="test")

        for idx, row in enumerate(dataset):
            passage = str(row["flores_passage"]).strip()
            question = str(row["question"]).strip()
            choices = {
                "A": str(row["mc_answer1"]).strip(),
                "B": str(row["mc_answer2"]).strip(),
                "C": str(row["mc_answer3"]).strip(),
                "D": str(row["mc_answer4"]).strip(),
            }
            gold_label = _LABEL_MAP[int(row["correct_answer_num"])]
            example_id = f"belebele-{locale}-{idx}"

            examples.append(
                PreparedExample(
                    example_id=example_id,
                    country=f"Belebele-{locale}",
                    locale=locale,
                    labels=["A", "B", "C", "D"],
                    gold_label=gold_label,
                    prompt_prefix=build_continuation_context(
                        question,
                        locale,
                        passage=passage,
                    ),
                    option_text_by_label=choices,
                    metadata={},
                )
            )
    LOGGER.info(
        "Loaded %d Belebele examples across %d locales.",
        len(examples),
        len(target_locales),
    )
    return examples
