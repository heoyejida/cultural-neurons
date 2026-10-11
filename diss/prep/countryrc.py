from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from diss.core.text import build_continuation_context
from diss.core.types import PreparedExample

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "CountryRC"
    / "data_multilingual.json"
)


from diss.core.locale_meta import (  # noqa: F401  (re-export)
    ALL_TARGET_COUNTRIES,
    COUNTRY_NAMES,
    COUNTRY_TO_LOCALE,
    LOCALE_TO_COUNTRY,
)


def _load_templates() -> dict[str, list[tuple[str, str]]]:
    """Load (context, question) templates per locale, in file order (id 0..49)."""
    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if [item["id"] for item in items] != [str(i) for i in range(50)]:
        raise ValueError("CountryRC requires unique source IDs 0..49 in order")
    locales = set().union(*(item["translations"] for item in items))
    out: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        missing = locales - set(item["translations"])
        if missing:
            raise ValueError(f"CountryRC source {item['id']} missing locales: {sorted(missing)}")
        for locale, t in item["translations"].items():
            out.setdefault(locale, []).append((t["context"], t["question"]))
    return out


TEMPLATES = _load_templates()


def _seed_for(country: str, index: int) -> int:
    return int(hashlib.md5(f"{country}|{index}".encode()).hexdigest()[:8], 16)


def _localized_name(locale: str, country: str) -> str:
    return COUNTRY_NAMES[locale].get(country, country)


def _stable_shuffle(items: list, seed: int) -> list:
    rng = random.Random(seed)
    cloned = list(items)
    rng.shuffle(cloned)
    return cloned


def load_countryrc_examples(
    *,
    target_locales: list[str],
    target_data: str = "all",
    count_per_country: int = 50,
) -> list[PreparedExample]:
    """Per locale, generate ``count_per_country`` examples with gold = the locale's
    LOCALE_TO_COUNTRY country and 3 distractors from the other six target countries.

    The (locale, country) pin matches BLEnD's 1:1 culture stimuli, so the CountryRC
    filter excludes neurons firing on the same (language, country) setting BLEnD uses.
    Default 50 examples (split 25/25 for neuron/non_neuron).
    """
    # Guard direct internal callers (CLI choices= covers the entry path); a typo here
    # would silently mix the train/eval halves.
    if target_data not in {"all", "neuron", "non_neuron"}:
        raise ValueError(
            f"countryrc target_data must be one of {{'all','neuron','non_neuron'}}; "
            f"got {target_data!r}"
        )
    if count_per_country < 0:
        raise ValueError("count_per_country must be nonnegative")

    examples: list[PreparedExample] = []
    for locale in target_locales:
        if locale not in TEMPLATES:
            raise ValueError(f"Missing CountryRC locale: {locale}")
        if count_per_country > len(TEMPLATES[locale]):
            raise ValueError(f"Insufficient CountryRC templates for {locale}: requested {count_per_country}")
        mentioned = LOCALE_TO_COUNTRY.get(locale)
        if mentioned is None:
            continue
        distractor_pool = [c for c in ALL_TARGET_COUNTRIES if c != mentioned]
        for idx in range(count_per_country):
            if target_data == "neuron" and idx >= count_per_country // 2:
                continue
            if target_data == "non_neuron" and idx < count_per_country // 2:
                continue
            seed_key = f"{locale}|{mentioned}"
            rng = random.Random(_seed_for(seed_key, idx))
            tpl_ctx, tpl_q = TEMPLATES[locale][idx]
            distractor_codes = rng.sample(distractor_pool, 3)
            if "{country_dummy}" in tpl_ctx:
                # CULNIG: passage dummy is drawn from the distractors so its mention aligns
                # with a wrong-answer country.
                dummy_code = rng.choice(distractor_codes)
                context = tpl_ctx.format(
                    country=_localized_name(locale, mentioned),
                    country_dummy=_localized_name(locale, dummy_code),
                )
            else:
                context = tpl_ctx.format(country=_localized_name(locale, mentioned))
            choices = [_localized_name(locale, mentioned)] + [
                _localized_name(locale, c) for c in distractor_codes
            ]
            labels = ["A", "B", "C", "D"]
            paired = _stable_shuffle(
                list(zip(labels, choices[:4])),
                _seed_for(seed_key, idx) + 7,
            )
            choice_map = {label: option for label, option in paired}
            gold = next(
                label
                for label, option in paired
                if option == _localized_name(locale, mentioned)
            )
            short = mentioned.lower().replace(" ", "_")
            examples.append(
                PreparedExample(
                    example_id=f"{locale}-{short}-{idx}",
                    country=mentioned,
                    locale=locale,
                    labels=labels,
                    gold_label=gold,
                    prompt_prefix=build_continuation_context(
                        tpl_q,
                        locale,
                        passage=context,
                    ),
                    option_text_by_label=choice_map,
                    metadata={},
                )
            )
    return examples
