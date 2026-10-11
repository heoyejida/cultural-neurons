from __future__ import annotations

from pathlib import Path

from diss.core.types import PreparedExample
from diss.prep.blend import load_blend_examples
from diss.prep.countryrc import load_countryrc_examples
from diss.prep.belebele import load_belebele_examples


def load_examples(
    *,
    dataset_name: str,
    target_countries: list[str] | None = None,
    target_locales: list[str] | None = None,
    data_root: Path,
    target_data: str = "all",
) -> list[PreparedExample]:
    if dataset_name == "blend_mcq":
        return load_blend_examples(
            target_countries=target_countries or [],
            data_root=data_root,
            target_data=target_data,
        )
    if dataset_name == "countryrc_localized":
        return load_countryrc_examples(
            target_locales=target_locales or [],
            target_data=target_data,
        )
    if dataset_name == "belebele":
        return load_belebele_examples(target_locales=target_locales or [])
    raise ValueError(f"Unsupported dataset: {dataset_name}")
