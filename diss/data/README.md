# Data and construction

The raw inputs used by the current dataset construction are included alongside the prepared data. Source files are preserved whole, including rows outside the six paper locales when they belong to a shared upstream release. `DATA_MANIFEST.csv` records SHA-256 hashes and sizes.

| Directory | Contents and construction |
| --- | --- |
| `BLEnD/questions/` | Original South Korea questions, used to recover topic/source labels by matching country-masked templates. |
| `eBLEnD/` | Original MCQ and SAQ inputs and answer references (`track_*.tsv`, `final_*.tsv`), `eblend_topic.csv`, final multilingual `eblend_final.csv`, construction scripts, translation prompt, and six saved translation caches. |
| `CountryRC/` | Original 50-template `data.json`, translated `data_multilingual.json`, translation script/prompt, and six saved translation caches. |
| `Belebele/` | Official raw JSONL for the six paper languages, upstream license files, and retrieval provenance in `SOURCE.json`. |

## Construction code

1. `eBLEnD/eblend_topic.py` joins the original BLEnD topics and MCQ answer keys, removes duplicate option permutations, and caps variants.
2. `eBLEnD/eblend_translate.py` uses the original SAQ questions and answers, translation prompt, and saved cache to build `eblend_final.csv`.
3. `CountryRC/countryrc_translate.py` translates the original templates into `data_multilingual.json` using its prompt and saved cache.
4. `../prep/{blend,countryrc,belebele}.py` prepares evaluation examples and splits.

The translation scripts use the OpenAI API for uncached work and require `OPENAI_API_KEY`. The saved prepared data and caches are included; no translations were rerun for this upload. The historical caches contain translations but do not record the original model revision or full rendered requests.

Belebele was originally loaded from `facebook/belebele` on Hugging Face. No historical local download was available. The included copy was retrieved at the pinned revision in `Belebele/SOURCE.json`; all 900 rows per language were checked against saved result IDs, gold labels/text, and predicted text. This verifies those fields, not the unrecorded historical download revision. The existing experiment loader still uses Hugging Face.

## Upstream sources

- BLEnD: https://github.com/nayeon212/BLEnD
- Extended BLEnD: SemEval-2026 Task 7 release files retained under their original filenames.
- CountryRC: https://huggingface.co/datasets/Taise228/CountryRC
- Belebele: https://huggingface.co/datasets/facebook/belebele and https://github.com/facebookresearch/belebele

The bundled source datasets retain their upstream terms; they are not assigned a new license by this repository.
