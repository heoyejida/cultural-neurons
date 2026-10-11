# Cultural Neurons

Experiments on cultural knowledge in monolingual, bilingual, and multilingual language models. The pipeline evaluates multiple-choice answers, identifies MLP neurons using activation × gradient attribution, and measures accuracy changes when those neurons are masked.

## Code

- `diss/cli.py`, `diss/run.py` — commands and experiment orchestration.
- `diss/core/` — model adapters and shared utilities.
- `diss/prep/` — dataset loaders and attribution/evaluation splits.
- `diss/pipeline/` — evaluation, attribution, neuron selection, masking, and checkpoint tracing.
- `diss/config/` — model manifest and checkpoint settings.

Answer scoring uses byte-normalized negative log-likelihood. Cultural-neuron selection takes the top 1% of MLP candidates by attribution and removes the top 1% from the corresponding CountryRC ranking. Masking is evaluated on held-out BLEnD topics, with random and same-layer-random controls over five seeds by default.

## Data

This shared repository includes:

| File | Purpose |
| --- | --- |
| `diss/data/eBLEnD/eblend_final.csv` | Prepared Extended BLEnD multiple-choice questions |
| `diss/data/CountryRC/data.json` | Source CountryRC templates |
| `diss/data/CountryRC/data_multilingual.json` | Localized CountryRC templates used by the loader |

Belebele is loaded from `facebook/belebele` on Hugging Face. Model weights and tokenizers are loaded from the repositories listed in `diss/config/model_manifest.yaml`; restricted repositories require access credentials.

## Run

Use Python 3.10+ in a compatible GPU environment with PyTorch, Transformers, Accelerate, datasets, NumPy, pandas, and PyYAML. Run commands from the directory containing `diss/`.

For one model:

```bash
python -m diss.cli run-behavior --only-models goldfish-ko --devices 0
python -m diss.cli run-attribution --only-models goldfish-ko --devices 0
python -m diss.cli summarize
```

`--only-models` takes IDs from the manifest. Omit it to run all eligible models. Run behavior evaluation before attribution to provide the unmasked baselines. `summarize` reads available results across the manifest.

Results and summary CSVs are written under `diss/outputs/`. Manifest runners reuse existing artifacts; add `--force` when rerunning after changing experiment settings.

Saved results, analysis/plotting scripts, and dataset-construction scripts are not included in this shared copy.

Checkpoint tracing additionally requires `diss/analysis/checkpoint_trace_summary.py` from the full repository to produce its final summary.
