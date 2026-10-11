# Saved results

The included experiments cover 30 behavioral model–locale pairs, 36 primary masking pairs, and 21 checkpoints across Korean, Arabic, and French HPLT models. [RESULTS_COHORT.json](RESULTS_COHORT.json) lists the exact selections.

The 4,005 result files contain per-example predictions, attribution scores, selected neurons, random/same-layer controls, and numerical summaries. Shared summary and attribution files are preserved whole. [RESULTS_MANIFEST.csv](RESULTS_MANIFEST.csv) records their paths, sizes, hashes, and uses.

JSON/CSV outputs larger than 1 MiB are gzip-compressed. Restore their original paths from the repository root:

```sh
find diss/outputs -name '*.gz' -exec gzip -dk {} +
```

Allow approximately 4 GB of additional disk space. Small summaries can be opened directly.

## Analysis code

All analysis scripts are under `diss/analysis/`:

| Analysis | Script |
| --- | --- |
| Causal effects and source-question bootstrap | `causal_effects.py` |
| Checkpoint effects and intervals | `checkpoint_effects.py` |
| Scoring sensitivity and overlap controls | `scoring_robustness.py` |
| Checkpoint trajectories | `checkpoint_trajectory_plot.py` |
| Effects by model size | `plot_masking_moderation.py` |
| Behavioral comparisons | `tradeoff_plots.py`, `rq1_per_model_table.py` |
| Neuron-layer concentration and locale overlap | `analyze_cultural_neurons.py`, `cultural_neuron_locale_overlap_plot.py` |
| Answer-option lengths | `option_length_stats.py` |

For example, `python3 diss/analysis/scoring_robustness.py` reads saved results and writes its analysis under `diss/outputs/analysis/`. Dependencies include NumPy, pandas, SciPy, Matplotlib, and PyYAML. Some legacy scripts also load tokenizer or model metadata. The Japanese option-length plot can use the saved `option_length_raw.csv` rows for CountryRC/ja.

Raw datasets, construction code, prompts, and translation caches are documented in [diss/data/README.md](diss/data/README.md). The saved results have not been recomputed; the current code is not a frozen software environment for every historical experiment.
