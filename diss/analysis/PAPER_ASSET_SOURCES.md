# Paper figures and tables

The PNG and TeX files used by the current manuscript are stored in `diss/outputs/figures/` and `diss/outputs/tables/`. The copies are the manuscript assets; analysis scripts write regenerated outputs to those same directories.

## Figures

| File | Generator and input |
| --- | --- |
| `cp_trajectory_hplt2c_rank.png` | [`checkpoint_trajectory_plot.py`](checkpoint_trajectory_plot.py): three HPLT checkpoint summary CSVs, `anchor_rank_percentile_mean`. |
| `cp_trajectory_hplt2c_blend_drop.png` | [`checkpoint_trajectory_plot.py`](checkpoint_trajectory_plot.py): same CSVs, `anchor_blend_drop`. |
| `cp_trajectory_hplt2c_belebele_drop.png` | [`checkpoint_trajectory_plot.py`](checkpoint_trajectory_plot.py): same CSVs, `anchor_belebele_drop`. |
| `fig_cultural_drop_by_param.png` | [`plot_masking_moderation.py`](plot_masking_moderation.py): `masking_significance_per_cell.csv`, grouped by parameter tier. |
| `tradeoff_small.png` | [`tradeoff_plots.py`](tradeoff_plots.py): `behavior_summary.csv`, sub-billion regime. |
| `japanese_countryrc_byte_length.png` | [`option_length_stats.py`](option_length_stats.py): Japanese CountryRC gold-answer and distractor option byte lengths. |
| `cultural_neuron_locale_overlap.png` | [`cultural_neuron_locale_overlap_plot.py`](cultural_neuron_locale_overlap_plot.py): selected-neuron JSON files across multilingual models and locales. |

The checkpoint script's combined mode writes all three checkpoint figures. Figure scripts save their PNG outputs under `diss/outputs/figures/`.

## Tables

| File | Generator and input |
| --- | --- |
| `tab_causal_summary.tex` | [`causal_effects.py`](causal_effects.py): frozen masking results and saved predictions. |
| `tab_scoring_robustness.tex` | [`scoring_robustness.py`](scoring_robustness.py): saved per-cell results and predictions. |
| `tab_checkpoint_trajectories.tex` | [`checkpoint_effects.py`](checkpoint_effects.py): HPLT checkpoint summaries and saved checkpoint results. |
| `tab_pair_slr_gap.tex` | [`causal_effects.py`](causal_effects.py): saved BLEnD predictions and masking results. |
| `tab_rq1_per_model_sub1b.tex`, `tab_rq1_per_model_ge1b.tex` | [`rq1_per_model_table.py`](rq1_per_model_table.py): `behavior_summary.csv` and the script's model selections. |
| `tab_overlap_observed_null.tex` | Manuscript table assembled from overlap statistics; values are summarized by [`scoring_robustness.py`](scoring_robustness.py). |
| `tab_layer_concentration.tex` | [`analyze_cultural_neurons.py`](analyze_cultural_neurons.py): consolidated mask-comparison results and selected-neuron JSON files. |

Table scripts save their TeX outputs under `diss/outputs/tables/`. The checked-in files in these directories preserve the versions included by the manuscript.
