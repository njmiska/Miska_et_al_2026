# Figure 2 ephys and optogenetic analysis

This directory is the dependency-closed source snapshot for the manuscript's
SNr-inhibition/downstream bias-selectivity analysis. It replaces the ambiguous
`CD_analysis/` repository location.

## Entrypoints

- `SNr_inhibition_BS_downstream_effect.py`: expensive raw-data analysis and
  sufficient-statistic export.
- `BS_postprocess.py`: population, insertion, multilevel, diagnostic, and
  example-unit plots.
- `BS_config.py`: acquisition-stage options and insertion selection.

The final Figure 2 postprocessing profile is
`configs/figures/figure2_snr_bs_no_glmhmm.json`: laser- and feedback-aligned,
`whole_control_scalar` normalization, `qp_preference` orientation, and the
`all_trials` estimator. Figure 2 does not use GLM-HMM indexing.

`CD_analysis_midbrain.py` is intentionally not included because it is not part
of the frozen Figure 2 workflow. It can be restored later as a separately
documented coding-direction analysis if the manuscript ultimately uses it.

## Snapshot and portability status

The nine scientific Python files were copied from the active working analysis
on 2026-07-27 without changing calculations (trailing whitespace was
normalized). Their repository hashes are in `SOURCE_SNAPSHOT.json`.

This snapshot still exposes several historical absolute paths and expects the
author's IBL/Brainbox/GLM-HMM environment. For that reason the generic
`scripts/reproduce.py` command permits a `--dry-run` configuration preview but
does not yet launch the expensive Figure 2 raw analysis. The next migration
step is a path-only adapter followed by one-PID and full-cohort numerical
regression checks. This explicit guard prevents a command from implying
reproducibility before the scientific output has been verified.

The Supplementary Figure 1 direct-stimulation cohort is normalized in
`metadata_final/metadata_direct_stimulation.py` and included in the release
manifest. Its two runs (`SNr_directstim`, `ZI_directstim`) are declarative, but
the legacy condition-selection route still needs to be wired and regression
tested before public raw execution is enabled.
