# Analysis migration plan

This document records which source is authoritative while the manuscript
repository is made reproducible. Scientific files must not be copied or
refactored merely because one version is newer; each migration needs an output
comparison against a trusted manuscript result.

## Source audit (2026-07-27)

### Behavioral optogenetics

- Active source: `/Users/natemiska/python/opto_analysis/`
- Repository `opto_analysis.py` and `helpers.py`: byte-identical to active source
  at audit time.
- Active `config.py` and `metadata_all.py`: differ from the repository copy.
- Implemented: computation is retained while profile-driven runtime overrides
  replace mutable/commented cohort selection. Output, GLM-HMM, and ONE cache
  paths are supplied at runtime.

### ZAPIT

- Active source: `/Users/natemiska/python/zapit/`
- Main analysis, helpers, configuration, metadata and YAML logs: byte-identical
  to the repository copy at audit time.
- Implemented: portable paths and a profile adapter. Remaining validation is a
  numerical regression check against the final Figure 4 outputs.

### Ephys bias-selectivity analysis

- Active source: `/Users/natemiska/python/CD_analysis/`
- The active `SNr_inhibition_BS_downstream_effect.py` was substantially newer
  than the repository copy and has now been migrated as one dependency-closed
  unit under `ephys_analysis/`.
- Required active files include:
  - `SNr_inhibition_BS_downstream_effect.py`
  - `BS_config.py`
  - `BS_postprocess.py`
  - `optostim_preprocessing.py`
  - `metadata_optostim_new.py`
  - `CD_config.py`
  - `functions_optostim.py` or its canonical packaged equivalent
  - `waveform_classify.py`
- Implemented: dependency snapshot and checksums. Remaining: path-only adapter,
  import smoke test in the full IBL environment, and comparison of compact
  derived outputs against the final trusted NOGLMHMM laser- and
  feedback-aligned results before any internal refactor.

## Staged implementation

1. **Reproducibility foundation (implemented on current branch)**
   - Declarative figure profiles.
   - GLM-HMM contract tests.
   - IBL release-manifest generator.
   - Stable derived-data format policy.

2. **Pipeline snapshots (Figure 2 implemented)**
   - Copy each authoritative dependency-closed pipeline.
   - Record SHA-256 hashes and source provenance.
   - Add `if __name__ == '__main__'` boundaries without changing calculations.

3. **Portable adapters (behavior/ZAPIT implemented; ephys pending)**
   - Convert absolute paths to CLI/configuration arguments.
   - Resolve cohorts and repeated run matrices from JSON profiles.
   - Save resolved IDs and run configuration with every output.

4. **Derived-data exporters**
   - Convert final trusted PKLs into versioned CSV/JSON/NPZ bundles.
   - Check numerical equality and file checksums.
   - Keep PKLs optional, never required by public plotting code.

5. **Figure rendering**
   - Add one generic `scripts/reproduce.py` interface.
   - Add figure-specific Python only where custom multi-panel assembly is
     genuinely required.

6. **Full validation and release**
   - Synthetic CI tests without IBL credentials.
   - One-EID/PID smoke runs in an IBL environment.
   - Full-cohort regression checks against manuscript reference outputs.
   - Environment lock, license, citation file and archival derived-data release.

## Worktree safety

This foundation was created in a clean worktree based on `origin/main`:

```text
/Users/natemiska/python/Miska_et_al_2026_repro
branch: codex/reproducibility-foundation
```

The older dirty checkout and its unfinished packaging work were intentionally
left untouched.
