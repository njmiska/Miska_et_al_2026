# Alberto's reproducibility review — implementation and validation

Changes are in a separate checkout on branch `reproducibility-review`. The
original laboratory scripts, GLM-HMM source, and state archive have not been
edited. This branch is prepared for review before merging into the main branch.

## 1. Software environment

- Added `requirements.txt`, `environment.yml`, optional Plotly requirements, and
  setup instructions, plus a full new setup resolution in `requirements-lock.txt`.
  Nate confirmed that `iblenv` generated the states.
- Recorded the current package inventory, source commits, Python-file checksums,
  and the GLM-HMM patch fingerprint under `provenance/`. The patch itself is
  retained locally while source distribution is being discussed with Alberto.
- The actual imported NumPy version is **2.3.5**. Duplicate distribution metadata
  also reports 2.2.0; blindly using the metadata would pin the wrong version.
  IBL 3.4.1 declares numpy<=2.2, so a fresh install with 2.3.5 cannot resolve.
  Requirements use 2.2.0 as an explicit compatibility adjustment; the three-session
  retrieval regression passed with both 2.3.5 and 2.2.0. The inventory preserves
  both metadata observations.
- The original script explicitly adds the separate `ssm` checkout at revision
  `e9b408b79e2ab22a05ca93c3a1f78a7dae461992`. The installed nested copy is instead
  `eb6c8aa33e5311d3564075807dec340759dd8081`. The new script selects `ssm-dir`
  before adding GLM-HMM to the import path, preventing the nested directory from
  shadowing the import. Regression checks used the original script's revision.
- `future==1.0.0` is a new setup pin for the older ssm package's declared
  dependency, not a historical environment claim. Optional Plotly 5.22.0 was
  observed in the base environment, not iblenv.

The compatible direct requirements resolved successfully in an ignore-installed
pip dry run. Transitive versions in the new lock can differ from the laboratory
inventory. A dependency resolution dry run is not a fresh installation or validation on
another OS. The environment snapshot cannot establish which package versions
were installed on each historical analysis date.

## 2. Retrieval script and the range error

Added `scripts/retrieve_allsubject_states.py`, adapted from the supplied local
script, with explicit subject, metadata, output, and path arguments.

The original loop allocates 18 lengths, then iterates `range(0,17)`. Its
`j == 17` intervals branch is unreachable. There is no explanation in the source
that establishes an intentional exclusion. If intervals was never shorter than
other fields, the omission would have remained dormant. This code only runs
when a cached table is unavailable and conversion of the trials object fails.

Merely changing the range would expose a second error: concatenating `[0]` with
an `(N, 2)` intervals array. `scripts/trial_alignment.py` now iterates field names,
validates shapes, handles a zero **row** for intervals, and rejects discrepancies
larger than one. It leaves its input untouched and preserves dtypes.

Padding is **not enabled automatically**. A short array does not establish that
its first entry is missing. An explicitly named, independently checked session
can use `--allow-leading-padding-eid`; the repair is logged. No real session was
padded or relabeled during this review.

Other retrieval changes:

- Preserved seed 41, design-matrix construction, model selection, posterior
  calculation, and four-state ordering.
- Removed the hard-coded working directory and private metadata import.
- Made the existing “skipping” message actually skip valid cached entries;
  `--overwrite` enables explicit recomputation into a new output archive.
- Deduplicated selected EIDs and preserved existing entries if recomputation
  fails. Fixed failure accounting that could reference an undefined subject.
- Rejects a posterior/table row-count mismatch before saving. The inherited
  individual-fit path loads a subject-wide design matrix; its session selection
  still needs validation before that path can be described as verified.
- Writes new `.pkl` output plus repair and run metadata sidecars, with nonzero
  exit status if counted session failures occurred. Existing output files are
  refused. Per-session tables and plots use the configured output directory.

## 3. State archive format

Confirmed that the historical `.csv` is a pickle. Readers prefer `.pkl`, accept
the legacy filename with a warning, and also support a portable NPZ archive.

`scripts/state_io.py` exports numeric arrays plus a versioned JSON manifest;
loading NPZ disables pickle. It preserves state labels, probability arrays,
dtypes, subject/session identifiers, row order, and empty entries. Pickles should
only be loaded from trusted sources.

The complete local archive contains **64 subjects, 1,549 populated session
entries, and 32 empty entries**. Every populated entry has matching label and
posterior row counts and labels equal to the posterior argmax plus one. All
populated posteriors have four columns. This is a structural check, not proof
that each row corresponds to the correct original behavioral trial.

## 4. Shared paths and data setup

`scripts/setup_data.py` writes ignored `paths.local.json`; `scripts/paths.py`
resolves its paths consistently. Configuration covers model source/data, ssm,
ONE cache, state archive, outputs, Allen assets, and the BWM bias table. CLI and
environment overrides remain supported. Relative JSON paths resolve against the
configuration file's directory.

The setup script optionally caches explicitly selected ONE objects by EID.
The download route was tested with a fake ONE client; no claim is made that the
whole release inventory is publicly downloadable. Fitted models, Allen assets,
and the other external tables still require supplied files or public release
locations. No public model-asset URL was available to add automatically.

## 5. Machine-specific analysis paths

Replaced active personal paths in all six files named by Alberto and in both
behavioral configurations. This includes additional `~/python/saved_figures`
paths in ephys postprocessing. Changed two imports of the unavailable local
`miska_analysis` package to the matching repository helper module.

The ephys state-index helper is copied directly from the untracked local
`psychometric_utils.py`, retaining its exact state grouping and tuple return
convention. Its original checksum is recorded. Ephys loads state data only when
GLM-HMM filtering is enabled. Behavioral and ZAPIT entrypoints no longer load
the previous-trial pickle files whose returned values were discarded; their
existing previous-trial filtering calculations remain unchanged.

Original ephys snapshot hashes are retained alongside updated hashes. The
existing Figure 2 execution guard remains: path adaptation does not replace the
needed one-PID/full-cohort numerical regression.

## Verification

- **15 automated tests pass**, including the existing manuscript/cohort tests,
  path precedence and relative paths, setup/download argument wiring, legacy
  filename compatibility, trial-padding shape and failure cases, and portable
  state round trips with state grouping preserved.
- Python compilation and Git whitespace checks pass for the changed code.
- The complete archive was exported to NPZ and read back with exact array and
  dtype equality, preserving all empty entries. The generated data stays outside
  the source checkout.
- The new retrieval entrypoint was run offline on three cached session tables
  using the original ssm source and global-fit assets. A local ONE adapter
  supplied cached trial data; it did not test network/session discovery.
  **All 2,392 labels matched the saved archive exactly**. Maximum posterior
  differences were 3.64e-14, 4.88e-14, and 5.67e-14. Details are in
  `provenance/validation-results.json`. This also passed with NumPy 2.2.0
  supplied from an isolated dependency directory; the original environment was
  not changed.
- No full manuscript regeneration, real padding repair, individual-fit
  regression, or clean-machine installation has been performed. Existing raw
  data and saved state labels were not replaced.

The first exploratory ONE-cache check also reported a local MD5/file-size
mismatch for a SWC_NM_004 session; the successful regression instead used saved
GLM-HMM parquet tables. Raw-cache integrity was not established by these tests.

## Remaining release work

Provide an accessible archive/location for the fitted model assets and required
external datasets, validate the individual-fit/session-selection path and any
specific session needing padding, and test a clean installation. Preserve the
current state archive as the manuscript baseline until broader regression is
complete. These changes are ready for code review, not a claim that every
manuscript figure has been independently reproduced.
