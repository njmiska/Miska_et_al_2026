# GLM-HMM states: use the saved labels or optionally recompute them

The recommended way to reproduce the manuscript analyses is to **use the saved
trial-state labels and probabilities**. You do not need to train a GLM-HMM,
install Alberto's model code, or install `ssm` for this workflow. You still need
the behavioral/ephys data required by the figure you are reproducing.

**Draft release status:** The source supports the workflows below, but the public
state-archive download, checksum, and model-asset links have not yet been added.
The examples assume you already have the relevant files. They do not download
an archive that has not been released. Model-code hosting is being discussed
with Alberto; no access to his private repository is assumed.

## Choose a workflow

| Objective | Needed | Status |
|---|---|---|
| Use the manuscript's saved states | This repository, the state archive, and the figure's underlying data | Recommended; state-archive release link pending |
| Recompute states with the original fitted models | Above, plus exact GLM-HMM/ssm source, fitted parameters, and preprocessing assets | Optional; source/asset access pending, partial regression completed |
| Refit models from the training data | Model source, training cohort, preprocessing, fit settings, and model-selection procedure | Separate upstream workflow; complete instructions pending |

Recomputing probabilities with fixed fitted parameters and training a new model
are different operations. The retrieval script in this repository performs the
former. A newly fitted model may have different parameters and state numbering.

## Recommended: use saved states

### 1. Install the analysis environment

From the repository root, follow the [environment setup](../README.md#setup-and-local-paths):

```bash
conda env create -f environment.yml
conda activate miska-2026
```

The environment supports analysis of saved labels without the separate GLM-HMM
and `ssm` installations. See the main README for the macOS dependency lock and
its validation limits. A clean-machine installation remains to be checked.

### 2. Obtain the archive and configure paths

The intended public format is `all_subject_states.npz`. The final release will
identify its download location, SHA-256 checksum, cohort coverage, and matching
code version. These release details are **pending**.

Once you have the archive, place it at `data/glmhmm/all_subject_states.npz`, or
choose another location and substitute it in this command:

```bash
python scripts/setup_data.py \
  --data-root ./data \
  --states-file ./data/glmhmm/all_subject_states.npz \
  --one-cache-dir ./data/ONE \
  --output-root ./outputs
```

This records paths in `paths.local.json`, which stays outside Git. If you already
have a ONE cache, use its location instead of `./data/ONE`. You do not need to set
`glmhmm-dir` or `ssm-dir` when using saved states. The setup command configures
paths; it does not itself download the state archive.

The other figure inputs are described in the [figure map](../README.md#manuscript-figure-map)
and [release inventory](../release_manifest/). Configuring paths does not make
unreleased datasets accessible.

### 3. Preview and run an analysis

```bash
# Inspect the cohort, parameters, and resolved paths without loading data.
python scripts/reproduce.py figure1_snr_glmhmm --dry-run

# Run after the state archive and required behavioral data are available.
python scripts/reproduce.py figure1_snr_glmhmm
```

To preview the four state-specific supplementary runs:

```bash
python scripts/reproduce.py supp1_glmhmm_states --dry-run
```

The runner executes all four runs unless `--run state1` (or another run name) is
specified. Some other manuscript profiles remain guarded pending further
validation; consult their status rather than assuming every figure is runnable.

### 4. Inspect the archive directly

From the repository root:

```python
from scripts.state_io import load_states

states = load_states("data/glmhmm/all_subject_states.npz")
subject = "SWC_NM_110"  # Substitute the mouse of interest.
for eid, record in states[subject].items():
    if not record:
        print(eid, "no saved state result")
        continue
    labels, posterior = record
    print(eid, labels.shape, posterior.shape)
```

Each populated session contains:

- `labels`: one label per trial, numbered 1 through 4.
- `posterior`: an `(N_trials, 4)` probability array. Column 0 corresponds to label
  1, column 1 to label 2, and so on.
- The subject and session EID identify the session; array rows retain the saved
  trial order. Match these rows to the original session trials **before** filtering
  trials or resetting a table's index. An EID identifies an experimental session.

Empty entries mean no saved result; they are not a fifth state and must not be
interpreted as disengagement. The inspected local archive has 64 subjects,
1,549 populated session entries, and 32 empty entries. These describe the local
archive inspected during development, not a finalized public-release cohort.
Use the selected figure profile to determine the manuscript analysis cohort.

## What the labels mean in this manuscript

The four fitted states are grouped by the existing analysis helpers as follows:

| Stored label | Interpretation used by the helpers | Four-state analysis option |
|---|---|---|
| 1 | Right disengaged | `state4` |
| 2 | Right engaged | `state3` |
| 3 | Left engaged | `state1` |
| 4 | Left disengaged | `state2` |

The historical `state1`–`state4` **analysis options are reordered aliases**, not
the raw archive label numbers. The table reflects the current helper code;
retain this distinction when interpreting supplementary runs.

Analysis setting `n_states=2` pools stored labels **2 and 3 as engaged**, and
**1 and 4 as disengaged**. It does not mean the original model was fitted with
only two hidden states. This mapping applies to the manuscript's fitted models;
it should not be assumed for a newly trained model.

The behavioral GLM-HMM manuscript profiles use `state_definition="previous"`:
trial `t` is selected using the saved state of trial `t-1`. The helpers shift
selected indices by one and intersect them with the eligible session trials.
Do not shift the archive itself; that would risk applying the shift twice.
Trial 0 has no previous-trial label within its session. See the individual
[figure profiles](../configs/figures/) for the actual selection and thresholds.

## Optional: recompute states using the original fitted models

### Source and asset access

The intended source is Alberto's [IBL GLM-HMM repository](https://github.com/int-brain-lab/GLM-HMM).
It is currently private according to the authors. The publication workflow will
use either a public, pinned upstream revision or an author-approved source
snapshot distributed with the manuscript. This hosting decision is pending.

For the existing local implementation, [source provenance](../provenance/local-source.json)
records:

- GLM-HMM base commit `47ca6a1ae36dd42fe4db0464f7710e3fc66a31c1`, plus local
  modifications. Source-file and patch fingerprints are recorded, but the patch
  itself is retained locally pending agreement on source distribution.
- The separate `ssm` checkout used by the original retrieval script:
  `e9b408b79e2ab22a05ca93c3a1f78a7dae461992` from [zashwood/ssm](https://github.com/zashwood/ssm).

A matching commit alone does not include the recorded local modifications or
fitted data. Before this route is publicly reproducible, the release must also
provide the selected model parameters, cross-validation/model-selection files,
applicable subject preprocessing files, and session trial data. Instructions for
installing the separate model source are in the [main README](../README.md#setup-and-local-paths).

### Configure inference and choose subjects

Once those sources and assets are available:

```bash
python scripts/setup_data.py \
  --glmhmm-dir /path/to/GLM-HMM \
  --ssm-dir /path/to/ssm \
  --glmhmm-data-dir /path/to/GLM-HMM/outputs \
  --glmhmm-figure-dir ./outputs/glmhmm-recomputed

python scripts/retrieve_allsubject_states.py \
  --subject SWC_NM_110 \
  --overwrite \
  --output ./outputs/recomputed_states.pkl
```

`glmhmm-data-dir` contains the `GLM-HMM/results` and applicable
`GLM-HMM/data_by_animal` subdirectories. Keep generated plots/tables in a separate
output directory from the original fitted assets.

Repeat `--subject` to choose additional mice. The default metadata is
`metadata_final/metadata_opto_behavior.py`; use
`--metadata metadata_final/metadata_zapit.py` for the ZAPIT session list.

The script loads the configured saved archive when it exists. Without
`--overwrite`, existing populated entries are skipped. With `--overwrite`, the
selected entries are recomputed into a **new** output file; the input archive is
not overwritten. Unselected input entries remain in that output archive. The
output filename must not already exist. Run and repair JSON sidecars accompany
it; counted session failures produce a nonzero exit status.

This recomputation route has been checked on three cached sessions using global
model parameters. Their 2,392 labels exactly matched the saved archive, with
posterior differences below 6e-14, using both NumPy 2.3.5 and the compatible
NumPy 2.2.0 setup. This was an offline check using saved trial tables, not a
full-cohort or fresh-data reproduction.

### Trial-length mismatches

The older retrieval script sometimes inserted a zero at the start of a shorter
trial array so all fields had the same length. This assumes the missing entry
belongs at the **beginning**. Lengths alone cannot establish that assumption.

When table construction fails, the revised script does not silently make this
repair. Only for a session independently verified to need the historical repair
should `--allow-leading-padding-eid SESSION_UUID` be used. It prepends one zero
row to each short field, handles the two-column `intervals` array, rejects larger
gaps, and logs the change. No real session was padded during our validation.

### Mouse-specific model path: validation still needed

The inherited mouse-specific path loads a subject-level processed file that may
contain multiple sessions. The upstream posterior helper returns concatenated
results for the supplied sessions. We still need to verify how the requested
session is selected before attaching those results to its trial table. The new
script rejects a posterior/table length mismatch, but equal lengths alone do not
prove session identity or trial alignment.

This is an outstanding check on that code path, not evidence that the manuscript
state archive is incorrect. The successful three-session regression exercised
the alternative path that builds a single-session input and uses global model
parameters.

## Legacy files and validation records

The historical `all_subject_states.csv` is a **pickle despite its filename**.
Existing trusted local copies remain readable. To export one to portable NPZ:

```bash
python scripts/state_io.py /path/to/all_subject_states.csv ./data/glmhmm/all_subject_states.npz
```

Use a new output path. The converter checks array and dtype equality after
reading the export back. It preserves empty entries and does not regenerate
states. NPZ loading disables pickle; legacy pickle files must come from a trusted
source. Public readers should receive the exported NPZ directly.

The full local archive passed an exact conversion check. Automated checks,
regression results, and limitations are recorded in the
[reproducibility review](../REPRODUCIBILITY_REVIEW.md) and
[validation results](../provenance/validation-results.json).

## Items to finalize before the public release

- State-archive download, checksum, final cohort/empty-entry inventory, and code tag.
- Alberto's preferred public source location, exact revision, attribution, and
  redistribution terms if code is included in this repository.
- Public fitted-model and preprocessing assets for optional recomputation.
- Clean-install check, mouse-specific session-selection validation, and any
  justified session-specific padding decisions.
- Training data, settings, and upstream instructions if full refitting is offered.

Using the saved states is the default even after optional recomputation becomes
fully available. The release archive should remain the reference for reproducing
the manuscript's analyses.
