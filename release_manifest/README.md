# IBL public-data release manifest

Run from the repository root:

```bash
python3 scripts/release_manifest.py
```

To add each ephys PID's parent EID and probe name through lightweight Alyx
queries (no dataset download):

```bash
python3 scripts/release_manifest.py --resolve-pids-with-one
```

Generated files:

- `figure_profiles.csv`: named manuscript cohorts, modes, and resolved counts.
- `profile_runs.csv`: every concrete run after applying state/hemisphere/direct-
  stimulation overrides, including serialized filters and thresholds.
- `metadata_rows.csv`: every final metadata row with compact trial ranges and
  figure-profile membership.
- `identifiers.csv`: deduplicated EIDs and PIDs.
- `dataset_requests.csv`: identifier-by-data-family requests inferred from the
  actual pipeline access patterns.
- `external_assets.csv`: GLM-HMM, ZAPIT, atlas, and derived-data assets that are
  not ordinary ONE datasets.
- `unresolved_identifiers.csv`: PIDs that are present in frozen manuscript
  metadata but do not currently resolve to a parent EID in Alyx. These are
  explicit IBL release follow-ups, not silently discarded rows.
- `summary.json`: machine-readable counts and schema version.

`dataset_requests.csv` deliberately uses semantic object names where ONE or
`SpikeSortingLoader` resolves the exact collection at runtime. Before the public
release is finalized, a validation pass should resolve those entries against
ONE and record the concrete dataset paths/collections for each identifier.

The manifest includes all identifiers in `metadata_final`, not only the primary
figure subsets. `figure_profiles.csv` and the `figure_profiles` column provide
the figure-specific view. Profiles awaiting colleague code or a frozen cohort
remain visible with zero identifiers and a non-ready status.
