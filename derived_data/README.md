# Stable derived-data bundles

Derived-data releases should let a reader regenerate manuscript plots without
loading raw spikes or repeating long permutation/PETH analyses. Python pickle is
not the required interchange format because it is Python-specific, unsafe to
load from untrusted sources, and sensitive to package/class changes.

## Required public formats

Each figure bundle should use:

- **CSV** for rectangular tables, cohort membership, scalar statistics, and QC.
- **NPZ** for dense numeric arrays such as time axes and population traces.
- **JSON** for configuration, units, array descriptions, provenance, and schema
  versions.
- **PDF/SVG** for reference vector figures, with PNG previews where useful.

Parquet or Zarr may be added for large tables/arrays, but a small CSV/NPZ view
should remain available so reproduction does not depend on a particular storage
library.

## Bundle contract

```text
derived_data/figure_2/
├── manifest.json
├── units.csv
├── insertion_summary.csv
├── population_traces.npz
└── reference/
    ├── laser.pdf
    └── feedback.pdf
```

`manifest.json` should contain at least:

- `schema_version`
- manuscript figure/profile IDs
- source Git commit
- analysis configuration
- source EIDs/PIDs or a link to `release_manifest/identifiers.csv`
- every NPZ key, shape, dtype, units, and axis meaning
- software/environment identifier
- SHA-256 checksums for every file in the bundle

NPZ keys must be descriptive rather than positional, for example:

```text
time_s
control_mean_hz
control_sem_hz
opto_mean_hz
opto_sem_hz
```

## Pickle policy

Existing PKLs can be archived as optional provenance files, preferably with the
exact Python environment and checksum. Public figure scripts must not require
them. A one-time exporter will convert the final trusted PKLs into the stable
bundle above and validate array/table equality before release.
