"""Shared runtime paths. CLI overrides > environment > JSON > repo defaults."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('data-root', 'output-root', 'glmhmm-dir', 'one-cache-dir',
             'ssm-dir', 'glmhmm-data-dir', 'glmhmm-figure-dir', 'states-file',
             'allen-annotation', 'allen-structure-tree', 'bias-selectivity-table')


def load_paths(config=None, overrides=None, *, allow_missing=False):
    explicit = config or os.environ.get('MISKA_PATHS_FILE')
    config = Path(explicit).expanduser().resolve() if explicit else REPO_ROOT / 'paths.local.json'
    raw = {}
    if config.exists():
        raw = json.loads(config.read_text())
        if not isinstance(raw, dict) or not all(isinstance(v, str) for v in raw.values()):
            raise ValueError(f'{config}: expected an object of path strings')
        unknown = set(raw) - set(PATH_KEYS)
        if unknown:
            raise ValueError(f"{config}: unknown path keys {sorted(unknown)}")
    elif explicit and not allow_missing:
        raise FileNotFoundError(config)
    defaults = {
        'data-root': REPO_ROOT / 'data',
        'output-root': REPO_ROOT / 'outputs',
        'glmhmm-dir': REPO_ROOT / 'external' / 'GLM-HMM',
    }
    def resolve(key, default):
        value = (overrides or {}).get(key)
        if value is not None:
            return Path(value).expanduser().resolve()
        value = os.environ.get('MISKA_' + key.upper().replace('-', '_'))
        if value:
            return Path(value).expanduser().resolve()
        if key in raw:
            path = Path(raw[key]).expanduser()
            return (config.parent / path).resolve()
        return Path(default).resolve()
    result = {k: resolve(k, v) for k, v in defaults.items()}
    dependent = {
        'one-cache-dir': result['data-root'] / 'ONE',
        'ssm-dir': result['glmhmm-dir'].parent / 'ssm',
        'glmhmm-data-dir': result['glmhmm-dir'] / 'outputs',
        'glmhmm-figure-dir': result['output-root'] / 'glmhmm',
        'states-file': result['glmhmm-dir'] / 'all_subject_states.pkl',
        'allen-annotation': result['data-root'] / 'allen' / 'annotation_volume_10um.npy',
        'allen-structure-tree': result['data-root'] / 'allen' / 'structure_tree_safe_2017.csv',
        'bias-selectivity-table': result['data-root'] / 'bs_analysis_allregions_dataframe_newanalysis.csv',
    }
    result.update({k: resolve(k, v) for k, v in dependent.items()})
    return result


def states_file(paths=None):
    """Prefer the honest .pkl suffix, while reading existing legacy archives."""
    paths = load_paths() if paths is None else paths
    path = paths['states-file']
    legacy = path.with_suffix('.csv')
    if not path.exists() and path.name == 'all_subject_states.pkl' and legacy.exists():
        warnings.warn(f'Using legacy pickle {legacy}; its .csv suffix does not mean CSV.', stacklevel=2)
        return legacy
    return path
