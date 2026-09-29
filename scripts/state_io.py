"""Load legacy trusted pickles or portable, object-free GLM-HMM NPZ archives."""
from pathlib import Path
import json
import pickle
import numpy as np


def load_states(path):
    path = Path(path)
    if path.suffix != '.npz':
        with path.open('rb') as f:
            return pickle.load(f)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest']))
        if manifest['schema_version'] != 1:
            raise ValueError('Unsupported states schema')
        result = {subject: {} for subject in manifest['subjects']}
        for record in manifest['sessions']:
            if record.get('empty'):
                value = {} if record['empty'] == 'dict' else []
            else:
                key = record['key']
                value = [archive[key + '_labels'], archive[key + '_posterior']]
            result[record['subject']][record['eid']] = value
        return result


def export_states(states, output):
    """Export labels and probabilities exactly; preserve empty session entries."""
    arrays = {}
    records = []
    for subject, sessions in states.items():
        for eid, value in sessions.items():
            record = {'subject': str(subject), 'eid': str(eid)}
            if not len(value):
                if not isinstance(value, (dict, list)):
                    raise ValueError(f'{subject}/{eid}: unsupported empty entry')
                record['empty'] = 'dict' if isinstance(value, dict) else 'list'
            else:
                labels, posterior = map(np.asarray, value)
                if (labels.ndim != 1 or posterior.ndim != 2
                        or len(labels) != len(posterior)
                        or labels.dtype.hasobject or posterior.dtype.hasobject):
                    raise ValueError(f'{subject}/{eid}: invalid state arrays')
                key = f'session_{len(records):06d}'
                arrays[key + '_labels'] = labels
                arrays[key + '_posterior'] = posterior
                record['key'] = key
            records.append(record)
    manifest = {'schema_version': 1, 'trial_indexing': 'zero-based array row order',
                'state_labels': 'one-based; original ordering preserved',
                'subjects': [str(subject) for subject in states], 'sessions': records}
    arrays['manifest'] = np.array(json.dumps(manifest))
    output = Path(output)
    if output.suffix != '.npz':
        raise ValueError('Output must end in .npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing publication archives.
    with output.open('xb') as f:
        np.savez_compressed(f, **arrays)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Convert a trusted local state pickle into a portable NPZ archive.')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args(argv)
    original = load_states(args.input)
    export_states(original, args.output)
    restored = load_states(args.output)
    for subject, sessions in original.items():
        for eid, value in sessions.items():
            other = restored[subject][eid]
            if not len(value):
                assert type(value) is type(other) and not len(other)
            else:
                for before, after in zip(value, other):
                    np.testing.assert_array_equal(before, after)
                    assert np.asarray(before).dtype == after.dtype
    print(f'Exported and verified exact array equality: {args.output}')

if __name__ == '__main__':
    main()
