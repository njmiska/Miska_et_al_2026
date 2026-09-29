#!/usr/bin/env python3
"""Write local paths and optionally download explicitly selected ONE objects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
try:
    from .paths import load_paths, REPO_ROOT, PATH_KEYS
except ImportError:
    from paths import load_paths, REPO_ROOT, PATH_KEYS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=REPO_ROOT / 'paths.local.json')
    for key in PATH_KEYS:
        p.add_argument('--' + key, type=Path)
    p.add_argument('--eid', action='append', default=[])
    p.add_argument('--object', action='append', dest='objects', default=[])
    p.add_argument('--base-url', default='https://alyx.internationalbrainlab.org')
    args = p.parse_args(argv)
    overrides = {k: getattr(args, k.replace('-', '_')) for k in PATH_KEYS}
    paths = load_paths(args.config, overrides, allow_missing=True)
    # Only create data/cache/output folders, never pretend model assets exist.
    for key in ('data-root', 'one-cache-dir', 'output-root', 'glmhmm-figure-dir'):
        paths[key].mkdir(parents=True, exist_ok=True)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps({k: str(v) for k, v in paths.items()}, indent=2) + '\n')
    print(f'Wrote {args.config.resolve()}')
    if args.eid:
        from one.api import ONE
        one = ONE(base_url=args.base_url, cache_dir=paths['one-cache-dir'])
        for eid in dict.fromkeys(args.eid):
            for obj in args.objects or ['trials']:
                one.load_object(eid, obj)
                print(f'Cached {eid}: {obj}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
