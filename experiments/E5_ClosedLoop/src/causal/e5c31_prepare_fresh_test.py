"""E5-C31: Generate sealed, fresh synthetic test datasets with provenance.
No training, calibration, inference, or ground-truth inspection is performed.
Seed novelty relative to historical runs must be verified by the researcher.
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generator', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--campaigns', type=int, default=100)
    p.add_argument('--duration', type=int, default=2400)
    p.add_argument('--background', type=int, default=10000)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Duplicate seeds')
    if a.duration <= 400 or a.campaigns < 2 or a.background < 1:
        p.error('Invalid generation parameters')
    generator = a.generator.resolve()
    if not generator.is_file():
        p.error(f'Generator not found: {generator}')
    spec = importlib.util.spec_from_file_location('e5c31_generator', generator)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Fail closed: do not overwrite an existing seed or existing manifest.
    manifest_file = out / 'manifest.json'
    if manifest_file.exists() or any((out / f'seed_{s}').exists() for s in a.seeds):
        raise FileExistsError('Output already contains requested seed folders or manifest. Use a new output path.')
    records = []
    for seed in a.seeds:
        dest = out / f'seed_{seed}'
        dest.mkdir(exist_ok=False)
        telemetry, truth = mod.generate(seed, a.campaigns, a.duration, a.background)
        audit = mod.audit(telemetry, truth, a.campaigns)
        mod.write_csv(dest / 'telemetry.csv', telemetry, list(telemetry[0]))
        mod.write_csv(dest / 'ground_truth.csv', truth, list(truth[0]))
        (dest / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        record = {'seed': seed, 'folder': dest.name, 'audit': audit,
                  'sha256': {name: sha256(dest / name) for name in
                             ('telemetry.csv', 'ground_truth.csv', 'audit.json')}}
        records.append(record)
        print(f'Seed {seed}: {audit["events"]} events, {audit["true_edges"]} true edges')
    manifest = {'design': 'Fresh synthetic evaluation sets, no model fitting',
                'generated_utc': datetime.now(timezone.utc).isoformat(),
                'generator_path': str(generator), 'generator_sha256': sha256(generator),
                'parameters': {'campaigns': a.campaigns, 'duration': a.duration,
                               'background': a.background},
                'historical_seed_disjointness_verified': False,
                'warning': 'Researcher must verify these seeds were not used for model selection or earlier experiments; avoid examining labels until predictions are frozen.',
                'datasets': records}
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('E5-C31 | FRESH SYNTHETIC TEST DATA GENERATED')
    print('Saved:', out)
    print('Manifest:', manifest_file)
    print('Do not use these test labels for model selection.')

if __name__ == '__main__':
    main()
