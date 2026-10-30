"""Prepare view banks and a pilot, including when the original project was moved."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def resolve_cats_root(old, explicit=None):
    """Locate the images without rewriting the original, hashed split manifest."""
    base = Path(explicit).expanduser().resolve() if explicit else old / 'data'
    candidates = [base, base / 'PetImages', base / 'catsdogs' / 'PetImages']
    found = sorted({p.resolve() for p in candidates
                    if (p / 'Cat').is_dir() and (p / 'Dog').is_dir()})
    if len(found) != 1:
        raise SystemExit('Cannot uniquely locate PetImages. Supply --cats-root '
                         '"/absolute/path/to/PetImages" containing Cat and Dog folders.')
    return found[0]


def preflight(old, cats):
    for dataset in ('catsdogs', 'cifar10'):
        cache = old / 'cache' / dataset / 'main'
        for name in ('metadata.json', 'split_manifest.json'):
            if not (cache / name).is_file():
                raise SystemExit('Missing original cache file: ' + str(cache / name))
    manifest = old / 'cache' / 'catsdogs' / 'main' / 'split_manifest.json'
    rows = json.loads(manifest.read_text())['train']
    missing = []
    for row in rows:
        original = Path(row['path'])
        relocated = cats / original.parent.name / original.name
        if not relocated.is_file():
            missing.append(str(relocated))
            if len(missing) == 3:
                break
    if missing:
        raise SystemExit('Training image(s) missing at the relocated dataset:\n' +
                         '\n'.join(missing) + '\nCheck --cats-root. No split files were changed.')


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-project')
    parser.add_argument('--cats-root', help='Optional relocated PetImages folder or parent')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.old_project:
        old = Path(args.old_project).expanduser().resolve()
    else:
        names = ('Tree_Network_Revisited_2025EEA3026', 'Tree_Network_2025EEA3026')
        candidates = [parent / name for parent in
                      (Path.home() / 'Downloads', Path.cwd().parent) for name in names]
        found = sorted({p.resolve() for p in candidates if all(
            (p / 'cache' / d / 'main' / 'metadata.json').is_file()
            for d in ('catsdogs', 'cifar10'))})
        if len(found) != 1:
            raise SystemExit('Provide --old-project "/absolute/path/to/ORIGINAL_training_project" '
                             '(must contain both cache/DATASET/main folders).')
        old = found[0]
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        raise SystemExit('Run from the extracted continuation folder.')
    cats = resolve_cats_root(old, args.cats_root)
    preflight(old, cats)
    print('Original project:', old, flush=True)
    print('Resolved Cats vs Dogs images:', cats, flush=True)
    run('selftest.py')
    for dataset in ('catsdogs', 'cifar10'):
        command = ['prepare_views.py', '--base-cache', old / 'cache' / dataset / 'main',
                   '--data-root', old / 'data', '--out', root / 'banks' / dataset,
                   '--device', args.device]
        if dataset == 'catsdogs':
            command += ['--cats-root', cats]
        run(*command)
    plan = root / 'plans/catsdogs/pilot.json'
    if not plan.exists():
        run('make_plan.py', '--dataset', 'catsdogs', '--phase', 'pilot', '--out', plan)
    run('experiment.py', '--bank', 'banks/catsdogs', '--plan', plan,
        '--out', 'runs/catsdogs', '--threads', '4')
    run('analyse.py', '--bank', 'banks/catsdogs', '--results', 'runs/catsdogs',
        '--out', 'analysis/catsdogs/pilot')
    run('bundle_results.py', '--dataset', 'catsdogs')
    print('Upload Continuation_catsdogs_evidence.zip. Copy the package and '
          'banks/cifar10 to Ubuntu for its pilot.')


if __name__ == '__main__':
    main()
