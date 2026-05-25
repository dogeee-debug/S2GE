import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in [ROOT, ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export offline bundles')
    parser.add_argument('--target', choices=['datasets', 'models', 'wheels', 'all'], default='all')
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    commands = []
    if args.target in ('wheels', 'all'):
        commands.append(['bash', str(repo_root / 'local_prep' / 'package' / 'build_wheelhouse.sh')])
    if args.target in ('datasets', 'all'):
        commands.append(['bash', str(repo_root / 'local_prep' / 'package' / 'pack_dataset_bundle.sh')])
    if args.target in ('models', 'all'):
        commands.append(['bash', str(repo_root / 'local_prep' / 'package' / 'pack_model_bundle.sh')])
    for cmd in commands:
        result = subprocess.call(cmd, shell=False)
        if result != 0:
            raise SystemExit(result)
    raise SystemExit(0)


if __name__ == '__main__':
    main()
