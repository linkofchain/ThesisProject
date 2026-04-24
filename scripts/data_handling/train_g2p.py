"""
Train an MFA g2p model from the prepared Ulster pronunciation dictionary.

Runs prep_g2p_dict.py first if the dict file doesn't exist, then calls:
    mfa train_g2p <dict> <output_model> [--evaluate] [--num_jobs N]

Usage:
    conda run -n thesis python scripts/data_handling/train_g2p.py
    conda run -n thesis python scripts/data_handling/train_g2p.py --evaluate --num_jobs 6
"""

import argparse
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path('.').resolve()
while not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DICT_PATH    = PROJECT_ROOT / 'data' / 'teanglann' / 'g2p_dicts' / 'ulster_for_g2p.txt'
MODEL_PATH   = PROJECT_ROOT / 'models' / 'mfa' / 'ulster_g2p_trained.zip'
TEMP_DIR     = PROJECT_ROOT / 'models' / 'mfa' / 'tmp'


def prep_dict():
    print("Dict not found — running prep_g2p_dict.py ...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' / 'data_handling' / 'prep_g2p_dict.py')],
        check=True,
    )


def train(evaluate: bool, num_jobs: int, overwrite: bool):
    if not DICT_PATH.exists():
        prep_dict()

    cmd = [
        'mfa', 'train_g2p',
        str(DICT_PATH),
        str(MODEL_PATH),
        '--num_jobs', str(num_jobs),
        '--temporary_directory', str(TEMP_DIR),
    ]
    if evaluate:
        cmd.append('--evaluate')
    if overwrite:
        cmd.append('--overwrite')

    print(f"Training g2p model...")
    print(f"  Dict:   {DICT_PATH}")
    print(f"  Output: {MODEL_PATH}")
    print()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train MFA g2p from Ulster dict.")
    parser.add_argument('--evaluate',  action='store_true', help="Hold out a subset to evaluate accuracy after training.")
    parser.add_argument('--num_jobs',  type=int, default=4, help="Number of parallel jobs (default: 4).")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing model if present.")
    args = parser.parse_args()
    train(args.evaluate, args.num_jobs, args.overwrite)


if __name__ == '__main__':
    main()
