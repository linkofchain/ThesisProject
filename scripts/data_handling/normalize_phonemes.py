"""
Normalize phoneme transcriptions in a HuggingFace dataset by stripping
unwanted diacritics from the phonetic column, using the collapsing rules
defined in scripts/eval/collapse_phonemes.py.

Usage:
    python scripts/data_handling/normalize_phonemes.py \
        --input  data/synthetic/paired \
        --output data/synthetic/paired_normalized \
        [--column phonetic]
"""

import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path('.').resolve()
while not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk
from scripts.eval.collapse_phonemes import collapse_phones


def normalize_row(row: dict, column: str) -> dict:
    phones = row[column].split(' ')
    row[column] = ' '.join(collapse_phones(phones))
    return row


def main():
    parser = argparse.ArgumentParser(description="Normalize phoneme diacritics in a dataset.")
    parser.add_argument('--input',  required=True, help="Path to the input dataset directory.")
    parser.add_argument('--output', required=True, help="Path to write the normalized dataset.")
    parser.add_argument('--column', default='phonetic', help="Column containing space-separated phonemes (default: phonetic).")
    args = parser.parse_args()

    input_path  = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    print(f"Loading dataset from {input_path} ...")
    ds = load_from_disk(str(input_path))

    print(f"Normalizing column '{args.column}' ...")
    ds = ds.map(lambda row: normalize_row(row, args.column))

    print(f"Saving to {output_path} ...")
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))

    # Quick sanity check — show one example before/after
    original = load_from_disk(str(input_path))[0][args.column]
    print("\nDone.")
    print(f"  Before: {original}")
    print(f"  After:  {ds[0][args.column]}")


if __name__ == '__main__':
    main()
