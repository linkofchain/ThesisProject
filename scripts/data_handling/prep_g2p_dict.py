"""
Prepare the cleaned Teanglann Ulster pronunciation dictionary for MFA g2p training.

Reads irish_data_cleaned.csv (columns: file, phonemes), normalises phonemes
via collapse_phones, strips standalone suprasegmental tokens, deduplicates,
and writes a tab-separated word\tpronunciation file ready for:

    mfa train_g2p <output_path> models/mfa/ulster_g2p_trained.zip
"""

import argparse
import csv
import pathlib
import sys

PROJECT_ROOT = pathlib.Path('.').resolve()
while not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_handling.collapse_phonemes import collapse_phones

primary_stress   = 'ˈ'
secondary_stress = 'ˌ'
suprasegmentals  = {primary_stress, secondary_stress}
is_suprasegmental = lambda x: x in suprasegmentals

DEFAULT_INPUT  = PROJECT_ROOT / 'data' / 'teanglann' / 'g2p_dicts' / 'irish_data_cleaned.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'teanglann' / 'g2p_dicts' / 'ulster_for_g2p.txt'


def prepare(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    seen = set()
    skipped = 0
    written = 0

    with open(input_path, newline='', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        for row in reader:
            word = row['file'].removesuffix('.wav').strip()
            if not word:
                skipped += 1
                continue

            phone_list   = row['phonemes'].split()
            phone_list   = collapse_phones(phone_list)
            phone_list   = list(filter(lambda x: not is_suprasegmental(x), phone_list))

            if not phone_list:
                skipped += 1
                continue

            pronunciation = ' '.join(phone_list)
            key = (word, pronunciation)
            if key in seen:
                continue
            seen.add(key)

            fout.write(f"{word}\t{pronunciation}\n")
            written += 1

    print(f"Written: {written} entries")
    print(f"Skipped: {skipped} (empty word or empty phone sequence after normalisation)")
    print(f"Output:  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prep Ulster g2p dict for MFA training.")
    parser.add_argument('--input',  type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == '__main__':
    main()
