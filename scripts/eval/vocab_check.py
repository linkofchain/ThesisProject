"""
Check panphon feature-vector coverage for every phone that appears in:
  - the ASR model vocabulary  (vocab.json)
  - canonical sequences       (manifest.json)
  - gold annotation sequences (manifest.json)

Phones where panphon returns an empty vector cannot participate in
feature-weighted alignment without a prior mapping step.
"""

import json
import pathlib
import panphon

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

VOCAB_PATH    = PROJECT_ROOT / "data/synthetic/unpaired_training_set/vocab.json"
MANIFEST_PATH = PROJECT_ROOT / "data/eval/segments/manifest.json"

SKIP_TOKENS = {'|', '[UNK]', '[PAD]'}


def load_phones() -> dict[str, set[str]]:
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    vocab_phones = {k for k in vocab if k not in SKIP_TOKENS}

    canonical_phones, gold_phones = set(), set()
    with open(MANIFEST_PATH) as f:
        records = json.load(f)
    for r in records:
        canonical_phones.update(r.get("canonical", []))
        gold_phones.update(r.get("gold", []))

    return {
        "vocab":     vocab_phones,
        "canonical": canonical_phones,
        "gold":      gold_phones,
    }


def check_coverage(ft: panphon.FeatureTable, phones: set[str]) -> tuple[set[str], set[str]]:
    covered, missing = set(), set()
    for phone in phones:
        vec = ft.word_to_vector_list(phone, numeric=True)
        (covered if vec else missing).add(phone)
    return covered, missing


def main():
    ft = panphon.FeatureTable()
    inventories = load_phones()

    all_phones = set().union(*inventories.values())
    covered, missing = check_coverage(ft, all_phones)

    print(f"Total unique phones across all sets: {len(all_phones)}")
    print(f"  Covered by panphon : {len(covered)}")
    print(f"  Missing from panphon: {len(missing)}")

    if missing:
        print("\n--- Missing phones (need mapping before NW alignment) ---")
        for phone in sorted(missing):
            sources = [src for src, s in inventories.items() if phone in s]
            print(f"  {phone!r:12s}  found in: {', '.join(sources)}")

    # Phones in gold that are not in the ASR vocab (annotation-only phones)
    gold_only = inventories["gold"] - inventories["vocab"]
    if gold_only:
        print("\n--- Gold phones not in ASR vocab ---")
        for phone in sorted(gold_only):
            status = "covered" if phone in covered else "MISSING from panphon"
            print(f"  {phone!r:12s}  panphon: {status}")


if __name__ == "__main__":
    main()
