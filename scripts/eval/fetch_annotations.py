"""
Fetch completed Label Studio annotations from GCS and pair them with
canonical phoneme sequences from the FLEURS test CSV.

Returns a list of records ready for evaluate_mispronunciation_detection:
    [
        {
            'audio_id':   '10045899353945045473',
            'canonical':  ['i', 'sˠ', 'eː', ...],
            'gold':       ['i', 'sˠ', 'eː', ...],
        },
        ...
    ]

Add more annotation files to the bucket and re-run — they are discovered
automatically. Pass refresh=True to re-fetch from GCS and overwrite the cache.

Requires: pip install google-cloud-storage
"""

import csv
import io
import json
import os
import pathlib

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise EnvironmentError(f"{var} is not set. Add it to your .env file.")
    return val

ANNOTATIONS_BUCKET = _require_env("GCS_ANNOTATIONS_BUCKET")
EVAL_BUCKET        = _require_env("GCS_EVAL_BUCKET")
CANONICAL_CSV_PATH = "test.csv"

# Label Studio textarea field names used across annotation tasks
_IPA_FIELD_NAMES = {"transcription", "ipa"}


def _parse_phones(ipa_string: str) -> list[str]:
    """Split a space-separated IPA string into a list of phone tokens."""
    return [p for p in ipa_string.strip().split() if p]


def _extract_ipa(result: list[dict]) -> str | None:
    """
    Pull the IPA string from a Label Studio result list.
    Handles both 'transcription' and 'ipa' field names.
    Returns None if no matching field is found.
    """
    for item in result:
        if item.get("from_name") in _IPA_FIELD_NAMES:
            texts = item.get("value", {}).get("text", [])
            if texts:
                return texts[0]
    return None


def _audio_id_from_path(audio_path: str) -> str:
    """Extract the bare filename stem from a GCS or local audio path."""
    filename = audio_path.rstrip("/").split("/")[-1]
    stem, _, _ = filename.rpartition(".")
    return stem or filename


def _load_canonical_index(client: storage.Client) -> dict[str, list[str]]:
    """
    Download test.csv from the eval bucket and build a dict:
        audio_id -> canonical phone list

    Canonical phones are stored as a space-separated string in the
    'phonetic' column; [UNK] tokens are dropped.
    """
    bucket = client.bucket(EVAL_BUCKET)
    blob   = bucket.blob(CANONICAL_CSV_PATH)
    content = blob.download_as_text(encoding="utf-8")

    index: dict[str, list[str]] = {}
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        gcs_url  = row.get("gcs_url", "")
        audio_id = _audio_id_from_path(gcs_url) if gcs_url else None
        phonetic = row.get("phonetic", "")
        if audio_id and phonetic:
            phones = [p for p in phonetic.split() if p != "[UNK]"]
            index[audio_id] = phones

    return index


def _fetch_from_gcs(annotations_bucket: str) -> list[dict]:
    client = storage.Client()
    canonical_index = _load_canonical_index(client)

    records = []
    for blob in client.list_blobs(annotations_bucket):
        raw = blob.download_as_text(encoding="utf-8")
        annotation = json.loads(raw)

        if annotation.get("was_cancelled"):
            continue

        result = annotation.get("result", [])
        ipa_string = _extract_ipa(result)
        if not ipa_string:
            continue

        audio_path = annotation.get("task", {}).get("data", {}).get("audio", "")
        audio_id   = _audio_id_from_path(audio_path)

        canonical = canonical_index.get(audio_id)
        if canonical is None:
            continue

        records.append({
            "audio_id":  audio_id,
            "canonical": canonical,
            "gold":      _parse_phones(ipa_string),
        })

    return records


def fetch_annotations(
    annotations_bucket: str = ANNOTATIONS_BUCKET,
    cache_path: str | pathlib.Path | None = None,
    refresh: bool = False,
) -> list[dict]:
    """
    Return completed annotations paired with canonical phone sequences.

    Parameters
    ----------
    annotations_bucket : str
        GCS bucket containing Label Studio annotation JSON files.
    cache_path : path-like, optional
        If given, load from this JSON file when it exists (unless refresh=True),
        and write to it after fetching from GCS.
    refresh : bool
        Force a GCS fetch even if the cache file exists.

    Returns
    -------
    list of dicts with keys: audio_id, canonical, gold
    """
    if cache_path is not None:
        cache_path = pathlib.Path(cache_path)

    if cache_path and cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    records = _fetch_from_gcs(annotations_bucket)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    return records


if __name__ == "__main__":
    records = fetch_annotations(cache_path="data/eval/annotations_cache.json", refresh=True) # set refresh to true for full eval
    print(f"Fetched {len(records)} annotated samples")
    for r in records:
        print(f"  {r['audio_id']}: {len(r['canonical'])} canonical, {len(r['gold'])} gold phones")
