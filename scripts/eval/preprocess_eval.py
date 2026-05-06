"""
One-time (or --refresh) preprocessing: slice long FLEURS eval audio into
word-level segments based on Label Studio annotations, then write a manifest
that pairs each slice with canonical and gold phone sequences.

Output (written to output_dir, default data/eval/segments/):
    audio/{audio_id}.wav   — one WAV per annotated segment
    manifest.json          — list[{audio_id, audio_path, canonical, gold}]

Requires: pip install google-cloud-storage soundfile
"""

import io
import json
import os
import sys, pathlib
from collections import defaultdict

import soundfile as sf
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise EnvironmentError(f"{var} is not set. Add it to your .env file.")
    return val

PROJECT_ROOT = pathlib.Path('.').resolve()
while not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANNOTATIONS_BUCKET = _require_env("GCS_ANNOTATIONS_BUCKET")
DEFAULT_OUTPUT_DIR = pathlib.Path(str(PROJECT_ROOT) + "/data/eval/segments")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/blob into (bucket_name, blob_name)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, _, blob = uri[5:].partition("/")
    return bucket, blob


def _audio_id_from_path(audio_path: str) -> str:
    filename = audio_path.rstrip("/").split("/")[-1]
    stem, _, _ = filename.rpartition(".")
    return stem or filename


def _parse_phone_list(text: str) -> list[str]:
    return [p for p in text.strip().split() if p]


def _parse_segments(result: list[dict]) -> list[dict]:
    """
    Group result items by their shared Label Studio id and extract
    (start, end, canonical, gold) for each word-level segment.
    Returns segments sorted by start time, skipping incomplete groups.
    """
    groups: dict[str, dict] = defaultdict(dict)
    for item in result:
        seg_id = item.get("id")
        fname = item.get("from_name")
        val = item.get("value", {})
        if fname == "labels":
            groups[seg_id]["start"] = val["start"]
            groups[seg_id]["end"] = val["end"]
        elif fname == "canonical":
            texts = val.get("text", [])
            if texts:
                groups[seg_id]["canonical"] = texts[0]
        elif fname == "gold":
            texts = val.get("text", [])
            if texts:
                groups[seg_id]["gold"] = texts[0]

    segments = []
    for seg_id, data in groups.items():
        required = ("start", "end", "canonical", "gold")
        if all(k in data for k in required):
            segments.append({
                "seg_id":    seg_id,
                "start":     data["start"],
                "end":       data["end"],
                "canonical": data["canonical"],
                "gold":      data["gold"],
            })
        else:
            missing = [k for k in required if k not in data]
            print(f"  [warn] segment {seg_id} missing fields: {missing}, skipping")

    return sorted(segments, key=lambda s: s["start"])


def _slice_audio(audio_bytes: bytes, start: float, end: float) -> bytes:
    """Return WAV bytes for the audio between start and end seconds."""
    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        end_frame = min(int(end * sr), f.frames)
        f.seek(start_frame)
        data = f.read(end_frame - start_frame, dtype="float32")

    out = io.BytesIO()
    sf.write(out, data, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()


def build_eval_dataset(
    annotations_bucket: str = ANNOTATIONS_BUCKET,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT_DIR,
    refresh: bool = False,
) -> list[dict]:
    """
    Fetch annotation JSONs from GCS, slice the corresponding audio files, and
    write sliced WAVs + a manifest to output_dir.

    Skips annotations without segment-style results (old whole-file format).
    Source audio files are cached in memory per task to avoid redundant downloads.

    Returns the manifest as a list of dicts with keys:
        audio_id, audio_path, canonical (list), gold (list)
    """
    output_dir = pathlib.Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    audio_dir = output_dir / "audio"

    if manifest_path.exists() and not refresh:
        print(f"Loading cached manifest from {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    client = storage.Client()
    records = []
    # Cache raw audio bytes per GCS URI to avoid re-downloading within one run
    audio_cache: dict[str, bytes] = {}

    for blob in client.list_blobs(annotations_bucket):
        raw = blob.download_as_text(encoding="utf-8")
        annotation = json.loads(raw)

        if annotation.get("was_cancelled"):
            continue

        segments = _parse_segments(annotation.get("result", []))
        if not segments:
            continue  # old whole-file format or unannotated

        task_data = annotation.get("task", {}).get("data", {})
        audio_uri = task_data.get("audio", "")
        if not audio_uri:
            continue

        base_audio_id = _audio_id_from_path(audio_uri)
        print(f"Processing {base_audio_id}: {len(segments)} segments")

        if audio_uri not in audio_cache:
            bucket_name, blob_name = _parse_gcs_uri(audio_uri)
            src_blob = client.bucket(bucket_name).blob(blob_name)
            audio_cache[audio_uri] = src_blob.download_as_bytes()

        audio_bytes = audio_cache[audio_uri]

        for seg in segments:
            audio_id = f"{base_audio_id}_{seg['seg_id']}"
            wav_path = audio_dir / f"{audio_id}.wav"

            if not wav_path.exists() or refresh:
                wav_bytes = _slice_audio(audio_bytes, seg["start"], seg["end"])
                wav_path.write_bytes(wav_bytes)

            records.append({
                "audio_id":   audio_id,
                "audio_path": str(wav_path.resolve()),
                "canonical":  _parse_phone_list(seg["canonical"]),
                "gold":       _parse_phone_list(seg["gold"]),
            })

    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} segment records to {manifest_path}")
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build word-level eval dataset from Label Studio annotations."
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write sliced WAVs and manifest.json"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch and reslice even if manifest already exists"
    )
    args = parser.parse_args()

    records = build_eval_dataset(output_dir=args.output_dir, refresh=args.refresh)
    print(f"\nDone: {len(records)} segments ready for evaluation")
