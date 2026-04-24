"""
Inference wrapper for the span-wise ensemble.

Takes records from fetch_annotations (keys: audio_id, canonical, gold)
and adds an 'asr' key: the ensemble's predicted phone list, one entry
per canonical phone position.

Audio is loaded from local disk if audio_dir is given and the file exists,
otherwise downloaded from GCS (gs://ls_eval_fleurs/audio/test/{audio_id}.wav).

Requires: pip install google-cloud-storage torchaudio
"""

import io
import pathlib

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as _ta_func

from scripts.asr_system.ensemble.ensemble import (
    build_ru_ipa_dict,
    build_pal_set,
    frame_gibbs_confidence,
    spanwise_ensemble,
)
from scripts.data_handling.collapse_phonemes import collapse_phones

import os
from dotenv import load_dotenv
load_dotenv()

def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise EnvironmentError(f"{var} is not set. Add it to your .env file.")
    return val

_AUDIO_GCS_BUCKET = _require_env("GCS_EVAL_BUCKET")
_AUDIO_GCS_PREFIX = "audio/test"


def _sf_load(source) -> tuple[np.ndarray, int]:
    """Load audio via soundfile. source may be a path or a file-like object."""
    data, sr = sf.read(source, dtype='float32', always_2d=False)
    return data, sr


def _to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    """Convert to mono float32 at 16 kHz."""
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = _ta_func.resample(tensor, sr, 16000)
        data = tensor.squeeze(0).numpy()
    return data


def _load_waveform(audio_id: str, audio_dir=None, gcs_client=None) -> np.ndarray:
    """Load audio as a 16 kHz mono float32 numpy array."""
    if audio_dir is not None:
        local_path = pathlib.Path(audio_dir) / f"{audio_id}.wav"
        if local_path.exists():
            data, sr = _sf_load(local_path)
            return _to_mono_16k(data, sr)

    if gcs_client is None:
        raise RuntimeError(
            f"Audio not found locally for {audio_id} and no GCS client available."
        )
    blob = gcs_client.bucket(_AUDIO_GCS_BUCKET).blob(f"{_AUDIO_GCS_PREFIX}/{audio_id}.wav")
    data, sr = _sf_load(io.BytesIO(blob.download_as_bytes()))
    return _to_mono_16k(data, sr)


def run_ensemble_inference(
    records: list[dict],
    ga_processor,
    ga_model,
    en_processor,
    en_model,
    ru_processor=None,
    ru_model=None,
    conf_func=frame_gibbs_confidence,
    pool_ga: bool = False,
    audio_dir: str | pathlib.Path | None = None,
    device: str = 'cpu',
) -> list[dict]:
    """
    Run span-wise ensemble inference over annotation records.

    Each input record (audio_id, canonical, gold) gains two new keys:
      'asr'          — predicted phone list, one per canonical position
      'span_details' — raw spanwise_ensemble output, useful for error analysis

    Parameters
    ----------
    records     : list of dicts from fetch_annotations
    ga_processor, ga_model : Irish phoneme ASR
    en_processor, en_model : English phoneme ASR
    ru_processor, ru_model : Russian phoneme ASR (optional)
    conf_func   : frame-level confidence function (default: Gibbs entropy)
    pool_ga     : if True, use broad/slender family pooling for Irish confidence
    audio_dir   : local directory of {audio_id}.wav files; falls back to GCS
    device      : 'cpu' or 'cuda'

    Returns
    -------
    New list of records with 'asr' and 'span_details' added.
    Does not mutate the input records.
    """
    ru_ipa_dict = build_ru_ipa_dict(ru_processor) if ru_processor else None
    pal_set     = build_pal_set(ga_processor, ru_ipa_dict) if ru_ipa_dict else None
    ga_dict     = ga_processor.tokenizer.get_vocab()

    # Create GCS client once, only if we might need it
    needs_gcs = audio_dir is None or not pathlib.Path(audio_dir).exists()
    gcs_client = None
    if needs_gcs:
        from google.cloud import storage
        gcs_client = storage.Client()

    output = []
    for i, record in enumerate(records):
        audio_id  = record['audio_id']
        canonical = record['canonical']

        print(f"[{i+1}/{len(records)}] {audio_id} ({len(canonical)} phones)...")

        # Collapse to ga model vocab for tokenization only — canonical in the
        # record is left unchanged so the eval notebook can normalise uniformly.
        collapsed_canonical = collapse_phones(canonical)
        oov = [p for p in collapsed_canonical if p not in ga_dict]
        if oov:
            raise ValueError(
                f"[{audio_id}] {len(oov)} phone(s) still not in ga vocab after "
                f"collapsing: {sorted(set(oov))}"
            )

        waveform     = _load_waveform(audio_id, audio_dir, gcs_client)
        span_results = spanwise_ensemble(
            waveform, collapsed_canonical,
            ga_processor, ga_model,
            en_processor, en_model,
            ru_processor=ru_processor, ru_model=ru_model,
            ru_ipa_dict=ru_ipa_dict, pal_set=pal_set,
            conf_func=conf_func,
            pool_ga=pool_ga,
            device=device,
        )

        asr = [s['predicted'] for s in span_results]

        result = dict(record)
        result['asr']          = asr
        result['span_details'] = span_results
        output.append(result)

    return output
