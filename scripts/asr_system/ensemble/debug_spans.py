"""
Minimal debug entry point for stepping through spanwise_ensemble.

Loads one record from the eval manifest, runs the full ensemble with
verbose=True, and prints per-span results. Set breakpoints anywhere in
ensemble.py and launch via the 'debug_spans' VSCode config.
"""

import json
import pathlib
import sys

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = pathlib.Path('.').resolve()
while not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.asr_system.ensemble.ensemble import (
    frame_gibbs_confidence,
    frame_prob_confidence,
    frame_tsallis_confidence,
    get_emission,
    spanwise_ensemble,
)
from scripts.data_handling.collapse_phonemes import collapse_phones

# ── config ────────────────────────────────────────────────────────────────────
GA_MODEL      = "duck-hug-567/xls-r-300m-gaphon-full-collapsed"
EN_MODEL      = "duck-hug-567/xls-r-300m-engphon-collapsed"
RU_MODEL_ID   = "snu-nia-12/wav2vec-large-xlsr-53_nia12_phone-nsu-ai-_russian"
SEGMENTS_DIR  = PROJECT_ROOT / "data" / "eval" / "segments"
RECORD_INDEX  = 0       # which record from the manifest to debug
POOL_GA       = True
DEVICE        = "cpu"
# ──────────────────────────────────────────────────────────────────────────────

manifest = SEGMENTS_DIR / "manifest.json"
with open(manifest, encoding="utf-8") as f:
    records = json.load(f)

record = records[RECORD_INDEX]
print(f"Record: {record['audio_id']}")
print(f"Canonical ({len(record['canonical'])}): {record['canonical']}")
print(f"Gold      ({len(record['gold'])}): {record['gold']}")
print()

print("Loading models...")
ga_processor = Wav2Vec2Processor.from_pretrained(GA_MODEL)
ga_model     = Wav2Vec2ForCTC.from_pretrained(GA_MODEL).eval()
en_processor = Wav2Vec2Processor.from_pretrained(EN_MODEL)
en_model     = Wav2Vec2ForCTC.from_pretrained(EN_MODEL).eval()

audio_path = SEGMENTS_DIR / "audio" / f"{record['audio_id']}.wav"
waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
if sr != 16000:
    import torchaudio.functional as F
    waveform = F.resample(torch.from_numpy(waveform).unsqueeze(0), sr, 16000).squeeze(0).numpy()

canonical = collapse_phones(record["canonical"])

print("Running spanwise_ensemble...")
results = spanwise_ensemble(
    waveform, canonical,
    ga_processor, ga_model,
    en_processor, en_model,
    conf_func=frame_prob_confidence,
    pool_ga=POOL_GA,
    device=DEVICE,
    verbose=True,
)

print(f"\n{'canonical':<12} {'predicted':<12} {'winner':<6} {'conf':<8} {'frames'}")
print("-" * 55)
for span in results:
    frames = span.get("frames", "—")
    width  = (frames[1] - frames[0]) if isinstance(frames, tuple) else "—"
    print(f"{span['canonical']:<12} {span['predicted']:<12} {span['winner']:<6} "
          f"{span['confidence']:<8.4f} {str(frames):<14} width={width}")

# ── Full frame-level view ─────────────────────────────────────────────────────
BAND  = 50   # frames per printed band
COL   = 5    # chars per frame column (phone truncated to COL-1, +1 space)
BLANK = '_' * (COL - 1) + ' '   # e.g. "____ "

def _col(phone):
    if not phone or phone in ('[PAD]', '[UNK]', '|', '<s>', '</s>'):
        return BLANK
    return phone[:COL - 1].ljust(COL)

print("\n\nFull frame-level argmax (__ = blank/PAD, span row shows aligned canonical)")

ga_em_full = get_emission(ga_processor, ga_model, waveform, DEVICE)
en_em_full = get_emission(en_processor, en_model, waveform, DEVICE)
ga_idx2phone = {v: k for k, v in ga_processor.tokenizer.get_vocab().items()}
en_idx2phone = {v: k for k, v in en_processor.tokenizer.get_vocab().items()}

frame_to_canonical = {}
for span in results:
    s, e = span["frames"]
    for f in range(s, e):
        frame_to_canonical[f] = span["canonical"]

T = ga_em_full.shape[0]
for band_start in range(0, T, BAND):
    band_end = min(T, band_start + BAND)
    fs = range(band_start, band_end)

    idx_row = ''.join(f'{f:<{COL}}' if f % 10 == 0 else ' ' * COL for f in fs)
    ga_row  = ''.join(_col(ga_idx2phone.get(ga_em_full[f].argmax().item())) for f in fs)
    en_row  = ''.join(_col(en_idx2phone.get(en_em_full[f].argmax().item())) for f in fs)
    sp_row  = ''.join(_col(frame_to_canonical.get(f, '')) for f in fs)

    print(f"\nf{band_start:04d}  {idx_row}")
    print(f"  ga:   {ga_row}")
    print(f"  en:   {en_row}")
    print(f"  span: {sp_row}")
