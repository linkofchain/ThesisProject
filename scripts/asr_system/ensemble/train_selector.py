"""
Train the logistic regression model selection block for the span-wise ensemble.

Loads a split of annotated records, runs ensemble inference with verbose=True
to capture per-span per-model confidence scores, then trains a logistic
regression to arbitrate between the Irish (ga) and English (en) models.

Labels are derived by aligning gold annotations to canonical phone positions
via the same SequenceMatcher logic used at eval time. Spans where neither or
both models match gold are skipped (ambiguous training signal).

The trained selector {'scaler': StandardScaler, 'lr': LogisticRegression} is
saved with joblib and can be loaded and passed to run_ensemble_inference:

    import joblib
    sel = joblib.load('models/ensemble/lr_selector.pkl')
    selector = lambda feat: sel['lr'].predict(sel['scaler'].transform(feat))
    run_ensemble_inference(..., selector=selector)

Usage:
    conda run -n thesis python scripts/asr_system/ensemble/train_selector.py \\
        --ga_model <path_or_hub_id> \\
        --en_model <path_or_hub_id> \\
        --audio_dir data/eval/segments/audio \\
        --n_train 100
"""

import argparse
import json
import pathlib
import random
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = pathlib.Path('.').resolve()
while not (PROJECT_ROOT / '.git').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.asr_system.ensemble.ensemble import (
    frame_gibbs_confidence,
    frame_prob_confidence,
    frame_tsallis_confidence,
)
from scripts.asr_system.ensemble.run_inference import run_ensemble_inference
from scripts.data_handling.collapse_phonemes import collapse_phones
from scripts.eval.eval import align_phones


def resolve_conf_func(name: str, alpha: float):
    if name == 'gibbs':
        return frame_gibbs_confidence
    if name == 'prob':
        return frame_prob_confidence
    if name == 'tsallis':
        from functools import partial
        return partial(frame_tsallis_confidence, alpha=alpha)
    raise ValueError(f"Unknown conf_func: {name!r}")

DEFAULT_OUTPUT      = PROJECT_ROOT / 'models' / 'ensemble' / 'lr_selector.pkl'
DEFAULT_SEGMENTS_DIR = PROJECT_ROOT / 'data' / 'eval' / 'segments'


def extract_features(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Extract (ga_conf, en_conf) feature vectors and winner labels from
    verbose inference records.

    Gold phones are aligned to canonical positions via align_phones so that
    insertions and deletions don't cause positional drift. Spans where neither
    or both models match the aligned gold phone are skipped.
    """
    X, y = [], []
    skipped = 0

    for record in records:
        canonical = collapse_phones(record['canonical'])
        gold      = collapse_phones(record['gold'])
        alignment = align_phones(canonical, gold)

        # One gold phone per canonical position; None where the speaker deleted.
        gold_per_canonical = [
            event['hypothesis']
            for event in alignment
            if event['op'] != 'insertion'
        ]

        for span, gold_phone in zip(record['span_details'], gold_per_canonical):
            if gold_phone is None:
                skipped += 1
                continue

            models     = span['models']
            ga_correct = models['ga']['predicted'] == gold_phone
            en_correct = models['en']['predicted'] == gold_phone

            if ga_correct == en_correct:
                skipped += 1
                continue

            X.append([models['ga']['confidence'], models['en']['confidence']])
            y.append('ga' if ga_correct else 'en')

    print(f"  Spans used for training : {len(X)}")
    print(f"  Skipped (tie/double-miss/deletion): {skipped}")
    return np.array(X), y


def train(args):
    manifest = pathlib.Path(args.segments_dir) / 'manifest.json'
    with open(manifest, encoding='utf-8') as f:
        records = json.load(f)
    random.seed(args.seed)
    random.shuffle(records)
    train_records = records[:args.n_train]
    print(f"Using {len(train_records)}/{len(records)} records for LR training "
          f"({len(records) - len(train_records)} reserved for eval)\n")

    print("Loading ga model...")
    ga_processor = Wav2Vec2Processor.from_pretrained(args.ga_model)
    ga_model     = Wav2Vec2ForCTC.from_pretrained(args.ga_model).eval()

    print("Loading en model...")
    en_processor = Wav2Vec2Processor.from_pretrained(args.en_model)
    en_model     = Wav2Vec2ForCTC.from_pretrained(args.en_model).eval()

    conf_func = resolve_conf_func(args.conf_func, args.tsallis_alpha)
    print(f"\nRunning inference on training split (conf_func={args.conf_func}"
          + (f", alpha={args.tsallis_alpha}" if args.conf_func == 'tsallis' else "")
          + ")...")
    train_out = run_ensemble_inference(
        train_records,
        ga_processor, ga_model,
        en_processor, en_model,
        conf_func=conf_func,
        pool_ga=args.pool_ga,
        verbose=True,
        audio_dir=args.audio_dir,
        device=args.device,
    )

    print("\nExtracting features...")
    X, y = extract_features(train_out)

    if len(X) == 0:
        raise ValueError("No usable training spans — check that verbose inference ran correctly.")
    if len(set(y)) < 2:
        raise ValueError(
            f"Only one class present: {set(y)}. "
            "Check gold annotations and model predictions."
        )

    class_counts = {c: y.count(c) for c in set(y)}
    print(f"\n  Class distribution: {class_counts}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr       = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_scaled, y)

    ga_idx = list(lr.classes_).index('ga')
    en_idx = list(lr.classes_).index('en')
    print(f"  LR coef  — ga_conf: {lr.coef_[0][ga_idx]:+.4f}, "
          f"en_conf: {lr.coef_[0][en_idx]:+.4f}")
    print(f"  Intercept: {lr.intercept_[0]:+.4f}")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'scaler': scaler, 'lr': lr}, out_path)
    print(f"\nSaved selector to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train LR model selector for the span-wise ensemble."
    )
    parser.add_argument('--ga_model',  required=True,
                        help="Path or HuggingFace ID for the Irish ASR model.")
    parser.add_argument('--en_model',  required=True,
                        help="Path or HuggingFace ID for the English ASR model.")
    parser.add_argument('--audio_dir', type=pathlib.Path, default=None,
                        help="Local directory of {audio_id}.wav files. "
                             "Falls back to GCS if absent.")
    parser.add_argument('--n_train',   type=int, default=100,
                        help="Annotated utterances to use for training (default: 100).")
    parser.add_argument('--seed',      type=int, default=42,
                        help="Random seed for train/eval split (default: 42).")
    parser.add_argument('--pool_ga',      action='store_true',
                        help="Use broad/slender family pooling for Irish confidence. "
                             "Should match your eval inference settings.")
    parser.add_argument('--conf_func',    default='gibbs',
                        choices=['gibbs', 'prob', 'tsallis'],
                        help="Confidence function to use (default: gibbs).")
    parser.add_argument('--tsallis_alpha', type=float, default=2.0,
                        help="Alpha for Tsallis entropy, only used when "
                             "--conf_func tsallis (default: 2.0).")
    parser.add_argument('--output',    type=pathlib.Path, default=DEFAULT_OUTPUT,
                        help=f"Output path for the trained selector "
                             f"(default: {DEFAULT_OUTPUT}).")
    parser.add_argument('--segments_dir', type=pathlib.Path, default=DEFAULT_SEGMENTS_DIR,
                        help=f"Directory containing manifest.json "
                             f"(default: {DEFAULT_SEGMENTS_DIR}).")
    parser.add_argument('--device',    default='cpu',
                        help="'cpu' or 'cuda' (default: cpu).")
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()