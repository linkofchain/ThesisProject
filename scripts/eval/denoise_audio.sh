#!/usr/bin/env bash
# Batch-denoise all WAVs in a given audio directory using ffmpeg afftdn.
# Adjust NOISE_FLOOR and NOISE_TYPE after experimenting with individual files.
# Output goes to AUDIO_OUT (defaults to a sibling _denoised directory) so
# originals are never overwritten.
#
# Usage:
#   bash scripts/eval/denoise_audio.sh <input_dir> [output_dir]
#
# Examples:
#   bash scripts/eval/denoise_audio.sh data/eval/segments/audio
#   bash scripts/eval/denoise_audio.sh data/selector_train/segments/audio data/selector_train/segments/audio_denoised

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_dir> [output_dir]" >&2
    exit 1
fi

AUDIO_IN="${1%/}"
AUDIO_OUT="${2:-${AUDIO_IN}_denoised}"
NOISE_FLOOR=-50      # dB — lower = more conservative, higher = more aggressive
NOISE_TYPE="w"       # w = white noise; remove this option to use auto-detection

mkdir -p "$AUDIO_OUT"

total=$(find "$AUDIO_IN" -maxdepth 1 -name "*.wav" | wc -l)
count=0

for input in "$AUDIO_IN"/*.wav; do
    filename=$(basename "$input")
    output="$AUDIO_OUT/$filename"
    ffmpeg -y -i "$input" \
        -af "afftdn=nf=${NOISE_FLOOR}:nt=${NOISE_TYPE}" \
        "$output" -loglevel error
    count=$((count + 1))
    printf "\r  %d / %d  %s" "$count" "$total" "$filename"
done

echo ""
echo "Done. Denoised files written to: $AUDIO_OUT"
