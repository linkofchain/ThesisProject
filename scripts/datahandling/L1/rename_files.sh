# removes apostrophes from .wav files so zip can be uploaded to kaggle
audio_dir="../../../data/teanglann/audio/wav_files/"
# move file from old name to new name
for oldf in $audio_dir*"'"*; do
    newname=$(echo "$oldf" | tr -d "'")
    mv "$oldf" "$newname"
done