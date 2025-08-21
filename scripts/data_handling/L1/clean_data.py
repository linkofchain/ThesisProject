import pandas as pd
import os
from pathlib import Path

proj_Root = Path().resolve()
audio_path = proj_Root / "data" / "teanglann" / "audio" / "wav_files"
data_path = proj_Root / "data" / "teanglann" / "irish_data.csv"
out_path = data_path.parent / "irish_data_cleaned.csv"

# handle apostrophes
df = pd.read_csv(data_path)
df['file'] = df['file'].str.replace("'", "")

# handle file duplicates that only differ by capitalization
dups = df[df['file'].str.lower().duplicated()]
df_cleaned = df.drop(dups.index)

# save
df_cleaned.to_csv(out_path, index=False)

# remove duplicate audio files
files = os.listdir(audio_path)

for file in dups['file']:
    file_path = os.path.join(audio_path, file)
    if os.path.exists(file_path):
        os.remove(file_path) 