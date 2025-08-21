import os, torchaudio

def resample(audio_folder_path, output_path):
    
    for file_path in os.listdir(audio_folder_path):
        waveform, orig_sample_rate = torchaudio.load(audio_folder_path+file_path)
        new_waveform = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=16000
            )(waveform)
    
        torchaudio.save(output_path+file_path[:-3]+"wav", 
                    new_waveform, 
                    16000)

