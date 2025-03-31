#An example script to run short-form ASR/ST:
#```python
#https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B
#https://huggingface.co/espnet/owsm_ctc_v3.1_1B/commit/24122a483435e62556d8ad9d9788bb9dc34a7b14 
import librosa
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.2_ft_1B",
    device="cpu",       # gpu too small on local machine.
    generate_interctc_outputs=False,
    lang_sym='<eng>',
    task_sym='<asr>',
)

# NOTE: OWSM-CTC is trained on 16kHz audio with a fixed 30s duration. Please ensure your input has the correct sample rate; otherwise resample it to 16k before feeding it to the model
speech, rate = librosa.load("data/tutorials/shorttest.wav", sr=16000)
speech = librosa.util.fix_length(speech, size=(16000 * 30))

res = s2t(speech)[0]
print(res)
