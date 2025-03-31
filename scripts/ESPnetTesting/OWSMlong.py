"""
https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B
https://huggingface.co/espnet/owsm_ctc_v3.1_1B/commit/24122a483435e62556d8ad9d9788bb9dc34a7b14
An example script to run long-form ASR:
"""
import soundfile as sf
import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

context_len_in_secs = 4   # left and right context when doing buffered inference
batch_size = 4   # depends on the GPU memory
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.2_ft_1B",
    device='cpu',#'cuda' if torch.cuda.is_available() else 'cpu',
    generate_interctc_outputs=False,
    lang_sym='<eng>',
    task_sym='<asr>',
)

speech, rate = sf.read(
    "data/tutorials/shorttest.wav"
)

text = s2t.decode_long_batched_buffered(
    speech,
    batch_size=batch_size,
    context_len_in_secs=context_len_in_secs,
)
print(text)