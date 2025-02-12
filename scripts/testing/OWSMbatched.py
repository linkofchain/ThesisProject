"""https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B
"""
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.2_ft_1B",
    device="cuda",
    use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym='<eng>',
    task_sym='<asr>',
)

res = s2t.batch_decode(
    "audio.wav",    # a single audio (path or 1-D array/tensor) as input
    batch_size=16,
    context_len_in_secs=4,
)   # res is a single str, i.e., the predicted text without special tokens

res = s2t.batch_decode(
    ["audio1.wav", "audio2.wav", "audio3.wav"], # a list of audios as input
    batch_size=16,
    context_len_in_secs=4,
)   # res is a list of str

# Please check the code of `batch_decode`