"""
An example for CTC forced alignment using `ctc-segmentation`. It can be efficiently applied to audio of an arbitrary length.
For model downloading, please refer to https://github.com/espnet/espnet?tab=readme-ov-file#ctc-segmentation-demo
https://huggingface.co/espnet/owsm_ctc_v3.1_1B/commit/24122a483435e62556d8ad9d9788bb9dc34a7b14
https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B
"""
import soundfile as sf
from espnet2.bin.s2t_ctc_align import CTCSegmentation
from espnet_model_zoo.downloader import ModelDownloader

# Download model first
d = ModelDownloader()
downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.2_ft_1B")

aligner = CTCSegmentation(
    **downloaded,
    fs=16000,
    batch_size=4,    # batched parallel decoding; reduce it if your GPU memory is smaller
    kaldi_style_text=True,
    time_stamps="auto",     # "auto" can be more accurate than "fixed" when converting token index to timestamp
    lang_sym="<eng>",
    task_sym="<asr>",
    context_len_in_secs=2,  # left and right context in buffered decoding
)

speech, rate = sf.read(
    "data/tutorials/shorttest.wav"
)
print(f"speech duration: {len(speech) / rate : .2f} seconds")
text = """
utt1 Hello
utt2 Do I need to do anything more?
utt3 I don't think so
"""

segments = aligner(speech, text)
print(segments)