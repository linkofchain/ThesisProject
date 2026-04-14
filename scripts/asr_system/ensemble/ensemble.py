"""
Span-wise confidence ensemble for Irish phoneme MDD.

Three monolingual CTC phoneme models (Irish, English, Russian) are combined
using frame-level confidence scores. Spans are defined by forced alignment of
the canonical transcript against the input audio using the Irish model.
The Russian model participates only when it detects a palatalized phoneme,
acting as a specialist for a feature absent from English phonology.
"""

import math
from functools import partial

import torch
import torchaudio.functional as F


# ---------------------------------------------------------------------------
# Russian vocab mapping
# ---------------------------------------------------------------------------

RU_PHON2IPA = {
    'U0': 'ˈu', 'U': 'u',
    'O0': 'ˈo', 'O': 'o',
    'A0': 'ˈa', 'A': 'a',
    'E0': 'ˈe', 'E': 'e',
    'Y0': 'ˈɨ', 'Y': 'ɨ',
    'I0': 'ˈi', 'I': 'i',
    'K0': 'kʲ',  'KH0': 'xʲ', 'KH': 'x',  'K': 'k',
    'GH0': 'ɣʲ', 'GH': 'ɣ',  'G0': 'gʲ', 'G': 'g',
    'J0': 'j',
    'TSH0': 'tɕ', 'TSH': 'tʂ',
    'SH0': 'ɕː',  'SH': 'ʂ',
    'ZH0': 'ʑː',  'ZH': 'ʐ',
    'DZ0': 'dzʲ', 'DZ': 'dz',
    'DZH0': 'dʑhʲ', 'DZH': 'dʑh',
    'R0': 'rʲ',  'R': 'r',
    'T0': 'tʲ',  'T': 't',
    'TS0': 'tsʲ', 'TS': 'ts',
    'S0': 'sʲ',  'S': 's',
    'D0': 'dʲ',  'D': 'd',
    'Z0': 'zʲ',  'Z': 'z',
    'N0': 'nʲ',  'N': 'n',
    'L0': 'lʲ',  'L': 'ɬ',
    'P0': 'pʲ',  'P': 'p',
    'F0': 'fʲ',  'F': 'f',
    'B0': 'bʲ',  'B': 'b',
    'V0': 'vʲ',  'V': 'v',
    'M0': 'mʲ',  'M': 'm',
}


def build_ru_ipa_dict(ru_processor):
    """Build IPA-keyed vocab dict from a Russian model processor."""
    ru_dict = ru_processor.tokenizer.get_vocab()
    ru_ipa_dict = {}
    for key, value in ru_dict.items():
        ipa_key = RU_PHON2IPA.get(key, key)
        ru_ipa_dict[ipa_key] = value
    return ru_ipa_dict


def build_pal_set(ga_processor, ru_ipa_dict):
    """
    Build the set of palatalized phones present in both Irish and Russian vocabs.
    Used to gate Russian model participation to relevant spans.
    """
    ga_dict = ga_processor.tokenizer.get_vocab()
    ga_pal = {x for x in ga_dict if 'ʲ' in x}
    ru_pal = {x for x in ru_ipa_dict if 'ʲ' in x}
    return ga_pal & ru_pal


def is_palatalized(phone, pal_set):
    return phone in pal_set


# ---------------------------------------------------------------------------
# Frame-level confidence functions
# All take log-prob tensor of shape (T, V) and return confidence tensor (T,).
# ---------------------------------------------------------------------------

def frame_gibbs_entropy(emissions):
    """Gibbs (Shannon) entropy over log-prob distribution."""
    p = emissions.exp()
    return -(p * emissions).sum(dim=-1)


def frame_tsallis_entropy(emissions, alpha=1.0):
    """Tsallis entropy. Reduces to Gibbs as alpha -> 1."""
    if alpha >= 1.0:
        return frame_gibbs_entropy(emissions)
    p = emissions.exp()
    return (1.0 - (p ** alpha).sum(dim=-1)) / (alpha - 1.0)


def frame_gibbs_confidence(emissions):
    """Gibbs entropy-based confidence, normalised to [0, 1]."""
    H = frame_gibbs_entropy(emissions)
    V = emissions.size(-1)
    return 1.0 - (H / math.log(V))


def frame_tsallis_confidence(emissions, alpha=2.0):
    """Tsallis entropy-based confidence, normalised to [0, 1]."""
    H = frame_tsallis_entropy(emissions, alpha=alpha)
    V = emissions.size(-1)
    H_max = (1.0 - (1.0 / V) ** (alpha - 1.0)) / (alpha - 1.0)
    return 1.0 - (H / H_max)


def frame_prob_confidence(emissions):
    """Raw max-probability confidence. Baseline — ignores distribution shape."""
    return emissions.exp().max(dim=-1).values


# ---------------------------------------------------------------------------
# Span-level helpers
# ---------------------------------------------------------------------------

def _non_blank_frames(emission, blank_id, start, end):
    """Return non-blank frames in [start, end), falling back to full span."""
    span = emission[start:end]
    mask = span.argmax(dim=-1) != blank_id
    non_blank = span[mask]
    return non_blank if non_blank.shape[0] > 0 else span


def span_confidence(emission, blank_id, start, end, conf_func):
    """
    Scalar confidence for a phoneme span.
    Aggregates frame-level confidence as max over non-blank frames.
    """
    frames = _non_blank_frames(emission, blank_id, start, end)
    return conf_func(frames).max().item()


def best_phone_in_span(emission, idx2phone, blank_id, start, end):
    """
    Predicted phoneme for a span: argmax of mean log-probs over non-blank frames.
    """
    frames = _non_blank_frames(emission, blank_id, start, end)
    best_idx = frames.mean(dim=0).argmax().item()
    return idx2phone.get(best_idx, f'<unk:{best_idx}>')


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

def align(emission, tokens, device='cpu'):
    """CTC forced alignment via torchaudio. emission: (1, T, V)."""
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]
    return alignments, scores.exp()


# ---------------------------------------------------------------------------
# Span-wise ensemble
# ---------------------------------------------------------------------------

def spanwise_ensemble(waveform, transcript,
                      ga_processor, ga_model,
                      en_processor, en_model,
                      ru_processor=None, ru_model=None,
                      ru_ipa_dict=None, pal_set=None,
                      conf_func=frame_gibbs_confidence,
                      device='cpu'):
    """
    Span-wise confidence ensemble over monolingual phoneme ASR models.

    Spans are defined by forced alignment of the canonical transcript against
    the input audio using the Irish model. For each span, the model with
    highest confidence (over non-blank frames) wins, and its predicted phoneme
    is taken from its own vocab.

    The Russian model is optional and participates only when it detects a
    palatalized phoneme in a span (requires ru_ipa_dict and pal_set).

    conf_func options:
        frame_gibbs_confidence                          — default
        frame_prob_confidence                           — baseline
        partial(frame_tsallis_confidence, alpha=1.5)    — Tsallis (any alpha)

    Returns: list of dicts {canonical, predicted, winner, confidence}
    """
    use_russian = (ru_processor is not None and ru_model is not None
                   and ru_ipa_dict is not None and pal_set is not None)

    def get_emission(processor, model, waveform):
        inputs = processor(waveform, sampling_rate=16000,
                           return_tensors='pt', padding=True)
        with torch.inference_mode():
            out = model(inputs.input_values.to(device),
                        attention_mask=inputs.attention_mask.to(device))
        return torch.nn.functional.log_softmax(out.logits[0], dim=-1)

    ga_em = get_emission(ga_processor, ga_model, waveform)
    en_em = get_emission(en_processor, en_model, waveform)
    if use_russian:
        ru_em = get_emission(ru_processor, ru_model, waveform)
        ru_blank = ru_processor.tokenizer.pad_token_id
        ru_idx2phone = {v: k for k, v in ru_ipa_dict.items()}

    ga_blank = ga_processor.tokenizer.pad_token_id
    en_blank = en_processor.tokenizer.pad_token_id
    ga_dict = ga_processor.tokenizer.get_vocab()
    ga_idx2phone = {v: k for k, v in ga_dict.items()}
    en_idx2phone = {v: k for k, v in en_processor.tokenizer.get_vocab().items()}

    tokenized = [ga_dict[p] for p in transcript]
    aligned_tokens, alignment_scores = align(ga_em.unsqueeze(0), tokenized, device)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

    results = []
    for span in token_spans:
        canonical_phone = ga_idx2phone[span.token]
        s, e = span.start, span.end

        ga_conf = span_confidence(ga_em, ga_blank, s, e, conf_func)
        en_conf = span_confidence(en_em, en_blank, s, e, conf_func)
        candidates = [(ga_conf, 'ga', ga_em, ga_idx2phone, ga_blank),
                      (en_conf, 'en', en_em, en_idx2phone, en_blank)]

        if use_russian:
            ru_pred = best_phone_in_span(ru_em, ru_idx2phone, ru_blank, s, e)
            if is_palatalized(ru_pred, pal_set):
                ru_conf = span_confidence(ru_em, ru_blank, s, e, conf_func)
                candidates.append((ru_conf, 'ru', ru_em, ru_idx2phone, ru_blank))

        best_conf, winner, win_em, win_idx2phone, win_blank = max(
            candidates, key=lambda x: x[0]
        )
        predicted_phone = best_phone_in_span(win_em, win_idx2phone, win_blank, s, e)

        results.append({
            'canonical': canonical_phone,
            'predicted': predicted_phone,
            'winner': winner,
            'confidence': round(best_conf, 4),
        })

    return results
