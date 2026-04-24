"""
Span-wise confidence ensemble for Irish phoneme MDD.

Three monolingual CTC phoneme models (Irish, English, Russian) are combined
using frame-level confidence scores. Spans are defined by forced alignment of
the canonical transcript against the input audio using the Irish model.
The Russian model participates only when it detects a palatalized phoneme,
acting as a specialist for a feature absent from English phonology.
"""

import math
from collections import defaultdict
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
    ru_pal = {x for x in ru_ipa_dict if 'ʲ' in x}
    ga_dict = ga_processor.tokenizer.get_vocab()
    ga_pal = {x for x in ga_dict if 'ʲ' in x}
    # add palatalized phones w/o diacritics
    unmarked_slender = {'ʃ','c','ɟ','ç','j','ɲ'}
    ga_pal = ga_pal | unmarked_slender

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


def span_mean_probs(emission, blank_id, start, end):
    """
    Mean probability distribution over non-blank frames in [start, end).
    Shape: (V,). Useful for inspecting distribution smoothness and overconfidence.
    """
    frames = _non_blank_frames(emission, blank_id, start, end)
    return frames.exp().mean(dim=0)


def best_phone_in_span(emission, idx2phone, blank_id, start, end):
    """
    Predicted phoneme for a span: argmax of mean log-probs over non-blank frames.
    """
    frames = _non_blank_frames(emission, blank_id, start, end)
    best_idx = frames.mean(dim=0).argmax().item()
    return idx2phone.get(best_idx, f'<unk:{best_idx}>')


# ---------------------------------------------------------------------------
# Phoneme family pooling
# ---------------------------------------------------------------------------

# Broad/slender pairs from the standard Irish consonant table.
# Regular pairs (e.g. p/pʲ, b/bʲ) differ only by ʲ; irregular pairs
# (e.g. s/ʃ, k/c, ɡ/ɟ) have phonemically distinct slender members that
# diacritic-stripping alone would not group together.
_BROAD_SLENDER_PAIRS = [
    ('p', 'pʲ'), ('b', 'bʲ'),           # labial stops
    ('f', 'fʲ'), ('w', 'vʲ'),            # labial fricative / approximant
    ('m', 'mʲ'),                          # labial nasal
    ('t', 'tʲ'), ('d', 'dʲ'),            # coronal stops
    ('s', 'ʃ'),                           # coronal fricative (irregular)
    ('n', 'nʲ'), ('l', 'lʲ'), ('ɾ', 'ɾʲ'),  # coronal sonorants
    ('k', 'c'),  ('ɡ', 'ɟ'),             # dorsal stops
    ('x', 'ç'),  ('ɣ', 'j'),             # dorsal fricatives / approximants
    ('ŋ', 'ɲ'),                           # dorsal nasal
]


def build_phoneme_families(ga_dict: dict) -> dict[str, frozenset]:
    """
    Build broad/slender phoneme families for the ga model vocab.

    Uses explicit Irish broad/slender pairs rather than diacritic stripping
    so that irregular pairs (s/ʃ, k/c, ɡ/ɟ, x/ç, ɣ/j, ŋ/ɲ) are handled
    correctly. Pairs where one member is absent from ga_dict are skipped.
    Phones not in any pair default to a singleton family.

    Returns: phone -> frozenset of family members present in ga_dict.
    """
    SPECIAL = {'[PAD]', '[UNK]', '|', '<s>', '</s>'}
    vocab = {p for p in ga_dict if p not in SPECIAL}

    families = {p: frozenset({p}) for p in vocab}

    for broad, slender in _BROAD_SLENDER_PAIRS:
        members = frozenset({p for p in (broad, slender) if p in vocab})
        if len(members) > 1:
            for p in members:
                families[p] = members

    return families


def span_pooled_confidence(emission, blank_id, start, end,
                           canonical_phone, phone2idx, families):
    """
    Confidence for a span based on pooled family probability mass.

    Sums exp(log_prob) across all phones in the canonical phone's family
    (broad + slender variants) per non-blank frame, then returns the max
    over frames. This prevents entropy-based measures from penalising the
    Irish model when probability is legitimately split across the
    broad/slender axis (e.g. /lʲ/ and /l/).

    Falls back to max-prob confidence if no family indices are found.
    """
    frames = _non_blank_frames(emission, blank_id, start, end)
    probs  = frames.exp()  # (T, V)

    family     = families.get(canonical_phone, frozenset({canonical_phone}))
    family_idx = [phone2idx[p] for p in family if p in phone2idx]

    if not family_idx:
        return probs.max(dim=-1).values.max().item()

    return probs[:, family_idx].sum(dim=-1).max().item()


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
                      pool_ga=False,
                      device='cpu',
                      verbose=False):
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

    pool_ga=True: replaces the Irish model's conf_func confidence with
        span_pooled_confidence, which sums probability mass across the
        canonical phone's broad/slender family before competing against
        other models. Addresses the case where the Irish model splits
        probability across /l/ and /lʲ/ and loses to a more peaked English
        distribution. Phone selection after winning is unchanged.

    verbose=False: returns list of dicts {canonical, predicted, winner, confidence}
    verbose=True:  each dict also contains 'models' — per-model breakdown with
                   confidence scalar, predicted phone, and mean_probs (V,) over
                   non-blank frames. mean_probs shows distribution shape directly,
                   useful for diagnosing smoothness and overconfidence.
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

    ga_families = build_phoneme_families(ga_dict) if pool_ga else None

    results = []
    for span in token_spans:
        canonical_phone = ga_idx2phone[span.token]
        s, e = span.start, span.end

        if pool_ga:
            ga_conf = span_pooled_confidence(
                ga_em, ga_blank, s, e, canonical_phone, ga_dict, ga_families
            )
        else:
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

        entry = {
            'canonical': canonical_phone,
            'predicted': predicted_phone,
            'winner': winner,
            'confidence': round(best_conf, 4),
        }

        if verbose:
            entry['models'] = {
                name: {
                    'confidence': round(conf, 4),
                    'predicted': best_phone_in_span(em, idx2phone, blank, s, e),
                    'mean_probs': span_mean_probs(em, blank, s, e),
                }
                for conf, name, em, idx2phone, blank in candidates
            }

        results.append(entry)

    return results
