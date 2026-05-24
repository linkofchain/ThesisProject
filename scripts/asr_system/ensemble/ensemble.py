"""
Span-wise confidence ensemble for Irish phoneme MDD.

Three monolingual CTC phoneme models (Irish, English, Russian) are combined
using frame-level confidence scores. Spans are defined by forced alignment of
the canonical transcript against the input audio using the Irish model.
The Russian model participates only when it detects a palatalized phoneme,
acting as a specialist for a feature absent from English phonology.
"""

import math
from dataclasses import dataclass

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

# some phonemes for Russian have equivalents in Irish written differently.
_RU_TO_GA_PAL = {
    'kʲ': 'c',
    'gʲ': 'ɟ',
    'sʲ': 'ʃ',
    'xʲ': 'ç',
    'ɣʲ': 'j',
}


def build_pal_set(ga_processor, ru_ipa_dict):
    """
    Build the set of Russian phones that gate Russian model participation.

    Includes Russian phones whose IPA matches an Irish palatalized phone
    (direct intersection), plus cross-system equivalences where the two
    systems use different IPA symbols for the same phonological contrast
    (e.g. Russian kʲ / Irish c, Russian sʲ / Irish ʃ).
    """
    ru_pal  = {x for x in ru_ipa_dict if 'ʲ' in x}
    ga_dict = ga_processor.tokenizer.get_vocab()
    ga_pal  = {x for x in ga_dict if 'ʲ' in x} | {'ʃ', 'c', 'ɟ', 'ç', 'j', 'ɲ'}

    cross_system = {ru for ru, ga in _RU_TO_GA_PAL.items()
                    if ru in ru_ipa_dict and ga in ga_dict}

    pal_set = (ga_pal & ru_pal) | cross_system
    print("pal_set:", pal_set)
    return pal_set


def is_palatalized(phone, pal_set):
    return phone in pal_set


# ---------------------------------------------------------------------------
# Frame-level confidence functions
# All take log-prob tensor of shape (T, V) and return confidence tensor (T,).
# ---------------------------------------------------------------------------

def frame_gibbs_entropy(emissions):
    """Gibbs (Shannon) entropy over log-prob distribution."""
    p = emissions.exp()
    # 0 * log(0) is defined as 0; nan_to_num converts the 0*(-inf)=NaN case
    return -(torch.nan_to_num(p * emissions, nan=0.0)).sum(dim=-1)


def frame_tsallis_entropy(emissions, alpha=.3):
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


def frame_tsallis_confidence(emissions, alpha=.3):
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

def _non_blank_frames(emission, filler_ids, start, end):
    """Return phonemic frames in [start, end), falling back to full span.

    Filters out any frame whose argmax token is in filler_ids (blank, [PAD],
    |, and other special tokens), so downstream helpers are not misled by
    frames dominated by non-phonemic tokens.
    """
    span   = emission[start:end]
    argmax = span.argmax(dim=-1)
    mask   = ~torch.isin(argmax, torch.tensor(sorted(filler_ids), device=argmax.device))
    non_blank = span[mask]
    return non_blank if non_blank.shape[0] > 0 else span


def has_phoneme_frame(emission, filler_ids, start, end):
    """True if any frame in [start, end) has a non-filler argmax.

    When False, the model predicts only blank/special tokens across the entire
    span — a deletion signal. Used to gate phone prediction vs. '<del>'.
    """
    span     = emission[start:end]
    argmax   = span.argmax(dim=-1)
    filler_t = torch.tensor(sorted(filler_ids), device=argmax.device)
    return not torch.isin(argmax, filler_t).all().item()


def span_confidence(emission, filler_ids, start, end, conf_func):
    """
    Scalar confidence for a phoneme span.

    Selects phonemic frames in [start, end) (falling back to full span), then
    renormalizes the log-prob distribution over phonemic tokens (all filler
    slots set to -inf) before applying conf_func. This ensures all confidence
    functions measure peakiness of the phone distribution rather than filler
    dominance: a blank-dominated span correctly yields low confidence.
    """
    frames      = _non_blank_frames(emission, filler_ids, start, end)
    no_filler   = frames.clone()
    filler_t    = torch.tensor(sorted(filler_ids), dtype=torch.long)
    no_filler[:, filler_t] = float('-inf')
    no_filler   = no_filler - torch.logsumexp(no_filler, dim=-1, keepdim=True)
    return conf_func(no_filler).max().item()


def span_peak_probs(emission, filler_ids, start, end):
    """
    Peak probability distribution over phonemic frames in [start, end).
    Shape: (V,). Element i is the max probability token i achieved across any
    phonemic frame.
    """
    frames = _non_blank_frames(emission, filler_ids, start, end)
    return frames.exp().max(dim=0).values


_SPECIAL_TOKENS = {'[PAD]', '[UNK]', '|', '<s>', '</s>'}


def best_phone_in_span(emission, idx2phone, filler_ids, start, end):
    """
    Predicted phoneme for a span: token with the highest peak log-prob across
    phonemic frames, with all filler tokens masked out.
    """
    frames   = _non_blank_frames(emission, filler_ids, start, end)
    peak_lp  = frames.max(dim=0).values.clone()
    filler_t = torch.tensor(sorted(filler_ids), dtype=torch.long)
    peak_lp[filler_t] = float('-inf')
    best_idx = peak_lp.argmax().item()
    return idx2phone.get(best_idx, f'<unk:{best_idx}>')


def decode_span(emission, idx2phone, filler_ids, start, end):
    """
    Canonical span decoder: returns '∅' if the model predicts no phonemic
    frame in [start, end), otherwise returns the best phone from
    best_phone_in_span. Use this wherever a span should be able to signal
    a deletion rather than being forced to produce a phone.
    """
    if not has_phoneme_frame(emission, filler_ids, start, end):
        return '∅'
    return best_phone_in_span(emission, idx2phone, filler_ids, start, end)


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


def span_pooled_confidence(emission, filler_ids, start, end,
                           canonical_phone, phone2idx, families):
    """
    Confidence for a span based on pooled family probability mass.

    Sums exp(log_prob) across all phones in the canonical phone's family
    (broad + slender variants) per phonemic frame, then returns the max
    over frames. This prevents entropy-based measures from penalising the
    Irish model when probability is legitimately split across the
    broad/slender axis (e.g. /lʲ/ and /l/).

    Falls back to max-prob confidence if no family indices are found.
    """
    frames = _non_blank_frames(emission, filler_ids, start, end)
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


def _compute_spans(ga_ctx, transcript, device='cpu', expand=0):
    """
    CTC forced alignment → per-phone frame spans.

    Returns list of (token_id, start, end) with end exclusive.

    When expand > 0, each span is widened symmetrically by up to `expand`
    frames on each side. Expansion is capped at the midpoint between adjacent
    spans so they never overlap. This gives each model more non-blank frames
    to draw confidence from when the raw forced alignment lands on a single
    blank-dominated frame.
    """
    tokenized = [ga_ctx.vocab[p] for p in transcript]
    aligned, scores = align(ga_ctx.emission.unsqueeze(0), tokenized, device)
    raw = F.merge_tokens(aligned, scores)

    if expand == 0:
        return [(sp.token, sp.start, sp.end) for sp in raw]

    T = ga_ctx.emission.shape[0]
    n = len(raw)
    result = []
    for i, sp in enumerate(raw):
        lo = (raw[i - 1].end + sp.start) // 2 if i > 0     else 0
        hi = (sp.end + raw[i + 1].start) // 2 if i < n - 1 else T
        result.append((sp.token, max(sp.start - expand, lo), min(sp.end + expand, hi)))
    return result


def _score_candidates(s, e, ga_ctx, en_ctx, ru_ctx, pal_set,
                      canonical_phone, conf_func, pool_ga, ga_families):
    """
    Score each model over frames [s, e) and return a candidates list.

    Irish confidence optionally uses broad/slender family pooling (pool_ga).
    Russian is included only when ru_ctx is provided and its top prediction
    in the span passes the palatalization gate.

    Returns list of (confidence, model_name, ModelCtx).
    """
    if pool_ga:
        ga_conf = span_pooled_confidence(
            ga_ctx.emission, ga_ctx.filler_ids, s, e,
            canonical_phone, ga_ctx.vocab, ga_families,
        )
    else:
        ga_conf = span_confidence(ga_ctx.emission, ga_ctx.filler_ids, s, e, conf_func)

    en_conf = span_confidence(en_ctx.emission, en_ctx.filler_ids, s, e, conf_func)
    candidates = [(ga_conf, 'ga', ga_ctx), (en_conf, 'en', en_ctx)]

    if ru_ctx is not None:
        ru_pred = decode_span(ru_ctx.emission, ru_ctx.idx2phone, ru_ctx.filler_ids, s, e)
        if is_palatalized(ru_pred, pal_set):
            ru_conf = span_confidence(ru_ctx.emission, ru_ctx.filler_ids, s, e, conf_func)
            candidates.append((ru_conf, 'ru', ru_ctx))

    return candidates


def _select_winner(candidates, selector):
    """
    Choose the winning model from a list of (conf, name, ctx) candidates.

    Priority order:
      1. Russian — when it fires the palatalization gate it always wins.
      2. LR selector — when a trained selector is provided it arbitrates ga/en.
      3. Argmax — fall back to whichever model has highest raw confidence.

    Returns the winning (conf, name, ctx) triple.
    """
    if any(name == 'ru' for _, name, _ in candidates):
        return next(c for c in candidates if c[1] == 'ru')
    if selector is not None:
        ga_conf = next(c[0] for c in candidates if c[1] == 'ga')
        en_conf = next(c[0] for c in candidates if c[1] == 'en')
        winner = selector.predict([[ga_conf, en_conf]])[0]
        return next(c for c in candidates if c[1] == winner)
    return max(candidates, key=lambda c: c[0])


# ---------------------------------------------------------------------------
# Span-wise ensemble
# ---------------------------------------------------------------------------

def get_emission(processor, model, waveform, device='cpu'):
    """Run a CTC model forward pass; returns log-softmax emissions (T, V)."""
    inputs = processor(waveform, sampling_rate=16000,
                       return_tensors='pt', padding=True)
    with torch.inference_mode():
        out = model(inputs.input_values.to(device),
                    attention_mask=inputs.attention_mask.to(device))
    return torch.nn.functional.log_softmax(out.logits[0], dim=-1)


@dataclass
class ModelCtx:
    """All per-model data needed for span-level inference."""
    emission:   torch.Tensor   # (T, V) log-softmax
    blank_id:   int            # CTC blank index — used only for forced alignment
    filler_ids: frozenset      # all non-phonemic token indices (blank + special tokens)
    idx2phone:  dict           # int -> str
    vocab:      dict           # str -> int


def _build_model_ctx(processor, model, waveform, device='cpu') -> ModelCtx:
    vocab = processor.tokenizer.get_vocab()
    return ModelCtx(
        emission=get_emission(processor, model, waveform, device),
        blank_id=processor.tokenizer.pad_token_id,
        filler_ids=frozenset(idx for token, idx in vocab.items() if token in _SPECIAL_TOKENS),
        idx2phone={v: k for k, v in vocab.items()},
        vocab=vocab,
    )


def _build_ru_ctx(ru_processor, ru_model, ga_processor, waveform, device='cpu'):
    """Build ModelCtx for the Russian model plus the palatalization gate set."""
    ru_ipa_dict = build_ru_ipa_dict(ru_processor)
    ru_vocab    = ru_processor.tokenizer.get_vocab()
    ctx = ModelCtx(
        emission=get_emission(ru_processor, ru_model, waveform, device),
        blank_id=ru_processor.tokenizer.pad_token_id,
        filler_ids=frozenset(idx for token, idx in ru_vocab.items() if token in _SPECIAL_TOKENS),
        idx2phone={v: k for k, v in ru_ipa_dict.items()},
        vocab=ru_ipa_dict,
    )
    return ctx, build_pal_set(ga_processor, ru_ipa_dict)


def spanwise_ensemble(waveform, transcript,
                      ga_processor, ga_model,
                      en_processor, en_model,
                      ru_processor=None, ru_model=None,
                      conf_func=frame_gibbs_confidence,
                      pool_ga=False,
                      selector=None,
                      expand=10,
                      device='cpu',
                      verbose=True):
    """
    Span-wise confidence ensemble over monolingual phoneme ASR models.

    Spans are defined by forced alignment of the canonical transcript against
    the input audio using the Irish model. For each span, the model with
    highest confidence (over non-blank frames) wins, and its predicted phoneme
    is taken from its own vocab.

    The Russian model is optional (pass ru_processor and ru_model to enable).
    It participates only on spans where it detects a palatalized phoneme.
    ru_ipa_dict and pal_set are derived automatically from the processors.

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

    selector: optional sklearn-compatible classifier with a .predict() method
        trained to arbitrate between ga and en. Accepts [[ga_conf, en_conf]]
        and returns ['ga'] or ['en']. When None, falls back to argmax.
        Russian always wins when it fires the palatalization gate, regardless
        of selector.

    expand: frames to grow each aligned span symmetrically on both sides
        (default 0 = raw forced alignment, typically 1-frame spans).
        Expansion is capped at the midpoint between adjacent spans so they
        never overlap. Useful when forced alignment lands on a blank-dominated
        frame and the confidence estimate needs more context.

    verbose=False: returns list of dicts {canonical, predicted, winner,
                   confidence, frames}
    verbose=True:  each dict also contains 'models' — per-model breakdown with
                   confidence scalar and predicted phone per non-blank span.
    """
    use_russian = ru_processor is not None and ru_model is not None

    ga_ctx = _build_model_ctx(ga_processor, ga_model, waveform, device)
    en_ctx = _build_model_ctx(en_processor, en_model, waveform, device)
    ru_ctx, pal_set = (
        _build_ru_ctx(ru_processor, ru_model, ga_processor, waveform, device)
        if use_russian else (None, frozenset())
    )

    ga_families = build_phoneme_families(ga_ctx.vocab) if pool_ga else None

    results = []
    for token, s, e in _compute_spans(ga_ctx, transcript, device, expand):
        canonical_phone = ga_ctx.idx2phone[token]

        candidates = _score_candidates(
            s, e, ga_ctx, en_ctx, ru_ctx, pal_set,
            canonical_phone, conf_func, pool_ga, ga_families,
        )
        best_conf, winner, win_ctx = _select_winner(candidates, selector)

        predicted_phone = decode_span(win_ctx.emission, win_ctx.idx2phone, win_ctx.filler_ids, s, e)
        if winner == 'ru' and predicted_phone != '∅':
            predicted_phone = _RU_TO_GA_PAL.get(predicted_phone, predicted_phone)

        entry = {
            'canonical': canonical_phone,
            'predicted': predicted_phone,
            'winner': winner,
            'confidence': round(best_conf, 4),
        }

        if verbose:
            entry['frames'] = (s, e)
            entry['models'] = {
                name: {
                    'confidence': round(conf, 4),
                    'predicted': decode_span(ctx.emission, ctx.idx2phone, ctx.filler_ids, s, e),
                    'peak_probs': span_peak_probs(ctx.emission, ctx.filler_ids, s, e),
                }
                for conf, name, ctx in candidates
            }

        results.append(entry)

    return results
