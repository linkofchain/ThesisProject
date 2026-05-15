from difflib import SequenceMatcher
from scipy.stats import binom as _binom
import panphon.distance as _pd

# Phones panphon can't parse natively; mapped to the nearest parseable IPA phone
# for feature-vector lookup only — originals are preserved in alignment output.
_PHONE_NORM = {
    'g':  'ɡ',    # ASCII g → IPA ɡ (U+0261)
    'ʤ':  'd͡ʒ',  # dʒ ligature → tie-bar form
    'ʧ':  't͡ʃ',  # tʃ ligature → tie-bar form
    'ɚ':  'ə',    # r-colored schwa → plain schwa
}
_dst = _pd.Distance()
_pair_score_cache: dict[tuple[str, str], float] = {}


def _pair_score(p1: str, p2: str) -> float:
    """Phonetic similarity score in [0, 1]: 1.0 = identical, 0.0 = maximally different."""
    key = (p1, p2)
    if key not in _pair_score_cache:
        n1 = _PHONE_NORM.get(p1, p1)
        n2 = _PHONE_NORM.get(p2, p2)
        _pair_score_cache[key] = 1.0 if n1 == n2 else 1.0 - _dst.feature_edit_distance(n1, n2)
    return _pair_score_cache[key]


def greedy_ctc_phones(processor, model, audio_array, device='cpu'):
    """CTC greedy decode returning individual phones, not word-level strings."""
    import torch
    inputs = processor(audio_array, sampling_rate=16000,
                       return_tensors='pt', padding=True)
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device),
                       attention_mask=inputs.attention_mask.to(device)).logits
    predicted_ids = logits[0].argmax(dim=-1).tolist()

    vocab       = processor.tokenizer.get_vocab()
    blank_id    = processor.tokenizer.pad_token_id
    word_delim  = processor.tokenizer.word_delimiter_token
    idx2phone   = {v: k for k, v in vocab.items()}

    phones, prev_id = [], None
    for tok_id in predicted_ids:
        if tok_id != blank_id and tok_id != prev_id:
            phone = idx2phone.get(tok_id, '')
            if phone and phone != word_delim:
                phones.append(phone)
        prev_id = tok_id
    return phones


def forced_ctc_phones(processor, model, audio_array, canonical, device='cpu', expand=10):
    """
    Forced-alignment CTC decode: one predicted phone per canonical slot.

    Constrains the model's emissions to the canonical transcript via torchaudio
    forced alignment, then selects the peak token per span (same logic as the
    ensemble's best_phone_in_span). Output length always equals len(canonical),
    so evaluation is directly position-comparable without a secondary difflib
    alignment step.

    canonical must already be in the model's vocab (apply collapse_phones first).
    """
    from scripts.asr_system.ensemble.ensemble import (
        _build_model_ctx, _compute_spans, best_phone_in_span,
    )

    ctx   = _build_model_ctx(processor, model, audio_array, device)
    spans = _compute_spans(ctx, canonical, device, expand=expand)

    return [best_phone_in_span(ctx.emission, ctx.idx2phone, ctx.filler_ids, s, e)
            for _, s, e in spans]


def show_alignment(record, key='asr', col_sep=' '):
    """
    Print a two-row grid of canonical vs predicted phones.

    For ensemble records with 'span_details', uses the exact 1-1 span mapping.
    For pipeline records (no span_details), falls back to difflib.

    Example output:
        audio_id : 10045899353945045473
        can : i  sˠ  *   l  *   ...
        asr : i  sˠ  eː  l  tʲ  ...
    Positions where canonical == predicted show the phone; mismatches show *.
    Deletions show — in the asr row; insertions show — in the can row.
    """
    audio_id = record.get('audio_id', '?')

    if 'span_details' in record:
        pairs = [(s['canonical'], s['predicted']) for s in record['span_details']]
    else:
        hyp = record.get(key, [])
        can = record['canonical']
        alignment = align_phones(can, hyp)
        pairs = [
            (ev['canonical'] or '—', ev['hypothesis'] or '—')
            for ev in alignment
        ]

    col_w = [max(len(c), len(p), 1) for c, p in pairs]

    can_row, asr_row = [], []
    for (c, p), w in zip(pairs, col_w):
        can_row.append(('*' if c != p else c).ljust(w))
        asr_row.append(p.ljust(w))

    sep = col_sep
    print(f"audio_id : {audio_id}")
    print(f"can : {sep.join(can_row)}")
    print(f"asr : {sep.join(asr_row)}")


def show_alignment_html(record, key='asr'):
    """
    Render a two-row alignment table as inline HTML in a Jupyter notebook.
    Columns where canonical != predicted are highlighted with an orange background.
    Deletions show — in the asr row; insertions show — in the can row.
    """
    from IPython.display import display, HTML

    audio_id = record.get('audio_id', '?')

    if 'span_details' in record:
        pairs = [(s['canonical'], s['predicted']) for s in record['span_details']]
    else:
        hyp = record.get(key, [])
        can = record['canonical']
        alignment = align_phones(can, hyp)
        pairs = [
            (ev['canonical'] or '—', ev['hypothesis'] or '—')
            for ev in alignment
        ]

    MATCH_STYLE    = 'padding:3px 8px; font-family:monospace; border:1px solid #ddd;'
    MISMATCH_STYLE = 'padding:3px 8px; font-family:monospace; border:1px solid #ddd; background:#ffe0cc;'
    LABEL_STYLE    = 'padding:3px 8px; font-family:monospace; font-weight:bold; background:#f5f5f5; border:1px solid #ddd;'

    can_cells = [f'<td style="{LABEL_STYLE}">can</td>']
    asr_cells = [f'<td style="{LABEL_STYLE}">asr</td>']
    for c, p in pairs:
        style = MISMATCH_STYLE if c != p else MATCH_STYLE
        can_cells.append(f'<td style="{style}">{c}</td>')
        asr_cells.append(f'<td style="{style}">{p}</td>')

    html = (
        f'<div style="font-family:monospace; margin-bottom:8px;">'
        f'<strong>audio_id:</strong> {audio_id}</div>'
        f'<table style="border-collapse:collapse; margin-bottom:16px;">'
        f'<tr>{"".join(can_cells)}</tr>'
        f'<tr>{"".join(asr_cells)}</tr>'
        f'</table>'
    )
    display(HTML(html))


# Ground truth label per canonical phone position.
# True = mispronounced, False = correct, None = insertion (no canonical slot)
OP_LABEL = {'match': False, 'substitution': True, 'deletion': True, 'insertion': None}

# Direction constants for the NW traceback pointer matrix
_DIAGONAL, _FROM_ABOVE, _FROM_LEFT = 0, 1, 2


def _nw_traceback(seq1, seq2, direction_matrix):
    """
    Walk direction_matrix from the bottom-right corner back to (0, 0),
    converting each step into a labelled alignment event.
    Returns events in forward (left-to-right) order.
    """
    events = []
    i, j = len(seq1), len(seq2)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and direction_matrix[i][j] == _DIAGONAL:
            norm1 = _PHONE_NORM.get(seq1[i-1], seq1[i-1])
            norm2 = _PHONE_NORM.get(seq2[j-1], seq2[j-1])
            op = 'match' if norm1 == norm2 else 'substitution'
            events.append({'canonical': seq1[i-1], 'hypothesis': seq2[j-1], 'op': op})
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or direction_matrix[i][j] == _FROM_ABOVE):
            events.append({'canonical': seq1[i-1], 'hypothesis': None, 'op': 'deletion'})
            i -= 1
        else:
            events.append({'canonical': None, 'hypothesis': seq2[j-1], 'op': 'insertion'})
            j -= 1
    return list(reversed(events))


def _nw_align(seq1: list[str], seq2: list[str], gap: float = 1.0) -> list[dict]:
    """
    see https://github.com/ATalhaTimur/Needleman-Wunsch-Algorithm
    see https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    see https://medium.com/@nandiniumbarkar/needleman-wunsch-algorithm-7bba68b510db

    Needleman-Wunsch global alignment using panphon feature-weighted pair scores.
    Pair score = 1 - feature_edit_distance, in [0, 1] (1.0 = identical phones).
    Gap penalty = -gap (default -1.0), subtracted for each insertion or deletion.
    Traceback pointers are stored explicitly to avoid floating-point equality issues.
    """
    n_canonical, n_hypothesis = len(seq1), len(seq2)

    # step 1: instantiate the score and direction_matrix matrices from string lengths =1
    # score_matrix[i][j] = minimum total cost to align seq1[:i] with seq2[:j]
    score_matrix  = [[0.0]  * (n_hypothesis + 1) for _ in range(n_canonical + 1)]
    # direction_matrix[i][j] = which of the three moves produced score_matrix[i][j]
    direction_matrix = [[None] * (n_hypothesis + 1) for _ in range(n_canonical + 1)]

    # step 2: border cells — aligning i phones against nothing costs i gaps, and vice versa
    for i in range(1, n_canonical + 1):
        score_matrix[i][0]     = -i * gap
        direction_matrix[i][0] = _FROM_ABOVE
    for j in range(1, n_hypothesis + 1):
        score_matrix[0][j]     = -j * gap
        direction_matrix[0][j] = _FROM_LEFT

    # step 3: fill via the standard NW recurrence (score maximization):
    # score[i,j] = max( score[i-1,j-1] + pair_score(seq1[i], seq2[j]),  # pair (diagonal)
    #                   score[i-1,j]   - gap,                             # delete seq1[i]
    #                   score[i,j-1]   - gap )                            # insert seq2[j]
    for i in range(1, n_canonical + 1):
        for j in range(1, n_hypothesis + 1):
            pair_phones   = score_matrix[i-1][j-1] + _pair_score(seq1[i-1], seq2[j-1])
            delete_seq1_i = score_matrix[i-1][j]   - gap
            insert_seq2_j = score_matrix[i][j-1]   - gap
            best = max(pair_phones, delete_seq1_i, insert_seq2_j)
            score_matrix[i][j] = best
            if best == pair_phones:
                direction_matrix[i][j] = _DIAGONAL
            elif best == delete_seq1_i:
                direction_matrix[i][j] = _FROM_ABOVE
            else:
                direction_matrix[i][j] = _FROM_LEFT

    return _nw_traceback(seq1, seq2, direction_matrix)


def align_phones(canonical: list[str], hypothesis: list[str]) -> list[dict]:
    """
    Align a hypothesis phone sequence to canonical using Needleman-Wunsch with
    panphon feature-weighted substitution costs. Phonetically similar phones
    (e.g. t/tʲ) get lower substitution cost than dissimilar ones, producing
    more plausible alignments than equal-weight string edit distance.

    Returns one dict per alignment event with keys: canonical, hypothesis, op.
    Phones not natively parseable by panphon are normalized via _PHONE_NORM
    for cost computation only; originals are preserved in the output dicts.
    """
    return _nw_align(canonical, hypothesis)


def phone_error_rate(canonical: list[str], hypothesis: list[str]) -> float:
    """PER = (substitutions + deletions + insertions) / len(canonical)."""
    if not canonical:
        return float('nan')
    errors = 0
    sm = SequenceMatcher(None, canonical, hypothesis, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'replace':
            errors += max(i2 - i1, j2 - j1)
        elif op == 'delete':
            errors += i2 - i1
        elif op == 'insert':
            errors += j2 - j1
    return errors / len(canonical)


def get_canonical_labels(canonical: list[str], hypothesis: list[str]) -> list[bool]:
    """
    For each canonical phone position, return True (mispronounced) or False (correct).
    Insertions (hypothesis phones with no canonical counterpart) are discarded.
    Length of result always equals len(canonical).
    """
    alignment = align_phones(canonical, hypothesis)
    return [
        OP_LABEL[event['op']]
        for event in alignment
        if event['op'] != 'insertion'
    ]


def evaluate_mispronunciation_detection(
    canonical: list[str],
    gold: list[str],
    asr: list[str],
) -> dict:
    """
    Compare ASR mispronunciation predictions against gold-standard ground truth.

    asr must be forced-aligned to canonical (1:1 correspondence, same length).
    gold is aligned to canonical via NW feature-weighted alignment.

    For each canonical phone position:
      - gt_label:   True if gold differs from canonical (actual mispronunciation)
      - pred_label: True if asr[i] != canonical[i] (predicted mispronunciation)

    Gold insertions (epenthetic phones with no canonical slot) are counted
    separately. asr_insertions is always 0 by forced-alignment guarantee.

    Returns:
      TR, FA, FR, TA and derived metrics:
        precision, recall, f1,
        false_acceptance_rate (FAR = FA / (TR+FA)),
        false_rejection_rate  (FRR = FR / (FR+TA)),
        per (Phoneme Error Rate of ASR output vs canonical),
        gold_insertions, asr_insertions,
        TR_diag — TRs where ASR phone matches gold phone (correct diagnosis),
        diagnostic_accuracy (TR_diag / TR)
    """
    if len(asr) != len(canonical):
        raise ValueError(
            f"asr length {len(asr)} != canonical length {len(canonical)}. "
            "asr must come from forced alignment."
        )

    gt_alignment = align_phones(canonical, gold)
    gold_insertions = sum(1 for ev in gt_alignment if ev['op'] == 'insertion')

    gt_labels   = [OP_LABEL[ev['op']] for ev in gt_alignment if ev['op'] != 'insertion']
    gt_phones   = [ev['hypothesis']   for ev in gt_alignment if ev['op'] != 'insertion']
    pred_labels = [c != a for c, a in zip(canonical, asr)]
    pred_phones = list(asr)

    TR = FA = FR = TA = TR_diag = 0
    for gt, pred, gt_ph, pred_ph in zip(gt_labels, pred_labels, gt_phones, pred_phones):
        if     gt and     pred:
            TR += 1
            if gt_ph == pred_ph:
                TR_diag += 1
        elif   gt and not pred: FA += 1
        elif not gt and   pred: FR += 1
        else:                   TA += 1

    precision    = TR / (TR + FR) if (TR + FR) > 0 else float('nan')
    recall       = TR / (TR + FA) if (TR + FA) > 0 else float('nan')
    f1           = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else float('nan'))
    far          = FA / (TR + FA) if (TR + FA) > 0 else float('nan')
    frr          = FR / (FR + TA) if (FR + TA) > 0 else float('nan')
    per          = phone_error_rate(canonical, asr)
    diag_acc     = TR_diag / TR  if TR > 0 else float('nan')
    diag_err     = 1 - diag_acc if TR > 0 else float('nan')

    return {
        'TR': TR, 'FA': FA, 'FR': FR, 'TA': TA,
        'precision':               precision,
        'recall':                  recall,
        'f1':                      f1,
        'false_acceptance_rate':   far,
        'false_rejection_rate':    frr,
        'per':                     per,
        'gold_insertions':         gold_insertions,
        'asr_insertions':          0,
        'TR_diag':                 TR_diag,
        'diagnostic_accuracy':     diag_acc,
        'diagnostic_error_rate':   diag_err,
    }


def mcnemar_mdd(records_a, records_b):
    """
    McNemar's test comparing two MDD systems on the same utterances.

    At each canonical phone position, a system is "correct" when its
    mispronunciation decision (True/False) matches the gold label.
    The test asks: does system A fix more of system B's mistakes than
    it introduces new ones?

    Uses the exact binomial form, which is appropriate for small disagreement
    counts (b+c). Phone positions within an utterance are not strictly independent; 
    treat p-values as approximate.

    Parameters
    ----------
    records_a, records_b : list of dicts
        Must be aligned (same audio_ids in the same order), each with
        keys: canonical, gold, asr.

    Returns
    -------
    dict with keys:
        b      — positions where A correct, B wrong
        c      — positions where A wrong, B correct
        pvalue — two-sided exact binomial p-value
    """
    b = c = 0
    for ra, rb in zip(records_a, records_b):
        if ra['audio_id'] != rb['audio_id']:
            raise ValueError(
                f"Record mismatch: {ra['audio_id']} vs {rb['audio_id']}. "
                "Both record lists must cover the same utterances in the same order."
            )
        gt   = get_canonical_labels(ra['canonical'], ra['gold'])
        pa   = [c != a for c, a in zip(ra['canonical'], ra['asr'])]
        pb   = [c != a for c, a in zip(rb['canonical'], rb['asr'])]
        for g, a, bb in zip(gt, pa, pb):
            a_correct = (g == a)
            b_correct = (g == bb)
            if     a_correct and not b_correct: b += 1
            elif not a_correct and b_correct:   c += 1

    # Keep in mind that the independence assumption of the statistics here does not necessarily hold, but this will serve as an approximation.
    n = b + c # b and c here being 
    if n == 0:
        return {'b': 0, 'c': 0, 'pvalue': float('nan')}
    # Exact binomial: two-sided p-value, capped at 1 for the symmetric case
    pvalue = min(2 * _binom.cdf(min(b, c), n, 0.5), 1.0)
    return {'b': b, 'c': c, 'pvalue': pvalue}