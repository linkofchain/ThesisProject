from difflib import SequenceMatcher
from scipy.stats import binom as _binom

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


def align_phones(canonical: list[str], hypothesis: list[str]) -> list[dict]:
    """
    Align a hypothesis phone sequence to canonical using difflib.
    Returns one dict per alignment event with keys: canonical, hypothesis, op.
    hypothesis may be a human annotation, an ASR output, or any phone sequence.

    Note: autojunk=False is required — difflib's junk-detection heuristic would
    silently skip common phones (e.g. 'a') in longer sequences otherwise.
    """
    aligned = []
    sm = SequenceMatcher(None, canonical, hypothesis, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        can_chunk = canonical[i1:i2]
        hyp_chunk = hypothesis[j1:j2]

        if op == 'equal':
            for c, h in zip(can_chunk, hyp_chunk):
                aligned.append({'canonical': c, 'hypothesis': h, 'op': 'match'})

        elif op == 'replace':
            # Pair up as substitutions, then spill remainder as del/ins
            for c, h in zip(can_chunk, hyp_chunk):
                aligned.append({'canonical': c, 'hypothesis': h, 'op': 'substitution'})
            n, m = len(can_chunk), len(hyp_chunk)
            for c in can_chunk[min(n, m):]:
                aligned.append({'canonical': c, 'hypothesis': None, 'op': 'deletion'})
            for h in hyp_chunk[min(n, m):]:
                aligned.append({'canonical': None, 'hypothesis': h, 'op': 'insertion'})

        elif op == 'delete':
            for c in can_chunk:
                aligned.append({'canonical': c, 'hypothesis': None, 'op': 'deletion'})

        elif op == 'insert':
            for h in hyp_chunk:
                aligned.append({'canonical': None, 'hypothesis': h, 'op': 'insertion'})

    return aligned


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

    Both gold and asr are independently aligned to canonical.
    For each canonical phone position:
      - gt_label:   True if gold differs from canonical (actual mispronunciation)
      - pred_label: True if ASR output differs from canonical (predicted mispronunciation)

    Insertions have no canonical slot and cannot participate in TR/FA/FR/TA.
    They are counted separately as supplementary fields:
      gold_insertions — extra phones in the human annotation (epenthesis etc.)
      asr_insertions  — extra phones predicted by the ASR system

    Returns:
      TR, FA, FR, TA and derived metrics:
        precision, recall, f1,
        false_acceptance_rate (FAR = FA / (TR+FA)),
        false_rejection_rate  (FRR = FR / (FR+TA)),
        per (Phoneme Error Rate of ASR output vs canonical),
        gold_insertions, asr_insertions
    """
    gt_alignment   = align_phones(canonical, gold)
    pred_alignment = align_phones(canonical, asr)

    gold_insertions = sum(1 for ev in gt_alignment   if ev['op'] == 'insertion')
    asr_insertions  = sum(1 for ev in pred_alignment if ev['op'] == 'insertion')

    gt_labels   = [OP_LABEL[ev['op']] for ev in gt_alignment   if ev['op'] != 'insertion']
    pred_labels = [OP_LABEL[ev['op']] for ev in pred_alignment if ev['op'] != 'insertion']

    if len(gt_labels) != len(pred_labels):
        raise ValueError(
            f"Label length mismatch after alignment: gt={len(gt_labels)}, "
            f"pred={len(pred_labels)}. Both must reduce to len(canonical)={len(canonical)}."
        )

    TR = FA = FR = TA = 0
    for gt, pred in zip(gt_labels, pred_labels):
        if     gt and     pred: TR += 1
        elif   gt and not pred: FA += 1
        elif not gt and   pred: FR += 1
        else:                   TA += 1

    precision = TR / (TR + FR) if (TR + FR) > 0 else float('nan')
    recall    = TR / (TR + FA) if (TR + FA) > 0 else float('nan')
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float('nan'))
    far       = FA / (TR + FA) if (TR + FA) > 0 else float('nan')
    frr       = FR / (FR + TA) if (FR + TA) > 0 else float('nan')
    per       = phone_error_rate(canonical, asr)

    return {
        'TR': TR, 'FA': FA, 'FR': FR, 'TA': TA,
        'precision':             precision,
        'recall':                recall,
        'f1':                    f1,
        'false_acceptance_rate': far,
        'false_rejection_rate':  frr,
        'per':                   per,
        'gold_insertions':       gold_insertions,
        'asr_insertions':        asr_insertions,
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
        pa   = get_canonical_labels(ra['canonical'], ra['asr'])
        pb   = get_canonical_labels(rb['canonical'], rb['asr'])
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