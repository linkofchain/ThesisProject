from difflib import SequenceMatcher


# Ground truth label per canonical phone position.
# True = mispronounced, False = correct, None = insertion (no canonical slot)
OP_LABEL = {'match': False, 'substitution': True, 'deletion': True, 'insertion': None}


def align_phones(canonical: list[str], annotated: list[str]) -> list[dict]:
    """
    Align annotated phones to canonical using difflib.
    Returns one dict per alignment event with keys: canonical, annotated, op.

    Note: autojunk=False is required — difflib's junk-detection heuristic would
    silently skip common phones (e.g. 'a') in longer sequences otherwise.
    """
    aligned = []
    sm = SequenceMatcher(None, canonical, annotated, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        can_chunk = canonical[i1:i2]
        ann_chunk = annotated[j1:j2]

        if op == 'equal':
            for c, a in zip(can_chunk, ann_chunk):
                aligned.append({'canonical': c, 'annotated': a, 'op': 'match'})

        elif op == 'replace':
            # Pair up as substitutions, then spill remainder as del/ins
            for c, a in zip(can_chunk, ann_chunk):
                aligned.append({'canonical': c, 'annotated': a, 'op': 'substitution'})
            n, m = len(can_chunk), len(ann_chunk)
            for c in can_chunk[min(n, m):]:
                aligned.append({'canonical': c, 'annotated': None, 'op': 'deletion'})
            for a in ann_chunk[min(n, m):]:
                aligned.append({'canonical': None, 'annotated': a, 'op': 'insertion'})

        elif op == 'delete':
            for c in can_chunk:
                aligned.append({'canonical': c, 'annotated': None, 'op': 'deletion'})

        elif op == 'insert':
            for a in ann_chunk:
                aligned.append({'canonical': None, 'annotated': a, 'op': 'insertion'})

    return aligned


def phone_error_rate(canonical: list[str], annotated: list[str]) -> float:
    """PER = (substitutions + deletions + insertions) / len(canonical)."""
    if not canonical:
        return float('nan')
    errors = 0
    sm = SequenceMatcher(None, canonical, annotated, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'replace':
            errors += max(i2 - i1, j2 - j1)
        elif op == 'delete':
            errors += i2 - i1
        elif op == 'insert':
            errors += j2 - j1
    return errors / len(canonical)


def get_canonical_labels(canonical: list[str], annotated: list[str]) -> list[bool]:
    """
    For each canonical phone position, return True (mispronounced) or False (correct).
    Insertions (annotated phones with no canonical counterpart) are discarded.
    Length of result always equals len(canonical).
    """
    alignment = align_phones(canonical, annotated)
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

    Returns:
      Confusion matrix counts (TR, FA, FR, TA) and derived metrics:
        precision, recall, f1,
        false_acceptance_rate (FAR = FA / all actual mispronunciations),
        false_rejection_rate  (FRR = FR / all actually correct phones),
        per (Phoneme Error Rate of ASR output vs canonical)
    """
    gt_labels   = get_canonical_labels(canonical, gold)
    pred_labels = get_canonical_labels(canonical, asr)

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
    }
