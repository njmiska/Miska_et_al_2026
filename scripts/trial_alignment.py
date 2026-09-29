"""Validate trial-array lengths before constructing the inference table."""
import numpy as np

TRIAL_FIELDS = (
    'included', 'stimOff_times', 'stimOffTrigger_times', 'stimOnTrigger_times',
    'goCueTrigger_times', 'quiescencePeriod', 'goCue_times', 'response_times',
    'choice', 'stimOn_times', 'contrastLeft', 'contrastRight', 'feedback_times',
    'feedbackType', 'rewardVolume', 'probabilityLeft', 'firstMovement_times',
    'intervals',
)


def align_trial_fields(trials, *, allow_leading_padding=False):
    """Return a copy and a repair log; never infer where a missing trial belongs.

    Explicit legacy mode prepends one zero row, including a (1, 2) row for
    intervals. Use only for sessions independently known to miss a first entry.
    Larger discrepancies and absent fields are rejected before any mutation.
    """
    arrays = {field: np.asarray(trials[field]) for field in TRIAL_FIELDS}
    for field, array in arrays.items():
        expected = 2 if field == 'intervals' else 1
        if array.ndim != expected or (field == 'intervals' and array.shape[1] != 2):
            raise ValueError(f'{field}: unexpected trial shape {array.shape}')
    lengths = {field: len(array) for field, array in arrays.items()}
    target = max(lengths.values())
    if target == 0:
        raise ValueError('Empty trial arrays')
    short = {field: n for field, n in lengths.items() if n != target}
    if short and (not allow_leading_padding or any(target-n != 1 for n in short.values())):
        raise ValueError(f'Trial lengths disagree: {lengths}. Verify missing-entry position before enabling leading padding.')
    result = trials.copy()
    for field in short:
        array = arrays[field]
        result[field] = np.concatenate((np.zeros((1, *array.shape[1:]), dtype=array.dtype), array), axis=0)
    return result, [{'field': field, 'before': n, 'after': target, 'operation': 'prepend_zero'} for field, n in short.items()]
