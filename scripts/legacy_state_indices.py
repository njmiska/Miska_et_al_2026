"""Ephys state-index helper copied from the local psychometric_utils.py.

Retains 2/3 engaged, 1/4 disengaged and the historical four-state return order.
See provenance/local-source.json for the source checksum.
"""
import numpy as np

def get_glmhmm_indices(subject, eid, state_probability, n_states):
    """
    n_states: Integer, 2: Concatenates right/left engaged into a single 'engaged' label. Likewise for 'disengaged'/
                    4: Returns right/left engaged and right/left disengaged labels.

    """
    states = state_probability[subject][eid][0]

    if n_states == 2:
        engaged_indices = np.where(np.logical_or(states == 2, states == 3))[0]
        disengaged_indices = np.where(np.logical_or(states == 1, states == 4))[0]

        return engaged_indices, disengaged_indices

    elif n_states == 4:
        left_engaged_idx = np.where(states == 3)
        left_disengaged_idx = np.where(states == 4)
        right_engaged_idx = np.where(states == 2)
        right_disengaged_idx = np.where(states == 1)

        return left_engaged_idx, left_disengaged_idx, right_engaged_idx, right_disengaged_idx
