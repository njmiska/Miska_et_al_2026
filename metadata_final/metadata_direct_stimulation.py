"""Normalized insertion metadata for Supplementary Figure 1 validation.

The trial lists remain defined in the historical ephys metadata module so there
is one exact source of truth.  This adapter exposes them in the same row-based
shape used by the public-release manifest.
"""

from ephys_analysis.metadata_optostim import (
    excitation_trials_range_list_SNr_directstim,
    excitation_trials_range_list_ZI_directstim,
    inhibition_trials_range_list_SNr_directstim,
    inhibition_trials_range_list_ZI_directstim,
    pids_list_SNr_directstim,
    pids_list_ZI_directstim,
)


def _rows(region, pids, excitation, inhibition):
    if not (len(pids) == len(excitation) == len(inhibition)):
        raise ValueError(f"Mismatched {region} direct-stimulation metadata lists")
    return [
        {
            "PID": pid,
            "brain region": region,
            "condition": f"{region}_directstim",
            "opto excitation trials": excitation_trials,
            "opto inhibition trials": inhibition_trials,
        }
        for pid, excitation_trials, inhibition_trials
        in zip(pids, excitation, inhibition)
    ]


insertions = (
    _rows(
        "SNr",
        pids_list_SNr_directstim,
        excitation_trials_range_list_SNr_directstim,
        inhibition_trials_range_list_SNr_directstim,
    )
    + _rows(
        "ZI",
        pids_list_ZI_directstim,
        excitation_trials_range_list_ZI_directstim,
        inhibition_trials_range_list_ZI_directstim,
    )
)
