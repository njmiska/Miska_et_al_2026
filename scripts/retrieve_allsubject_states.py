#!/usr/bin/env python3
"""Retrieve GLM-HMM states using the laboratory's inference procedure.

Adapted from Nate's retrieve_allsubject_states.py. Scientific model selection,
seed, design-matrix construction and state ordering are retained. See
REPRODUCIBILITY_REVIEW.md for guards and known validation limits.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import load_paths, states_file, REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-config', type=Path)
    parser.add_argument('--subject', action='append', required=True)
    parser.add_argument('--metadata', type=Path, default=REPO_ROOT / 'metadata_final' / 'metadata_opto_behavior.py')
    parser.add_argument('--states', type=int, choices=(4,), default=4, help='Four fitted states; analysis n_states=2 pools these into engaged/disengaged')
    parser.add_argument('--output', type=Path, required=True, help='New .pkl output; existing files are refused')
    parser.add_argument('--overwrite', action='store_true', help='Recompute selected entries from the input archive')
    parser.add_argument('--allow-leading-padding-eid', action='append', default=[], help='EID independently verified to miss first entries; prepends one zero row')
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.output.suffix != '.pkl' or args.output.exists():
        parser.error('--output must be a new .pkl file')
    runtime_paths = load_paths(args.paths_config)
    # Match the original script's explicit ssm checkout before loading GLM-HMM.
    sys.path.insert(0, str(runtime_paths['ssm-dir']))
    import ssm
    if not hasattr(ssm, 'HMM'):
        raise ImportError('Configured ssm does not expose HMM; check ssm-dir and installation')
    sys.path.insert(0, str(runtime_paths['glmhmm-dir']))
    import requests
    import json
    import pandas as pd
    import os
    from os.path import join
    import matplotlib.pyplot as plt
    import numpy as np
    from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
        get_file_name_for_best_model_fold, partition_data_by_session, \
        create_violation_mask, get_marginal_posterior
    from glm_hmm_utils import get_raw_data, create_design_mat, remap_choice_vals
    from one.api import ONE
    from one.alf.io import AlfBunch

    one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir=runtime_paths['one-cache-dir'])
    # one = ONE(mode='remote')

    K = args.states
    '''

    Procedure:
        1. get the `eid`
        2. get the corresponding `subject`
        3. If trials table for `subject` exists
            - get the corresponding `hmm_params` from individual fit
           otherwise
            - get the `hmm_params` from global fit
        4. get the trials info for `eid` and format `input`
        5. run `get_marginal_posterior`
    '''

    # Use a frozen manuscript cohort, or an explicit trusted metadata module.
    from scripts.reproducibility import _load_python_metadata
    metadata_rows = _load_python_metadata(args.metadata, 'sessions')
    selected = [row for row in metadata_rows if row.get('Mouse_ID') in args.subject]
    eids = list(dict.fromkeys(row['EID'] for row in selected))
    MouseIDs = [row['Mouse_ID'] for row in selected]
    if not eids:
        raise ValueError('No sessions match the requested subjects in the metadata file')
    """
    state_probability: {
        ['subject']: {
            ['eid']: [state trial labels] <- GLM-HMM trial-by-trial labels [1, 2, 3, 4]
                     [probability x states] <- posterior probability for all GLM-HMM states [expected: N trial x 4 states]

            ['eid']
            ['eid']
            ...
        }
        ['subject']
        ...

        }
    """
    # initialise dictionary for storing posterior probabilities
    input_states = states_file(runtime_paths)
    if input_states.exists():
        from scripts.state_io import load_states
        state_probability = load_states(input_states)
    else:
        state_probability = {}
    repair_log = []
    processed_sessions = []

    failed_session = {}
    for this_subject in np.unique(MouseIDs):

        failed_session[this_subject] = 0

        if this_subject not in state_probability:
                state_probability[this_subject] = {}


    # Paths
    figure_path = str(runtime_paths['glmhmm-figure-dir'])
    data_path = str(runtime_paths['glmhmm-data-dir'])

    dict_dir = join(data_path, 'GLM-HMM')
    data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
    output_dir = join(figure_path, 'GLM-HMM', 'individual_sessions')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']


    for i, eid in enumerate(eids):

        if eid == 'b8d63536-7330-40ac-883a-7bf3da6b0215':
            continue

        subject = next(row['Mouse_ID'] for row in selected if row['EID'] == eid)
        try:
            subject = one.get_details(eid)['subject']

            state_probability.setdefault(subject, {})
            if not state_probability[subject].get(eid) or args.overwrite:
                pass
            else:
                print(f'eid: {eid} already been previously processed. skipping...')
                continue


        except requests.HTTPError as e:
            print(f'Skipping session {eid}: No session matches the given query')
            failed_session[subject] = failed_session.get(subject, 0) + 1
            continue
        details = one.get_details(eid)
        session_date = (details['date'].year, details['date'].month, details['date'].day)
        print(f"\n{i+1:03d}/{len(eids)} --> {subject}")

        figure_dir = join(output_dir, subject, eid)
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        try:
            trials = one.load_object(eid, 'trials')

        except Exception as e:
            print(f'Error occurred for session {eid:}: {e}')
            failed_session[subject] = failed_session.get(subject, 0) + 1
            continue

        try:
            cached_table = Path(data_path) / 'GLM-HMM' / 'individual_sessions' / subject / eid / 'trials.pqt'
            trials_df = pd.read_parquet(cached_table)

        except FileNotFoundError as e:
            print("Trials table for this session not found - this likely reflects a strange trials inconsistency that hasn't yet been resolved")
            try:
                trials_df = trials.to_df()
            except ValueError:
                from scripts.trial_alignment import align_trial_fields
                aligned, repairs = align_trial_fields(
                    trials, allow_leading_padding=eid in args.allow_leading_padding_eid)
                trials = AlfBunch(aligned)
                repair_log.extend(dict(eid=eid, subject=subject, **repair) for repair in repairs)
                trials_df = trials.to_df()
                print('trials table successfully reformatted!')



        # IF THE MOUSE WAS FILTERED OUT AT THE VERY BEGINNING,
        # WE WILL NOT HAVE THE PROPERLY FORMATTED DESIGN MATRIX FOR IT,
        # OR THE PARAMETERS OF THE INDIVIDUAL FIT.
        # WE CREATE THE DESIGN MATRIX FROM SCRATCH AND RUN THE GLM-HMM MODEL
        # WITH THE GLOBAL FIT PARAMETERS.
        try:

            results_dir = join(dict_dir, 'results', 'individual_fit', subject)

            np.random.seed(41)

            # Also get data for subject:
            inpt, y, session = load_data(join(data_dir, subject + '_processed.npz'))
            all_trials = pd.read_parquet( join(data_dir, subject + '_trials_table.pqt') )

            _string = "Using saved design matrix and individual fit parameters"

        except Exception as e:
            print("MISSING SUBJECT DATA:", e)

            results_dir = join(dict_dir, 'results')

            try:
                subject, stim_left, stim_right, rewarded, choice, bias_probs \
                    = get_raw_data(eid, one)
                inpt = create_design_mat(trials.choice,
                                        trials.contrastLeft,
                                        trials.contrastRight,
                                        trials.feedbackType)
                y = np.expand_dims(remap_choice_vals(trials.choice), axis=1)
                session = np.array([eid for j in range(y.shape[0])])
                _string = "Created design matrix and using global fit parameters"

            except Exception as e:
                print("ANOTHER ERROR:", e)
                print("SKIPPING SESSION")
                failed_session[subject] = failed_session.get(subject, 0) + 1
                continue

        print(_string)

        cv_file = join(results_dir, "cvbt_folds_model.npz")
        cvbt_folds_model = load_cv_arr(cv_file)
        with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
            best_init_cvbt_dict = json.load(f)

        raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                    results_dir,
                                                    best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        all_sessions, sess_loc = np.unique(session, return_index=True)

        # Create mask:
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                    inpt.shape[0])

        y[np.where(y == -1), :] = 1
        inputs, datas, train_masks = partition_data_by_session(
            np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask,
            session)

        print("Computing GLM-HMM posterior")
        posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                                hmm_params, K, range(K))
        if len(posterior_probs) != len(trials_df):
            raise ValueError(f'{eid}: posterior has {len(posterior_probs)} rows but session table has {len(trials_df)}; inspect saved design-matrix session selection')
        trials_df[f"glm-hmm_{K}"] = pd.Series(list(posterior_probs))
        trials_df.to_parquet( join(figure_dir, "trials.pqt") )

        ##
        # Add posterior probabilities to dictionary
        trial_state = np.argmax(posterior_probs, axis=1) + 1  # dominant state on each trial
        state_probability[subject][eid] = [trial_state, posterior_probs] # probabilities across all states

        # Plot posterior probabilities
        fig, ax = plt.subplots()

        ax.set_ylabel("Posterior prob.")
        ax.set_xlabel("Trial")
        for k, prob in enumerate(posterior_probs.T):
            ax.plot(prob, label=f"State {k+1}", color=cols[k])
        ax.plot(trials_df.probabilityLeft, c='k', ls='--', label="P(L)")
        ax.set_title(f"{subject:} {session_date:}")
        ax.legend(loc="upper right")

        fig.savefig( join(figure_dir, f"posterior_K={K}.png"))

        plt.close(fig)
        processed_sessions.append({"subject": str(subject), "eid": eid, "trials": len(trial_state), "model_file": str(raw_file)})

    print(f'Failed sessions: {failed_session}')

    # Write a new, honestly named pickle atomically; never overwrite the legacy CSV.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as f:
        temporary = Path(f.name)
        pickle.dump(state_probability, f)
    try:
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    args.output.with_suffix('.repairs.json').write_text(json.dumps(repair_log, indent=2) + '\n')
    args.output.with_suffix('.run.json').write_text(json.dumps({
        'processed_sessions': processed_sessions, 'failed_session_counts': failed_session,
        'states': K, 'seed': 41, 'ssm_source': ssm.__file__,
        'numpy_version': np.__version__, 'python': sys.version,
        'metadata': str(args.metadata.resolve()),
    }, indent=2) + '\n')
    print(f'Saved states to {args.output}')
    return 1 if any(failed_session.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
