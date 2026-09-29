import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.paths import load_paths, states_file
from scripts.trial_alignment import TRIAL_FIELDS, align_trial_fields
from scripts.state_io import export_states, load_states
from scripts.legacy_state_indices import get_glmhmm_indices


class RuntimeTests(unittest.TestCase):
    def test_relative_json_paths_and_override_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config = Path(tmp) / 'paths.json'
            config.write_text(json.dumps({'data-root': 'data', 'glmhmm-dir': 'model'}))
            paths = load_paths(config)
            self.assertEqual(paths['one-cache-dir'], (Path(tmp) / 'data' / 'ONE').resolve())
            self.assertEqual(paths['states-file'], (Path(tmp) / 'model' / 'all_subject_states.pkl').resolve())
            with patch.dict(os.environ, {'MISKA_ONE_CACHE_DIR': '/tmp/env-cache'}):
                self.assertEqual(load_paths(config)['one-cache-dir'], Path('/tmp/env-cache').resolve())
                self.assertEqual(load_paths(config, {'one-cache-dir': '/tmp/cli-cache'})['one-cache-dir'], Path('/tmp/cli-cache').resolve())
            with self.assertRaises(FileNotFoundError): load_paths(Path(tmp) / 'missing.json')

    def test_setup_uses_selected_config_and_passes_cache_to_one(self):
        from scripts.setup_data import main
        from unittest.mock import MagicMock
        import one.api as one_api
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root=Path(tmp).resolve()
            config=root/'paths.json'
            one=MagicMock()
            with patch.object(one_api, 'ONE', return_value=one) as constructor:
                main(['--config',str(config),'--data-root',str(root/'data'),
                      '--output-root',str(root/'output'),'--eid','session-id'])
                self.assertEqual(constructor.call_args.kwargs['cache_dir'],root/'data'/'ONE')
                one.load_object.assert_called_once_with('session-id','trials')
            paths=load_paths(config)
            self.assertEqual(paths['output-root'],root/'output')
            self.assertTrue(config.exists())

    def test_legacy_pickle_name_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical=Path(tmp)/'all_subject_states.pkl'; legacy=canonical.with_suffix('.csv')
            legacy.touch()
            with self.assertWarns(UserWarning): self.assertEqual(states_file({'states-file':canonical}),legacy)
            canonical.touch()
            self.assertEqual(states_file({'states-file':canonical}),canonical)

    def test_padding_includes_intervals_and_preserves_input(self):
        trials={field:np.arange(4, dtype=float) for field in TRIAL_FIELDS}
        trials['intervals']=np.arange(6, dtype=float).reshape(3,2)
        trials['choice']=np.array([-1, 0, 1], dtype=np.int8)
        with self.assertRaises(ValueError): align_trial_fields(trials)
        aligned, log=align_trial_fields(trials, allow_leading_padding=True)
        self.assertEqual(aligned['intervals'].shape,(4,2))
        np.testing.assert_array_equal(aligned['intervals'][0],[0,0])
        np.testing.assert_array_equal(aligned['intervals'][1:],trials['intervals'])
        np.testing.assert_array_equal(aligned['choice'],[0,-1,0,1])
        self.assertEqual(aligned['choice'].dtype,np.int8)
        self.assertEqual(len(trials['choice']),3)
        self.assertEqual({r['field'] for r in log},{'intervals','choice'})
        trials['choice']=np.zeros(2)
        with self.assertRaises(ValueError):align_trial_fields(trials,allow_leading_padding=True)

    def test_equal_lengths_unchanged(self):
        trials={field:np.arange(4) for field in TRIAL_FIELDS}
        trials['intervals']=np.arange(8).reshape(4,2)
        aligned,log=align_trial_fields(trials)
        self.assertEqual(log,[])
        for field in trials:np.testing.assert_array_equal(aligned[field],trials[field])

    def test_portable_roundtrip_preserves_indices_and_empty_entries(self):
        labels=np.array([1,2,3,4,2],dtype=np.int16)
        states={'mouse':{'eid':[labels,np.eye(4,dtype=np.float32)[labels-1]],'empty':{}},'empty-mouse':{}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'states.npz'
            export_states(states,path); restored=load_states(path)
            self.assertEqual(restored['mouse']['empty'],{})
            self.assertEqual(restored['empty-mouse'],{})
            for a,b in zip(states['mouse']['eid'],restored['mouse']['eid']):
                np.testing.assert_array_equal(a,b);self.assertEqual(a.dtype,b.dtype)
            engaged,disengaged=get_glmhmm_indices('mouse','eid',restored,2)
            np.testing.assert_array_equal(engaged,[1,2,4]);np.testing.assert_array_equal(disengaged,[0,3])
            four=get_glmhmm_indices('mouse','eid',restored,4)
            for actual,expected in zip(four,[[2],[3],[1,4],[0]]):np.testing.assert_array_equal(actual[0],expected)
            with self.assertRaises(FileExistsError):export_states(states,path)

if __name__=='__main__':unittest.main()
