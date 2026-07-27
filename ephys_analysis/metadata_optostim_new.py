"""
Optostim insertion metadata (consolidated, one dict per insertion).

Each entry is a single probe insertion. Fields:
  'PID'                  : probe insertion ID
  'opto inhibition trials': trial indices in the inhibition epoch
                           (list of ints, or 'ALL' for the whole session)
  'hemisphere stim'      : hemisphere of the stimulated region (FILL IN: 'left'/'right'/'both')
  'hemisphere recorded'  : hemisphere of the recorded midbrain (FILL IN: 'left'/'right')
  'condition'            : 'ipsi' or 'contra' (recorded hemi relative to stim)
  'brain region'         : stimulated region — 'ZI' / 'SNr' / 'STN'
  'mouse'                : mouse ID (auto-extracted; 'nan' where unknown)

All mice here are fully trained. Light-artifact units are no longer
listed — the pipeline detects them automatically from waveforms.
"""

insertions = [
    # ===== STN — ipsi =====
    {'PID': 'c547dda9-2006-4e4d-9498-396aac25d54b', 'opto inhibition trials': list(range(191, 354)) + list(range(493, 9999)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_024'}, #unclear laser bump
    {'PID': 'c4051389-6e82-4c48-9116-fc5be6ebace9', 'opto inhibition trials': list(range(156, 355)) + list(range(524, 9999)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_024'}, #clear laser bump
    {'PID': '60668d72-b4ba-4c59-830a-915458f62192', 'opto inhibition trials': list(range(0, 150)) + list(range(326, 472)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_024'}, #clear laser bump
    {'PID': '6ee63cc8-37ef-49ff-8c00-70a03e503725', 'opto inhibition trials': list(range(0, 180)) + list(range(432, 9999)), 'hemisphere stim': 'right', 'hemisphere recorded': 'right', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_025'}, #clear laser bump
    {'PID': '126cfd2e-2fa4-4d81-a520-25bbeab59fae', 'opto inhibition trials': list(range(0, 185)) + list(range(361, 559)), 'hemisphere stim': 'right', 'hemisphere recorded': 'right', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_025'}, #clear laser bump
    {'PID': '6bfade8f-e22a-42fa-b8d0-8c22d172c237', 'opto inhibition trials': list(range(3, 168)) + list(range(493, 9999)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_025'}, #unclear laser bump
    # {'PID': '442a8e82-4be2-4fdf-b457-6d7f4b0111e5', 'opto inhibition trials': list(range(0, 154)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_026'}, #only inhib stim data from one bias block type
    {'PID': 'bc5d45e1-7e63-4d32-94f2-f785be9e75f8', 'opto inhibition trials': list(range(155, 343)), 'hemisphere stim': 'right', 'hemisphere recorded': 'right', 'condition': 'ipsi', 'brain region': 'STN', 'mouse': 'SWC_NM_026'}, #very few stim inhibition trials (below 30) #clear laser bump
    # ===== STN — contra =====
    {'PID': '19b9e2f8-5e46-4ff4-bef3-4652019be01a', 'opto inhibition trials': list(range(191, 354)) + list(range(493, 9999)), 'hemisphere stim': 'left', 'hemisphere recorded': 'right', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_024'}, #clear laser bump
    {'PID': 'd082c08d-2c7e-4761-9936-8bbefa8068d1', 'opto inhibition trials': list(range(156, 355)) + list(range(524, 9999)), 'hemisphere stim': 'left', 'hemisphere recorded': 'right', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_024'}, #clear laser bump
    {'PID': 'ae535cc1-87aa-4fa8-acca-8892c8f81e4c', 'opto inhibition trials': list(range(0, 150)) + list(range(326, 472)), 'hemisphere stim': 'left', 'hemisphere recorded': 'right', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_024'}, #clear laser bump
    {'PID': '6509394a-ee93-4164-a516-d4d483fb5da0', 'opto inhibition trials': list(range(0, 185)) + list(range(361, 559)), 'hemisphere stim': 'right', 'hemisphere recorded': 'left', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_025'}, #clear laser bump
    {'PID': '1bffcd91-1e56-4724-9047-69e18bc7a104', 'opto inhibition trials': list(range(3, 168)) + list(range(493, 9999)), 'hemisphere stim': 'left', 'hemisphere recorded': 'right', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_025'}, #unclear laser bump
    # {'PID': '855ef2f9-4a68-4398-bf3d-c9a6b7c44702', 'opto inhibition trials': list(range(0, 154)), 'hemisphere stim': 'left', 'hemisphere recorded': 'right', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_026'}, #only inhib stim data from one bias block type
    {'PID': 'dcd1c7d3-b413-47dc-a120-c3cafbbbbd96', 'opto inhibition trials': list(range(155, 343)), 'hemisphere stim': 'right', 'hemisphere recorded': 'left', 'condition': 'contra', 'brain region': 'STN', 'mouse': 'SWC_NM_026'}, #very few stim inhibition trials (below 30) #clear laser bump
    # ===== SNr — ipsi =====
    {'PID': '518b61c2-45bc-40c2-bee1-d87b0d1986ac', 'opto inhibition trials': list(range(139, 242)) + list(range(364, 461)) + list(range(581, 674)) + list(range(769, 798)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_018'}, #few units - 45 midbrain after filtering. effect is noisy if not filtering, but likely not strong. #somewhat clear laser bump
    {'PID': 'e4696ffd-248e-41cb-a62a-16e320b8cd7e', 'opto inhibition trials': list(range(121, 180)) + list(range(215, 283))+ list(range(423, 528)) + list(range(660, 767)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_018'}, #partial effect #clear laser bump
    {'PID': '59bf32ee-1d83-4a7f-bf14-590b610c96e0', 'opto inhibition trials': list(range(422, 485)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_018'}, #very few stim inhibition trials (below 30) #unclear laser bump
    {'PID': '930adc32-438a-4548-a741-dc8a487ebd4f', 'opto inhibition trials': list(range(0, 410)) + list(range(516, 610)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #full effect # strange BS results, opto below control baseline - may need to exclude some units? cleaned up with zscore=0.5 #small laser bump; list(range(516, 776))
    {'PID': '7778d726-767e-4d1b-a879-01faf2075828', 'opto inhibition trials': list(range(0, 550)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #few units (32 midbrain after filtering). CD coding is weak. still, likely strong partial effect #clear laser bump
    {'PID': '068538e4-08a7-4d11-a807-e9cf698d63b8', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #no effect? - strong pre-laser BS collapse #clear laser bump
    {'PID': '72b7c463-ed30-4c24-8ddf-5e8eddafc46e', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #few units (below 50 midbrain) AND very few inhibition stim trials - strange BS activity, opto trace is under control trace for whole trace... #small laser bump
    {'PID': 'c9a6b866-2d9b-481c-86ec-0d4937fbd696', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #little BS effect #clear laser bump, CD separation control is weak!
    {'PID': '68288763-9572-4678-9eb4-3866e3e9fb3d', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #no CD effect, little BS effect #clear laser bump
    {'PID': 'fc4f446b-177c-4b94-89d2-14c0500374a4', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #no CD effect, little BS effect #clear laser bump
    {'PID': '32425853-de5f-4e5d-8a73-fe1285893c7f', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #small CD effect, no BS effect #somewhat clear laser bump
    {'PID': '9583d73c-ee29-45d1-9aa1-2b5917bcf726', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #small CD effect, no BS effect #clear laser bump
    {'PID': '141fba8a-403d-44f2-89bc-0d8cd45f611e', 'opto inhibition trials': list(range(0, 320)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #small CD effect, Some BS effect, without baseline problems #clear laser bump
    {'PID': 'a1289836-79d8-45ce-9481-072c9e5c71b0', 'opto inhibition trials': list(range(0, 320)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #partial effect #somewhat clear laser bump
    {'PID': '414329ec-11b7-48a7-a011-0ec05948c66b', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #no clear CD effect, #clear laser bump
    {'PID': '77c33d3e-8b71-43f9-9a9c-b7dc49a25e30', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #full effect #clear laser bump
    {'PID': '9c43c3e2-1019-4e66-b8b2-b8b693fa5254', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #full effect #somewhat clear laser bump
    {'PID': '227e4a3a-9340-48bc-82a3-dd0a04b123a9', 'opto inhibition trials': list(range(150, 9999)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #Strong CD effect, few units (27 midbrain after filtering) - strange BS effect, fluctuates a lot #clear laser bump
    {'PID': '85caca51-d501-4d81-85f2-74f084c7e99e', 'opto inhibition trials': list(range(150, 9999)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #full effect #clear laser bump
    {'PID': '09ee9be3-3c85-46bb-aed3-3143862ef03d', 'opto inhibition trials': list(range(130, 9999)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #full effect #clear laser bump
    {'PID': '2e13e28b-8ec3-436c-8d63-408b323e9511', 'opto inhibition trials': list(range(130, 9999)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #full effect #clear laser bump
    {'PID': '15eca47c-a6fe-4b7b-99fd-2657549c3258', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #weak partial effect #unclear/small laser bump
    {'PID': 'af47abd5-e9cb-4130-ba66-5a277141a1bb', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #partial effect #small laser bump
    {'PID': '89c36fa2-e889-46a2-af2f-ee0fad10de43', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #near full effect #clear laser bump
    {'PID': 'e1b4c254-0455-4cd3-9642-0e23892aef85', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #near full effect #clear laser bump
    # ===== SNr — contra =====
    {'PID': 'b96ed9ce-1a0a-4818-b896-8aa79ca26801', 'opto inhibition trials': list(range(139, 242)) + list(range(364, 461)) + list(range(581, 674)) + list(range(769, 798)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_018'}, #strong partial effect #clear laser bump
    {'PID': 'b7998d00-b4c4-4695-8fc3-f8001539c90e', 'opto inhibition trials': list(range(121, 283)) + list(range(423, 528)) + list(range(660, 767)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_018'}, #strong partial effect #small laser bump
    {'PID': 'caf25d31-2d5b-45a9-85b4-585e380ebab2', 'opto inhibition trials': list(range(0, 410)) + list(range(516, 776)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #very few units after filtering (~10) #no visible laser bump
    {'PID': '7a79b7cf-4d29-4a83-8d33-6bd3e4ef3307', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #no effect #NO VISIBLE laser bump
    {'PID': '94a5c20d-82be-4040-80cc-03ef2e5854df', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #few units (34 midbrain) - still, CD separation looks decent, no effect on CD #clear laser bump
    {'PID': '3dd67257-217e-4d25-bb2b-59a606d944a7', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_096'}, #very few inhibition stim trials #unclear laser bump
    {'PID': 'a327ddee-8b7c-4463-9c24-6f82d2bfe590', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #unclear laser bump
    {'PID': '6bf18fe0-fca9-4cd3-aa69-546d34d24c12', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #no visible laser bump
    {'PID': '77946f89-7b49-43b0-b34d-c17fc70504c4', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #no visible laser bump
    {'PID': '4743a9f7-24d3-4cac-b956-d0323d4269db', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_102'}, #unclear laser bump
    {'PID': 'a02790bb-1dcf-4e9d-a8f0-c0a071bc2e37', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'SNr', 'mouse': 'SWC_NM_113'}, #strong partial effect #unclear laser bump
    # ===== ZI — ipsi =====
    # {'PID': '0d26b5b4-e951-49f5-a0b5-2c62f46d4c63', 'opto inhibition trials': list(range(172, 401)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_022'}, #no effect - some expression in SNr in this hemisphere!
    {'PID': 'f54b959b-fee4-4130-951e-e366d34a5cbc', 'opto inhibition trials': list(range(246, 514)) + list(range(700, 713)), 'hemisphere stim': 'right', 'hemisphere recorded': 'right', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_022'}, #no effect #clear laser bump
    {'PID': '7f712873-42c8-42bc-a782-76b03ae3fb0f', 'opto inhibition trials': list(range(0, 113)) + list(range(366, 504)), 'hemisphere stim': 'left', 'hemisphere recorded': 'left', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_022'}, #full effect (maybe noisy - check this) - some expression in SNr in this hemisphere! #clear laser bump
    {'PID': 'c7a7990a-fb2f-4329-99bc-5a85765969d6', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #weak partial effect #clear laser bump
    {'PID': 'e71a89e5-9526-4be3-a5e7-5f9af217927d', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #few units (below 30 midbrain after filtering) - CD very noisy w/ poor separation #clear laser bump
    {'PID': '966d7f41-bca0-425a-8216-3a757231ea64', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #no effect #clear laser bump
    {'PID': '820e6e2f-4a1d-4777-8d67-5a62d382efa0', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #no effect / very weak #clear laser bump
    {'PID': '27b42a9d-05bc-4635-9eb3-418c2894e5b2', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #no effect #clear laser bump
    {'PID': 'e7919a12-238d-44da-b9d7-37adf5fff9ba', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #no effect #clear laser bump
    {'PID': 'b4b1e085-c8ba-4c4b-8125-369f79029d4e', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #strong partial effect (maybe noisy - check this) #clear laser bump
    {'PID': 'c6ce3c19-6037-412b-ba92-f5dff157aeba', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #strong partial effect (maybe noisy - check this) #clear laser bump
    {'PID': '246d8fa8-dbf5-486a-b074-a064b33f29e3', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #partial effect #clear laser bump
    {'PID': 'e4e69307-ff03-492e-a738-e742d47a092a', 'opto inhibition trials': 'ALL', 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'ipsi', 'brain region': 'ZI', 'mouse': 'SWC_NM_111'}, #partial effect #unclear laser bump
    # ===== ZI — contra =====
    {'PID': '52ed488c-0cbe-4518-880f-c52c162a8999', 'opto inhibition trials': list(range(172, 401)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'ZI', 'mouse': 'SWC_NM_022'}, #weak partial effect #unclear laser bump
    {'PID': '1579269a-17c2-46ef-a14e-448217386454', 'opto inhibition trials': list(range(246, 514)) + list(range(700, 713)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'ZI', 'mouse': 'SWC_NM_022'}, #no effect #unclear laser bump
    {'PID': '7f94e86e-3a8b-4026-b120-89eeadf45a8d', 'opto inhibition trials': list(range(0, 113)) + list(range(366, 504)), 'hemisphere stim': 'nan', 'hemisphere recorded': 'nan', 'condition': 'contra', 'brain region': 'ZI', 'mouse': 'SWC_NM_022'}, #no effect #clear laser bump
]


def find_insertions(insertions=insertions, **criteria):
    """Filter insertions by exact-match or callable criteria.

    Example:
        find_insertions(**{'brain region': 'SNr', 'condition': 'ipsi'})
        find_insertions(**{'brain region': lambda r: r in ('SNr','ZI')})
    Returns a list of matching insertion dicts.
    """
    out = []
    for ins in insertions:
        ok = True
        for key, val in criteria.items():
            if callable(val):
                if not val(ins.get(key)):
                    ok = False; break
            elif ins.get(key) != val:
                ok = False; break
        if ok:
            out.append(ins)
    return out
