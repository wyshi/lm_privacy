import pandas as pd 
import os
from glob import glob

BASE_DIR = "./"

episilon = pd.read_csv("attacks/canary_insertion/partialdp_missed/lr0.1_sigma0.5_norm0.001_seed0.csv")


# resume
def concate(p1, p2):
    df0_0 = pd.read_csv(os.path.join(BASE_DIR, p1))
    df0_1 = pd.read_csv(os.path.join(BASE_DIR, p2))
    try:
        assert df0_1.iloc[0]['epoch'] == df0_0.iloc[-1]['epoch'] and (int(df0_1.iloc[0]['model_ppl']) == int(df0_0.iloc[-1]['model_ppl']))
    except:
        import pdb; pdb.set_trace()
    df0_1 = df0_1.iloc[1:]
    df = pd.concat([df0_0, df0_1])
    return df


canary_csv =  [ #(, 'attacks/canary_insertion/partialdp/param_search/resume/lr0.1_sigma0.1_norm0.005_seed1111_resume_50epochs.csv'),
    ('attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed0.csv', 'attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed0_resume50epochs.csv'),
    ('attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed22.csv', 'attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed22_resume50epochs.csv'),
    ('attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed123.csv', 'attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed123_resume50epochs.csv'),
    ('attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed300.csv', 'attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed300_resume50epochs.csv'),
    ('attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed1111.csv', 'attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed1111_resume_50epochs.csv')
]

member_csv = [
    ('attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed0.csv', 'attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed0.csv'),
    ('attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed22.csv', 'attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed22.csv'),
    ('attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed123.csv', 'attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed123.csv'),
    ('attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed300.csv', 'attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed300.csv'),
    ('attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed1111.csv', 'attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed1111.csv'),

]

for p1, p2 in canary_csv:
    df = concate(p1, p2)
    # replace epsilon with the 100 epsilon from partialdp_missed
    df['model_epsilon'] = episilon['model_epsilon'].tolist()
    fname = p1.split("/")[-1].replace(".csv", "_100epochs.csv")
    df.to_csv(os.path.join(BASE_DIR, "attacks/canary_insertion/partialdp/final_concat", fname), index=None)

for p1, p2 in member_csv:
    df = concate(p1, p2)
    # replace epsilon with the 100 epsilon from partialdp_missed
    df['model_epsilon'] = episilon['model_epsilon'].tolist()
    fname = p1.split("/")[-1].replace(".csv", "_100epochs.csv")
    df.to_csv(os.path.join(BASE_DIR, "attacks/membership_inference//partialdp/final_concat", fname), index=None)

# param search concat
param_canary_csv =  [ 
    #('attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv', 'attacks/canary_insertion/partialdp/param_search/resume/lr0.1_sigma0.1_norm0.005_seed1111_resume_50epochs.csv'),

]

# param search concat
param_member_csv =  [ 
    ('attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv', 'attacks/membership_inference/partialdp/final_fix/param_search/resume/lr0.1_sigma0.1_norm0.005_seed1111.csv'),
            
]

for p1, p2 in param_canary_csv:
    df = concate(p1, p2)
    # cannot replace epsilon because we didn't run with sigma=0.1 for 100 epochs
    fname = p1.split("/")[-1].replace(".csv", "_100epochs.csv")
    df.to_csv(os.path.join(BASE_DIR, "attacks/canary_insertion//partialdp/param_search/", fname), index=None)


for p1, p2 in param_member_csv:
    df = concate(p1, p2)
    # cannot replace epsilon because we didn't run with sigma=0.1 for 100 epochs
    fname = p1.split("/")[-1].replace(".csv", "_100epochs.csv")
    df.to_csv(os.path.join(BASE_DIR, "attacks/membership_inference//partialdp/final_fix/param_search/", fname), index=None)
