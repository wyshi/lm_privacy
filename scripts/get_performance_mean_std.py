import pandas as pd
from glob import glob
import os
BASE_DIR = "data/adjusted_metrics"

paths = """
model/nodp/20210418/181252/
model/nodp/20210418/192226/
model/nodp/20210417/144949/
model/nodp/20210419/052248/
model/nodp/20210418/210231/
"""


paths = paths.split('\n')
paths = [p for p in paths if p != '']


dfs = []
for p in glob(os.path.join(BASE_DIR, "test_*.csv")):
    dfs.append(pd.read_csv(p))

dfs = pd.concat(dfs)

dfs_model = []
for p in paths:
    dfs_model.append(dfs[dfs.model_path.str.startswith(p)])

dfs_model = pd.concat(dfs_model)  
    
      
assert dfs_model.shape[0] == len(paths)

print("test metrics: ")
def print_metric(df, metrics):
    mean = df[metrics].mean()
    std = df[metrics].std()
    print(f"{metrics}: {mean}, std {std}")
    print()

print_metric(dfs_model, 'overall_ppl')
print_metric(dfs_model, 'overall_acc')
print_metric(dfs_model, 'nonprivate_ppl')
print_metric(dfs_model, 'nonprivate_acc')
print_metric(dfs_model, 'private_ppl')
print_metric(dfs_model, 'private_acc')
