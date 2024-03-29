import pandas as pd
from glob import glob
import os
BASE_DIR = "data/adjusted_metrics"


# nodp
paths = """
model/nodp/20210418/181252/
model/nodp/20210418/192226/
model/nodp/20210417/144949/
model/nodp/20210419/052248/
model/nodp/20210418/210231/
"""

# full dp
# paths = """
# model/dp/20210425/151315
# model/dp/20210425/151340
# model/dp/20210425/151359
# model/dp/20210425/151417
# model/dp/20210425/151429
# """

# # s dp
# paths = """
# model/partialdp/20210423/221533
# model/partialdp/20210423/221514/
# model/partialdp/20210427/213802
# model/partialdp/20210427/213754
# model/partialdp/20210427/213900
# """

# # dialog, no dp
# paths = """
# model/nodp/20210515/131136
# model/nodp/20210515/131151
# model/nodp/20210515/131209
# model/nodp/20210515/131323
# model/nodp/20210515/131125
# """

# # dialog, dp
# paths = """
# model/dp/20210502/123625/
# model/dp/20210512/232905
# model/dp/20210508/123015
# model/dp/20210506/022233
# model/dp/20210511/110444
# """

# # dialog, sdp
# paths = """
# model/partialdp/20210503/230732
# model/partialdp/20210503/230450
# model/partialdp/20210503/230904
# model/partialdp/20210503/230952
# model/partialdp/20210504/100507/
# """


# # missed, no dp
# paths = """
# model/nodp/20210426/182422
# model/nodp/20210426/234106
# model/nodp/20210426/234125
# model/nodp/20210426/234146
# model/nodp/20210426/234201
# """

# # missed, sdp
# paths = """
# model/partialdp/20210426/223009
# model/partialdp/20210427/211240
# model/partialdp/20210427/211322
# model/partialdp/20210427/211339
# model/partialdp/20210427/211411
# """


BASE_DIR = "data/adjusted_metrics/eps4.91_epoch37"

# s dp
paths = """
model/partialdp/20210421/123500/
model/partialdp/20210421/123522/
model/partialdp/20210421/123511/
model/partialdp/20210421/123530/
model/partialdp/20210418/191438/
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
    print(f"{metrics}: {mean:.2f} $\pm$ {std:.2f}")
    print()

print_metric(dfs_model, 'overall_ppl')
print_metric(dfs_model, 'overall_acc')
print_metric(dfs_model, 'nonprivate_ppl')
print_metric(dfs_model, 'nonprivate_acc')
print_metric(dfs_model, 'private_ppl')
print_metric(dfs_model, 'private_acc')
