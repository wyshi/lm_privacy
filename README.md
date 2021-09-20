# Selective Differential Privacy for Language Modeling
This is the repo for the paper [Selective Differential Privacy for Language Modeling](https://arxiv.org/pdf/2108.12944.pdf]. You can cite the paper using the following bibtex.

```
@article{shi2021selective,
  title={Selective Differential Privacy for Language Modeling},
  author={Shi, Weiyan and Cui, Aiqi and Li, Evan and Jia, Ruoxi and Yu, Zhou},
  journal={arXiv preprint arXiv:2108.12944},
  year={2021}
}
```

## How to install
original_privacy_engine.py is the original code for privacy engine
pe.py is the to-do my own privacy engine
```
# git clone the repo first
# create an environment called "privacy"
conda env create -f environment.yml

# just in case, permission to execute the file
chmod +x env_transfer.sh

# create the folders using this script
./env_transfer.sh

# install the correct torch version
pip uninstall torch
pip install torch==1.7.1

# As a quick fix to achieve the selective dp, simply copy privacy_engine.py in this git repo to the privacy_engine in opacus (usually it's in ~/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py), should import in the future.
# first make a copy of the original privacy_engine.py
mv ~/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py ~/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine_original.py
# then copy the privacy_engine.py
cp privacy_engine.py ~/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py
```

## For the dialogue system task: how to get the CUSTOMERSIM dataset
```
cd simdial_privacy/
python multiple_domains.py --domain track_package --complexity mix --train_size 10000 --test_size 1000 --valid_size 1000 --save_dir output
```

# The dataset


# how to train the SDP models
```
# selective DP, turn both `-dp` and `--partial` on, 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 0.01 2>&1 | tee logs/partial_dp/20210414/1128/nohidden_lr0.1_norm0.01 
# DPSGD, turn only `-dp` on 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -norm 0.02 --seed 0 2>&1 | tee logs/dp/20210414/1224/lr0.1_sigma0.5_norm0.02_seed0 # screen dp
# No DP
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:0 2>&1 | tee logs/nodp/20210416/1710/bs16.log

# dialogue model
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs
```

# how to run the canary insertion attack
```
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/111019 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv
```

# how to run the member inference attack
```
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210418/191438 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed1111.csv
```