# lm_privacy

## tmp fix
as a quick fix, simply copy privacy_engine.py to /home/wyshi/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py
to achieve the selective dp, should import in the future.

original_privacy_engine.py is the original code for privacy engine
pe.py is the to-do my own privacy engine

# get simdial
cd simdial_privacy/
python multiple_domains.py --domain track_package --complexity mix --train_size 10000 --test_size 1000 --valid_size 1000 --save_dir output