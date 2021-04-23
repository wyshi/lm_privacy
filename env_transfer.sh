# conda env create -f environment.yml
# chmod +x env_transfer.sh
# ./env_transfer.sh
# pip uninstall torch
# pip install torch==1.7.1
# mv /home/wyshi/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py /home/wyshi/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine_original.py
# cp privacy_engine.py /home/wyshi/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py
mkdir -p model
python -m spacy download en_core_web_sm
mkdir -p logs/partial_dp/
mkdir -p logs/dp/
mkdir -p logs/nodp/
mkdir -p attacks/membership_inference/
mkdir -p attacks/canary_insertion/