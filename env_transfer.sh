# chmod +x env_transfer.sh
# pip uninstall torch
# pip install torch==1.7.1
mkdir -p model
python -m spacy download en_core_web_sm
mkdir -p logs/partial_dp/
mkdir -p logs/dp/
mkdir -p logs/nodp/
mkdir -p attacks/membership_inference/
mkdir -p attacks/canary_insertion/