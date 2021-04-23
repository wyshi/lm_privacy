# conda env create -f environment.yml
# chmod +x env_transfer.sh
# ./env_transfer.sh
mkdir -p model
python -m spacy download en_core_web_sm
mkdir -p logs/partial_dp/
mkdir -p logs/dp/
mkdir -p logs/nodp/
mkdir -p attacks/membership_inference/
mkdir -p attacks/canary_insertion/