# add 10 
# screen -R
python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:0 --log-interval 10 --seed 1111 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed1111.log
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 0 2>&1 | tee logs/partial_dp/dialog/20210503/sigma0.7_norm5e-3_seed0


# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:1 --log-interval 10 --seed 0 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed0.log
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 123 2>&1 | tee logs/partial_dp/dialog/20210503/sigma0.7_norm5e-3_seed123



# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:2 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed123.log
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:2 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 22 2>&1 | tee logs/partial_dp/dialog/20210503/sigma0.7_norm5e-3_seed22

# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:4 --log-interval 10 --seed 22 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed22.log
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:4 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 300 2>&1 | tee logs/partial_dp/dialog/20210503/sigma0.7_norm5e-3_seed300

# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:5 --log-interval 10 --seed 300 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed300.log


# canary insertion
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed300.csv
