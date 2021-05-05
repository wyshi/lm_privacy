# add 10 
# screen -R
python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:0 --log-interval 10 --seed 1111 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed1111.log


# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:1 --log-interval 10 --seed 0 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed0.log



# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:2 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed123.log

# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:4 --log-interval 10 --seed 22 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed22.log

# screen -R

python -u main.py -bs 32 --lr 20 --data data/simdial --data_type dial --cuda cuda:5 --log-interval 10 --seed 300 2>&1 | tee logs/nodp/dialog/20210503/add10/dialog_bs32_seed300.log
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:5 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 1111 2>&1 | tee logs/partial_dp/dialog/20210503/sigma0.7_norm5e-3_seed1111
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:6 -dp -partial -bs 3 --sigma 0.6 -norm 5e-3 --epochs 50 --seed 1111 2>&1 | tee logs/partial_dp/dialog/20210503/sigma0.6_norm5e-3_seed1111

# 1. canary insertion for no dp
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210503/220336 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210503/220400 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210503/220414 --cuda cuda:2 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210503/220437 --cuda cuda:3 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210503/220449 --cuda cuda:4 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed300.csv

# 2. membership for no dp
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220336 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220400 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220414 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220437 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220449 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for no dp, first name
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220336 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220400 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220414 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220437 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210503/220449 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 3. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/nodp/20210503/220336 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:1 -model_dir model/nodp/20210503/220400 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:2 -model_dir model/nodp/20210503/220414 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:3 -model_dir model/nodp/20210503/220437 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:4 -model_dir model/nodp/20210503/220449 --data_type dial --data data/simdial




# repeat 5 times
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 300 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed300
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 22 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed22

python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 0 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed0
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 123 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed123

python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 1111 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed1111



# canary insertion for no dp
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/nodp_seed300.csv

# canary insertion for partial dp
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed300.csv

# canary insertion for dp
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint  --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/nodp_seed300.csv

################################################
# membership running, dialog
################################################
# no dp
python attacks/mem_inference.py  --data_type dial --data data/simdial -bs 64 --N 1000 --checkpoint model/nodp/20210427/214226 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/test.csv