#############################################
# training
#############################################
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


# repeat 5 times
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 300 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed300
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 22 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed22

python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 0 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed0
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 123 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed123

python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 --seed 1111 2>&1 | tee logs/partial_dp/dialog/20210503/dialog_server/sigma0.7_norm5e-3_seed1111



# full dp, repeat 5 times
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 --seed 0 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs_seed0
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 --seed 123 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs_seed123
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 --seed 22 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs_seed22
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 --seed 300 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs_seed300



#############################################
# after training
#############################################

################################
# no dp
################################

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








###########################
# for partial dp
###########################
# 1. membership for partial dp
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230732 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix/lr0.1_sigma0.7_norm0.005_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230450 --cuda cuda:1 --outputf attacks/membership_inference/dialog/partialdp/final_fix/lr0.1_sigma0.7_norm0.005_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230904 --cuda cuda:2 --outputf attacks/membership_inference/dialog/partialdp/final_fix/lr0.1_sigma0.7_norm0.005_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230952 --cuda cuda:3 --outputf attacks/membership_inference/dialog/partialdp/final_fix/lr0.1_sigma0.7_norm0.005_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210504/100507/ --cuda cuda:4 --outputf attacks/membership_inference/dialog/partialdp/final_fix/lr0.1_sigma0.7_norm0.005_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for partial dp, first name
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230732 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname/lr0.1_sigma0.7_norm0.005_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230450 --cuda cuda:1 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname/lr0.1_sigma0.7_norm0.005_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230904 --cuda cuda:2 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname/lr0.1_sigma0.7_norm0.005_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230952 --cuda cuda:3 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname/lr0.1_sigma0.7_norm0.005_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210504/100507/ --cuda cuda:4 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname/lr0.1_sigma0.7_norm0.005_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
# membership for partial dp, first name, test from US census
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230732 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_fromUSCensus/lr0.1_sigma0.7_norm0.005_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230450 --cuda cuda:1 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_fromUSCensus/lr0.1_sigma0.7_norm0.005_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230904 --cuda cuda:2 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_fromUSCensus/lr0.1_sigma0.7_norm0.005_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210503/230952 --cuda cuda:3 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_fromUSCensus/lr0.1_sigma0.7_norm0.005_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/partialdp/20210504/100507 --cuda cuda:4 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_fromUSCensus/lr0.1_sigma0.7_norm0.005_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000



# 2. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/partialdp/20210503/230732 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:1 -model_dir model/partialdp/20210503/230450 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:2 -model_dir model/partialdp/20210503/230904 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:3 -model_dir model/partialdp/20210503/230952 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:4 -model_dir model/partialdp/20210504/100507/ --data_type dial --data data/simdial

# 3. canary insertion for no dp
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210503/230732 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/final/lr0.1_sigma0.7_norm0.005_seed0.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210503/230450 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/final/lr0.1_sigma0.7_norm0.005_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210503/230904 --cuda cuda:2 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/final/lr0.1_sigma0.7_norm0.005_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210503/230952 --cuda cuda:3 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/final/lr0.1_sigma0.7_norm0.005_seed300.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210504/100507/ --cuda cuda:4 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/final/lr0.1_sigma0.7_norm0.005_seed1111.csv



###########################
# for dp
###########################
# 1. membership for dp
python attacks/mem_inference.py   --checkpoint model/dp/20210502/123625/ --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix/lr0.1_sigma0.6_norm0.05_seed1111_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for dp, first name
python attacks/mem_inference.py   --checkpoint model/dp/20210502/123625/ --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix_firstname/lr0.1_sigma0.6_norm0.05_seed1111_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 2. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:1 -model_dir model/dp/20210502/123625/ --data_type dial --data data/simdial

# 3. canary insertion 
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210502/123625/ --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/dp/final/lr0.1_sigma0.6_norm0.05_seed1111_100epochs.csv 


###############################
# seed 300
# 1. membership for dp
python attacks/mem_inference.py   --checkpoint model/dp/20210512/232905 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix/lr0.1_sigma0.6_norm0.05_seed300_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for dp, first name
python attacks/mem_inference.py   --checkpoint model/dp/20210512/232905 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix_firstname/lr0.1_sigma0.6_norm0.05_seed300_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 2. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 24 --cuda cuda:1 -model_dir model/dp/20210512/232905 --data_type dial --data data/simdial

# 3. canary insertion 
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210512/232905 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/dp/final/lr0.1_sigma0.6_norm0.05_seed300_100epochs.csv 

###############################
# seed 123
# 1. membership for dp
python attacks/mem_inference.py   --checkpoint model/dp/20210508/123015 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix/lr0.1_sigma0.6_norm0.05_seed123_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for dp, first name
python attacks/mem_inference.py   --checkpoint model/dp/20210508/123015 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix_firstname/lr0.1_sigma0.6_norm0.05_seed123_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 2. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 24 --cuda cuda:1 -model_dir model/dp/20210508/123015 --data_type dial --data data/simdial

# 3. canary insertion 
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210508/123015 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/dp/final/lr0.1_sigma0.6_norm0.05_seed123_100epochs.csv 

###############################
# seed 0
# 1. membership for dp
conda activate privacy
python attacks/mem_inference.py   --checkpoint model/dp/20210506/022233 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix/lr0.1_sigma0.6_norm0.05_seed0_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for dp, first name
python attacks/mem_inference.py   --checkpoint model/dp/20210506/022233 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix_firstname/lr0.1_sigma0.6_norm0.05_seed0_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 2. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 24 --cuda cuda:1 -model_dir model/dp/20210506/022233 --data_type dial --data data/simdial

# 3. canary insertion 
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210506/022233 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/dp/final/lr0.1_sigma0.6_norm0.05_seed0_100epochs.csv 

###############################
# seed 22
# 1. membership for dp
python attacks/mem_inference.py   --checkpoint model/dp/20210511/110444 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix/lr0.1_sigma0.6_norm0.05_seed22_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for dp, first name
python attacks/mem_inference.py   --checkpoint model/dp/20210511/110444 --cuda cuda:1 --outputf attacks/membership_inference/dialog/dp/final_fix_firstname/lr0.1_sigma0.6_norm0.05_seed22_100epochs.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 2. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 24 --cuda cuda:1 -model_dir model/dp/20210511/110444 --data_type dial --data data/simdial

# 3. canary insertion 
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210511/110444 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/dp/final/lr0.1_sigma0.6_norm0.05_seed22_100epochs.csv 



# baseline retrain, on language
# screen -R dial0
python -u main.py -bs 4 --lr 0.5 --data data/simdial --data_type dial --cuda cuda:0  --seed 1111 --epochs 50 -save_epoch_num 5 2>&1 | tee logs/nodp/dialog/20210511/add10/dialog_bs4_lr0.5_seed1111.log
# screen -R dial1
python -u main.py -bs 4 --lr 0.5 --data data/simdial --data_type dial --cuda cuda:1  --seed 0 --epochs 50 -save_epoch_num 5 2>&1 | tee logs/nodp/dialog/20210511/add10/dialog_bs4_lr0.5_seed0.log
# screen -R dial2
python -u main.py -bs 4 --lr 0.5 --data data/simdial --data_type dial --cuda cuda:2  --seed 123 --epochs 50 -save_epoch_num 5 2>&1 | tee logs/nodp/dialog/20210511/add10/dialog_bs4_lr0.5_seed123.log
# screen -R dial3
python -u main.py -bs 4 --lr 0.5 --data data/simdial --data_type dial --cuda cuda:3  --seed 22 --epochs 50 -save_epoch_num 5 2>&1 | tee logs/nodp/dialog/20210511/add10/dialog_bs4_lr0.5_seed22.log
# screen -R dial4
python -u main.py -bs 4 --lr 0.5 --data data/simdial --data_type dial --cuda cuda:4  --seed 300 --epochs 50 -save_epoch_num 5 2>&1 | tee logs/nodp/dialog/20210511/add10/dialog_bs4_lr0.5_seed300.log

# 1. canary insertion for no dp
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed300.csv

# 2. membership for no dp
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for no dp, first name
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
# membership for no dp, first name, US census
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_fromUSCensus/lr0.5_bs4/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_fromUSCensus/lr0.5_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_fromUSCensus/lr0.5_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_fromUSCensus/lr0.5_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_fromUSCensus/lr0.5_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name-US-census/test --path1 attacks/membership_inference/candidates/dialog-first-name-US-census/train -bs 64 --N 1000


# 3. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/nodp/20210511/201522 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:1 -model_dir model/nodp/20210511/201529 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:2 -model_dir model/nodp/20210511/201538 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:3 -model_dir model/nodp/20210511/201546 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:4 -model_dir model/nodp/20210511/201555 --data_type dial --data data/simdial





# 
# baseline retrain, on nlp
# screen -R dial0
python -u main.py -bs 8 --lr 1 --data data/simdial --data_type dial --cuda cuda:1  --seed 1111 --epochs 50 -save_epoch_num 3 2>&1 | tee logs/nodp/dialog/20210511/add10_lr1_bs8/dialog_bs8_lr1_seed1111.log
# screen -R dial1
python -u main.py -bs 8 --lr 1 --data data/simdial --data_type dial --cuda cuda:3  --seed 0 --epochs 50 -save_epoch_num 3 2>&1 | tee logs/nodp/dialog/20210511/add10_lr1_bs8/dialog_bs8_lr1_seed0.log
# screen -R dial2
python -u main.py -bs 8 --lr 1 --data data/simdial --data_type dial --cuda cuda:4  --seed 123 --epochs 50 -save_epoch_num 3 2>&1 | tee logs/nodp/dialog/20210511/add10_lr1_bs8/dialog_bs8_lr1_seed123.log
# screen -R dial3
python -u main.py -bs 8 --lr 1 --data data/simdial --data_type dial --cuda cuda:6  --seed 22 --epochs 50 -save_epoch_num 3 2>&1 | tee logs/nodp/dialog/20210511/add10_lr1_bs8/dialog_bs8_lr1_seed22.log
# screen -R dial4
# python -u main.py -bs 8 --lr 1 --data data/simdial --data_type dial --cuda cuda:4  --seed 300 --epochs 50 -save_epoch_num 3 2>&1 | tee logs/nodp/dialog/20210511/add10_lr1_bs8/dialog_bs8_lr1_seed300.log

# 1. canary insertion for no dp
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/bs32/lr0.5_bs4/nodp_seed300.csv

# 2. membership for no dp
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr0.5_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for no dp, first name
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201522 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201529 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201538 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201546 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210511/201555 --cuda cuda:4 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr0.5_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 3. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/nodp/20210511/201522 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:1 -model_dir model/nodp/20210511/201529 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:2 -model_dir model/nodp/20210511/201538 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:3 -model_dir model/nodp/20210511/201546 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:4 -model_dir model/nodp/20210511/201555 --data_type dial --data data/simdial




#########
model/nodp/20210515/131136
model/nodp/20210515/131151
model/nodp/20210515/131209
model/nodp/20210515/131323
model/nodp/20210515/131125




# 1. canary insertion for no dp
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210515/131136 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/lr2_bs4/nodp_seed0.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210515/131151 --cuda cuda:1 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/lr2_bs4/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210515/131209 --cuda cuda:2 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/lr2_bs4/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210515/131323 --cuda cuda:3 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/lr2_bs4/nodp_seed300.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210515/131125 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/lr2_bs4/nodp_seed1111.csv

# 2. membership for no dp
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131136 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr2_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131151 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr2_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131209 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr2_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131323 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr2_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131125 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix/lr2_bs4/nodp_seed111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000
# membership for no dp, first name
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131136 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr2_bs4/nodp_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131151 --cuda cuda:1 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr2_bs4/nodp_seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131209 --cuda cuda:2 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr2_bs4/nodp_seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131323 --cuda cuda:3 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr2_bs4/nodp_seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000
python attacks/mem_inference.py   --checkpoint model/nodp/20210515/131125 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/lr2_bs4/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000


# 3. adjust ppl
python -u scripts/adjust_ppl_acc.py -bs 16 --cuda cuda:0 -model_dir model/nodp/20210515/131136 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 16 --cuda cuda:1 -model_dir model/nodp/20210515/131151 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 16 --cuda cuda:2 -model_dir model/nodp/20210515/131209 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 16 --cuda cuda:3 -model_dir model/nodp/20210515/131323 --data_type dial --data data/simdial
python -u scripts/adjust_ppl_acc.py -bs 16 --cuda cuda:0 -model_dir model/nodp/20210515/131125 --data_type dial --data data/simdial

# to generate the debug csv, and pick the names
python attacks/mem_inference.py --debug  --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname/test/nodp_seed1111_lr0.5_bs4_to_generate_debug.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000  --checkpoint model/nodp/20210511/201522/data-simdial_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50260_bs-4_bptt-35_lr-0.5_dp-False_partial-False_0hidden-False.pt_ppl-3.2092630_acc-0.75526_epoch-50_ep-0.000_dl-0_ap-0.00


python attacks/mem_inference.py  --N 400  --checkpoint model/nodp/20210511/201522/ --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_picked/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names/test --path1 attacks/membership_inference/candidates/dialog-pick-names/train -bs 64
python attacks/mem_inference.py  --N 300  --checkpoint model/nodp/20210511/201522/ --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_picked_without_ground_truth_help/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names-without-ground-truth-help/test --path1 attacks/membership_inference/candidates/dialog-pick-names-without-ground-truth-help/train -bs 64

python attacks/mem_inference.py  --N 400  --checkpoint model/nodp/20210511/201529 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_picked/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names/test --path1 attacks/membership_inference/candidates/dialog-pick-names/train -bs 64
python attacks/mem_inference.py  --N 400  --checkpoint model/nodp/20210511/201522/ --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_picked/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names/test --path1 attacks/membership_inference/candidates/dialog-pick-names/train -bs 64
python attacks/mem_inference.py  --N 400  --checkpoint model/nodp/20210511/201522/ --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_picked/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names/test --path1 attacks/membership_inference/candidates/dialog-pick-names/train -bs 64
python attacks/mem_inference.py  --N 400  --checkpoint model/nodp/20210511/201522/ --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix_firstname_picked/nodp_seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names/test --path1 attacks/membership_inference/candidates/dialog-pick-names/train -bs 64

python attacks/mem_inference.py  --N 400 --checkpoint model/partialdp/20210503/230732 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_picked/lr0.1_sigma0.7_norm0.005_seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names/test --path1 attacks/membership_inference/candidates/dialog-pick-names/train -bs 64



# for partial dp, generate debug
python attacks/mem_inference.py --debug  --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname/test/seed1111_to_generate_debug.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-first-name/test --path1 attacks/membership_inference/candidates/dialog-first-name/train -bs 64 --N 1000  --checkpoint model/partialdp/20210504/100507/data-simdial_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50260_bs-3_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.7_norm-0.005_dl-8e-05.pt_ppl-9.6761020_acc-0.70224_epoch-50_ep-2.744_dl-8e-05_ap-6.70
## without ground truth help
model/partialdp/20210503/230732
model/partialdp/20210503/230450
model/partialdp/20210503/230904
model/partialdp/20210503/230952
python attacks/mem_inference.py  --N 300  --checkpoint model/partialdp/20210504/100507 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_picked_without_ground_truth_help/seed1111.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/test --path1 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/train -bs 64
python attacks/mem_inference.py  --N 300  --checkpoint model/partialdp/20210503/230732 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_picked_without_ground_truth_help/seed0.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/test --path1 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/train -bs 64
python attacks/mem_inference.py  --N 300  --checkpoint model/partialdp/20210503/230450 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_picked_without_ground_truth_help/seed123.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/test --path1 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/train -bs 64
python attacks/mem_inference.py  --N 300  --checkpoint model/partialdp/20210503/230904 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_picked_without_ground_truth_help/seed22.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/test --path1 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/train -bs 64
python attacks/mem_inference.py  --N 300  --checkpoint model/partialdp/20210503/230952 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix_firstname_picked_without_ground_truth_help/seed300.csv --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/test --path1 attacks/membership_inference/candidates/dialog-pick-names-partialdp-without-ground-truth-help/train -bs 64
