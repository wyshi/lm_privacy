python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial -partial_hidden_zero --sigma 0.35 2>&1 | tee logs/partial_dp/20210411/1201/hiddenzero_lr0.1_sigma0.35
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 --sigma 0.5 -norm 0.02 2>&1 | tee logs/dp/20210411/1511/lr0.1_sigma0.5_norm0.02

python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial --sigma 0.5 -norm 0.02 --epochs 1 2>&1 | tee logs/partial_dp/20210413/1147/lr0.1_sigma0.5_norm0.02_epoch1



python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial --sigma 0.5 -norm 0.02 --epochs 1 2>&1 | tee logs/partial_dp/20210413/1457/lr0.1_sigma0.5_norm0.02_epoch1_save_every_batch


# repeat 4 times, partial dp
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 0.02 --seed 0 2>&1 | tee logs/partial_dp/20210414/1211/nohidden_lr0.1_norm0.02_seed0 # screen partialdp2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial -norm 0.02 --seed 123 2>&1 | tee logs/partial_dp/20210414/1211/nohidden_lr0.1_norm0.02_seed123 # screen partialdp
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 0.02 --seed 22 2>&1 | tee logs/partial_dp/20210414/1211/nohidden_lr0.1_norm0.02_seed22 # screen partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 0.02 --seed 300 2>&1 | tee logs/partial_dp/20210414/1211/nohidden_lr0.1_norm0.02_seed300 # screen partialdp4

# repeat 2 times, dp
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -norm 0.02 --seed 0 2>&1 | tee logs/dp/20210414/1224/lr0.1_sigma0.5_norm0.02_seed0 # screen dp
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -norm 0.02 --seed 123 2>&1 | tee logs/dp/20210414/1224/lr0.1_sigma0.5_norm0.02_seed123 # screen dp2

# partial dp, norm=0.01
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 0.01 2>&1 | tee logs/partial_dp/20210414/1128/nohidden_lr0.1_norm0.01 # screen partialdp4

# join canary insertion csv with privacy
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210411/122315/ -log logs/partial_dp/20210411/1219/nohidden_lr0.1_norm0.02 -csv attacks/canary_insertion/partial_dp_sigma05_lr01_norm002.csv
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210413/145857/ -log logs/partial_dp/20210413/1457/lr0.1_sigma0.5_norm0.02_epoch1_save_every_batch  -csv attacks/canary_insertion/partial_dp_sigma05_lr01_norm002_epoch1_save_every_batch.csv
python attacks/join_epsilon_with_attack.py -ckpt model/dp/20210409/223157/ -log logs/dp/20210409/2329/lstm.log -csv attacks/canary_insertion/dp_sigma05_lr005_norm01.csv
python attacks/join_epsilon_with_attack.py -ckpt model/dp/20210411/151450/ -log logs/dp/20210411/1511/lr0.1_sigma0.5_norm0.02 -csv attacks/canary_insertion/dp_sigma05_lr01_norm002.csv
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210413/145857/ -log logs/partial_dp/20210413/1457/lr0.1_sigma0.5_norm0.02_epoch1_save_every_batch -csv attacks/membership_inference/partial_dp_lr01_sigma05_norm002_1000_epoch1_save_every_batch.csv
python attacks/join_epsilon_with_attack.py -ckpt model/nodp/20210409/185850/ -log logs/nodp/20210409/1855/lstm.log -csv attacks/canary_insertion/nodp_10insertion_6digits.csv
python attacks/join_epsilon_with_attack.py -ckpt model/nodp/20210413/123134/ -log logs/nodp/20210413/1107/epoch1.log -csv attacks/canary_insertion/nodp_10insertion_epoch1_bs32.csv
python attacks/join_epsilon_with_attack.py -ckpt model/nodp/20210413/115956/ -log logs/nodp/20210413/1121/bs16.log -csv attacks/canary_insertion/nodp_10insertion_epoch1_bs16.csv
python attacks/join_epsilon_with_attack.py -ckpt model/dp/20210409/222642/ -log logs/dp/20210409/2225/lstm.log -csv attacks/canary_insertion/dp_sigma05_lr01_norm01.csv
python attacks/join_epsilon_with_attack.py -ckpt model/dp/20210411/123315/ -log logs/dp/20210411/1232/sigma0.25_lr0.1_norm0.1 -csv attacks/canary_insertion/dp_sigma025_lr01_norm01.csv
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210410/093833/ -log logs/partial_dp/20210410/0937/sigma0.25 -csv attacks/canary_insertion/partial_dp_sigma025_lr01_norm01.csv


# join membership csv with privacy
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210410/093833/ -log logs/partial_dp/20210410/0937/sigma0.25 -csv attacks/membership_inference/partialdp_lr01_sigma025_norm01_1000.csv
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210411/122315/  -log logs/partial_dp/20210411/1219/nohidden_lr0.1_norm0.02 -csv attacks/membership_inference/partial_dp_lr01_sigma05_norm002_1000_epoch50.csv
python attacks/join_epsilon_with_attack.py -ckpt model/partialdp/20210413/145857/ -log logs/partial_dp/20210413/1457/lr0.1_sigma0.5_norm0.02_epoch1_save_every_batch -csv attacks/membership_inference/partial_dp_lr01_sigma05_norm002_1000_epoch1_save_every_batch_with_privacy.csv
python attacks/join_epsilon_with_attack.py -ckpt model/dp/20210409/223157/ -log logs/dp/20210409/2329/lstm.log -csv attacks/membership_inference/dp_lr005_sigma05_norm01_1000.csv
python attacks/join_epsilon_with_attack.py -ckpt model/dp/20210411/151450/ -log logs/dp/20210411/1511/lr0.1_sigma0.5_norm0.02 -csv attacks/membership_inference/dp_lr05_sigma05_norm002_1000.csv
python attacks/join_epsilon_with_attack.py -ckpt model/nodp/20210409/185850/ -log logs/nodp/20210409/1855/lstm.log -csv attacks/membership_inference/nodp_1000.csv


# membership inference attack
python attacks/mem_inference.py -ckpt model/nodp/20210409/185850/ --outputf attacks/membership_inference/nodp_1000.csv --cuda cuda:5 --N 1000 -bs 64
python attacks/mem_inference.py -ckpt model/dp/20210409/223157/ --outputf attacks/membership_inference/dp_lr005_sigma05_norm01_1000.csv --cuda cuda:5 --N 1000 -bs 64
python attacks/mem_inference.py -ckpt model/dp/20210411/151450/ --outputf attacks/membership_inference/dp_lr05_sigma05_norm002_1000.csv --cuda cuda:5 --N 1000 -bs 64
python attacks/mem_inference.py -ckpt model/partialdp/20210410/093833/ --outputf attacks/membership_inference/partialdp_lr01_sigma025_norm01_1000.csv --cuda cuda:5 --N 1000 -bs 64
