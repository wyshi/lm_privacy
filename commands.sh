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


# no dp, repeat
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:0 2>&1 | tee logs/nodp/20210416/1710/bs16.log
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:1 --seed 0 2>&1 | tee logs/nodp/20210416/1710/bs16_see0.log
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:2 --seed 123 2>&1 | tee logs/nodp/20210416/1710/bs16_seed123.log
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:4 --seed 22 2>&1 | tee logs/nodp/20210416/1710/bs16_seed22.log
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:6 --seed 300 2>&1 | tee logs/nodp/20210416/1710/bs16_seed300.log

# canary insertion for partial-dp repeat
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210414/001535/ --cuda cuda:5 --outputf attacks/canary_insertion/partial_dp_sigma05_lr01_norm002_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210414/002043/ --cuda cuda:5 --outputf attacks/canary_insertion/partial_dp_sigma05_lr01_norm002_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210414/002122/ --cuda cuda:5 --outputf attacks/canary_insertion/partial_dp_sigma05_lr01_norm002_seed22.csv

# canary insertion for dp repeat
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210414/002943/ --cuda cuda:0 --outputf attacks/canary_insertion/dp_sigma05_lr01_norm002_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210414/002746/ --cuda cuda:0 --outputf attacks/canary_insertion/p_sigma05_lr01_norm002_seed123.csv


# partial dp, parameter search
    # screen -r partialdp
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 0.01  --sigma 0.45 --seed 1111 2>&1 | tee logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.01_sigma0.45 
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:0 2>&1 | tee logs/nodp/20210416/2354/bs16.log

    # screen -r partialdp2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 0.005 --sigma 0.45 --seed 1111 2>&1 | tee logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.005_sigma0.45 
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:1 --seed 0 2>&1 | tee logs/nodp/20210416/2354/bs16_see0.log

    # screen -r partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 0.01  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.01_sigma0.5 
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:2 --seed 123 2>&1 | tee llogs/nodp/20210416/2354/bs16_seed123.log

    # screen -r partialdp4
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 0.005 --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.005_sigma0.5 
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:5 --seed 22 2>&1 | tee logs/nodp/20210416/2354/bs16_seed22.log

    # screen -R partialdp5
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 0.01   --sigma 0.55 --seed 1111 2>&1 | tee logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.01_sigma0.55 
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:4 --seed 300 2>&1 | tee logs/nodp/20210416/2354/bs16_seed300.log

    # screen -R partialdp6
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 0.005  --sigma 0.55 --seed 1111 2>&1 | tee logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.005_sigma0.55 


# parameter tunning on the norm
    # screen -R partialdp6
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 1e-3  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210418/1912/nohidden_lr0.1_norm1e-3_sigma0.5 
    # screen -R partialdp5 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 5e-4  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210418/1912/nohidden_lr0.1_norm5e-4_sigma0.5 
    # screen -R partialdp4 # rerun
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 1e-4  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210418/1912/nohidden_lr0.1_norm1e-4_sigma0.5 
    # screen -R partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 5e-5  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210418/1912/nohidden_lr0.1_norm5e-5_sigma0.5 
    # screen -R partialdp2 # rerun
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 1e-5  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210418/1912/nohidden_lr0.1_norm1e-5_sigma0.5 
    # screen -R partialdp # rerun
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 5e-6  --sigma 0.5 --seed 1111 2>&1 | tee logs/partial_dp/20210418/1912/nohidden_lr0.1_norm5e-6_sigma0.5 



# repeat 5 times for norm=1e-3
    # screen -R partialdp6
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 1e-3  --sigma 0.5 --seed 0 2>&1 | tee logs/partial_dp/20210421/1021/nohidden_lr0.1_norm1e-3_sigma0.5_seed0 
    # screen -R partialdp5 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 --seed 123 2>&1 | tee logs/partial_dp/20210421/1021/nohidden_lr0.1_norm1e-3_sigma0.5_seed123
    # screen -R partialdp4 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 1e-3  --sigma 0.5 --seed 22 2>&1 | tee logs/partial_dp/20210421/1021/nohidden_lr0.1_norm1e-3_sigma0.5_seed22
    # screen -R partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 1e-3  --sigma 0.5 --seed 300 2>&1 | tee logs/partial_dp/20210421/1021/nohidden_lr0.1_norm1e-3_sigma0.5_seed300


# repeat 5 times for norm=1e-3, dp
# screen -R dp # have run
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -norm 1e-3 --seed 1111 2>&1 | tee logs/dp/20210421/1029/lr0.1_sigma0.5_norm1e-3_seed1111 # screen dp
# repeat 5 times for norm=1e-2, dp
# screen -R dp2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -norm 1e-2 --seed 1111 2>&1 | tee logs/dp/20210422/1437/lr0.1_sigma0.5_norm1e-2_seed1111 # screen dp
# # screen -R dp3
# python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -norm 1e-2 --seed 123 2>&1 | tee logs/dp/20210421/1029/lr0.1_sigma0.5_norm1e-3_seed123 # screen dp
# # screen -R dp4
# python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -norm 1e-2 --seed 22 2>&1 | tee logs/dp/20210421/1029/lr0.1_sigma0.5_norm1e-3_seed22 # screen dp
# # screen -R dp5
# python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -norm 1e-2 --seed 300 2>&1 | tee logs/dp/20210421/1029/lr0.1_sigma0.5_norm1e-3_seed300 # screen dp


# canary for no-dp, new runs
# canary insertion for dp repeat
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210418/181252 --cuda cuda:0 --outputf attacks/canary_insertion/nodp/nodp_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210418/192226 --cuda cuda:0 --outputf attacks/canary_insertion/nodp/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210417/144949 --cuda cuda:0 --outputf attacks/canary_insertion/nodp/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210419/052248 --cuda cuda:0 --outputf attacks/canary_insertion/nodp/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210418/210231 --cuda cuda:0 --outputf attacks/canary_insertion/nodp/nodp_seed300.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210417/000742 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.45_norm0.01_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210417/000240 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.45_norm0.005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210417/000251 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.01_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210417/111003 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210417/000327 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.55_norm0.01_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210417/000344 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.55_norm0.005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210418/191438 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210419/063839 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.0005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210419/134334 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.0001_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210419/064057 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm5e-05_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210419/134345 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm1e-05_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210419/134357 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm5e-06_seed1111.csv

# membership
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/nodp/20210418/181252 --cuda cuda:0 --outputf attacks/membership_inference/nodp/nodp_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/nodp/20210418/192226 --cuda cuda:0 --outputf attacks/membership_inference/nodp/nodp_seed0.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/nodp/20210417/144949 --cuda cuda:0 --outputf attacks/membership_inference/nodp/nodp_seed123.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/nodp/20210419/052248 --cuda cuda:0 --outputf attacks/membership_inference/nodp/nodp_seed22.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/nodp/20210418/210231 --cuda cuda:0 --outputf attacks/membership_inference/nodp/nodp_seed300.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000742 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.45_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000240 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.45_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000251 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/111003 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000327 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.55_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000344 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.55_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210418/191438 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/063839 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm0.0005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134334 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm0.0001_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/064057 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm5e-05_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134345 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm1e-05_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134357 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/lr0.1_sigma0.5_norm5e-06_seed1111.csv



# parameter search on the sigma and norm
    # screen -R partialdp 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 5e-3  --sigma 0.1 --seed 1111 2>&1 | tee logs/partial_dp/20210423/nohidden_lr0.1_norm5e-3_sigma0.1_seed1111 
    # screen -R partialdp5 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 0.25  --sigma 0.1 --seed 1111 2>&1 | tee logs/partial_dp/20210423/nohidden_lr0.1_norm0.25_sigma0.1_seed1111  
    # screen -R partialdp4 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 5e-3  --sigma 0.05 --seed 1111 2>&1 | tee logs/partial_dp/20210423/nohidden_lr0.1_norm5e-3_sigma0.05_seed1111  
    # screen -R partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 0.25  --sigma 0.05 --seed 1111 2>&1 | tee logs/partial_dp/20210423/nohidden_lr0.1_norm0.25_sigma0.05_seed1111  
    # screen -R partialdp2 # notyet
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 5e-3  --sigma 0.01 --seed 1111 2>&1 | tee logs/partial_dp/20210423/nohidden_lr0.1_norm5e-3_sigma0.01_seed1111 
    # screen -R partialdp6 # not yet
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 0.25  --sigma 0.01 --seed 1111 2>&1 | tee logs/partial_dp/20210423/nohidden_lr0.1_norm0.25_sigma0.01_seed1111  


# resume 50 epochs for sigma=0.5, on dialog
# screen -R resume1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 --seed 1111 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210418/191438/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-161.1260678_acc-0.33143_epoch-50_ep-5.376_dl-8e-05_ap-3.60 2>&1 | tee logs/partial_dp/20210423/resume/nohidden_lr0.1_norm1e-3_sigma0.5_seed1111  
# screen -R resume2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 --seed 0 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210421/123500/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-166.0325998_acc-0.30857_epoch-50_ep-5.376_dl-8e-05_ap-3.60 2>&1 | tee logs/partial_dp/20210423/resume/nohidden_lr0.1_norm1e-3_sigma0.5_seed0  


# dp param search, on dialog
# screen -R dp1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -norm 0.25 --seed 1111 2>&1 | tee logs/dp/20210423/param_search/lr0.1_sigma0.5_norm0.25_seed1111 
# screen -R dp2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -norm 0.5  --seed 1111 2>&1 | tee logs/dp/20210423/param_search/lr0.1_sigma0.5_norm0.5_seed1111 


# dp repeat 5 times, on interaction
# screen -R dp1
python -u main.py --epochs 100 -bs 7 --lr 0.05 -dp --cuda cuda:1 -norm 0.1 --seed 1111 2>&1 | tee logs/dp/20210424/repeat/lr0.05_sigma0.5_norm0.1_seed1111 
# screen -R dp2
python -u main.py --epochs 100 -bs 7 --lr 0.05 -dp --cuda cuda:2 -norm 0.1 --seed 0 2>&1 | tee logs/dp/20210424/repeat/lr0.05_sigma0.5_norm0.1_seed0 
# screen -R dp3
python -u main.py --epochs 100 -bs 7 --lr 0.05 -dp --cuda cuda:4 -norm 0.1 --seed 123 2>&1 | tee logs/dp/20210424/repeat/lr0.05_sigma0.5_norm0.1_seed123
# screen -R dp4
python -u main.py --epochs 100 -bs 7 --lr 0.05 -dp --cuda cuda:5 -norm 0.1 --seed 22 2>&1 | tee logs/dp/20210424/repeat/lr0.05_sigma0.5_norm0.1_seed22 
# screen -R dp4
python -u main.py --epochs 100 -bs 7 --lr 0.05 -dp --cuda cuda:6 -norm 0.1 --seed 300 2>&1 | tee logs/dp/20210424/repeat/lr0.05_sigma0.5_norm0.1_seed300 


# parameter search on the much smaller sigma and norm, dialog
    # screen -R partialdp1 # 
python -u main.py -bs 7 --lr 0.1 -dp --epochs 100 --cuda cuda:1 -partial -norm 1e-3  --sigma 0.1 --seed 1111 2>&1 | tee logs/partial_dp/20210425/param_search_new/nohidden_lr0.1_norm1e-3_sigma0.1_seed1111 
    # screen -R partialdp2 # 
python -u main.py -bs 7 --lr 0.1 -dp --epochs 100 --cuda cuda:1 -partial -norm 2.5e-3  --sigma 0.1 --seed 1111 2>&1 | tee logs/partial_dp/20210425/param_search_new/nohidden_lr0.1_norm2.5e-3_sigma0.1_seed1111  
    # screen -R partialdp3 # not yet 
python -u main.py -bs 7 --lr 0.1 -dp --epochs 100 --cuda cuda:1 -partial -norm 7.5e-3  --sigma 0.1 --seed 1111 2>&1 | tee logs/partial_dp/20210425/param_search_new/nohidden_lr0.1_norm7.5e-3_sigma0.1_seed1111  
    # screen -R partialdp4 # not yet 
python -u main.py -bs 7 --lr 0.1 -dp --epochs 100 --cuda cuda:1 -partial -norm 1e-2  --sigma 0.1 --seed 1111 2>&1 | tee logs/partial_dp/20210425/param_search_new/nohidden_lr0.1_norm1e-2_sigma0.1_seed1111  
python -u main.py -bs 7 --lr 0.1 -dp --epochs 100 --cuda cuda:1 -partial -norm 0.25  --sigma 0.005 --seed 1111 2>&1 | tee logs/partial_dp/20210425/param_search/nohidden_lr0.1_norm0.25_sigma0.005_seed1111  


# resume 50 epochs for sigma=0.01, norm=0.25 on interaction
# screen -R resume1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 5e-3  --sigma 0.1 --seed 1111 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210423/111019/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.1_norm-0.005_dl-8e-05.pt_ppl-151.1701144_acc-0.33714_epoch-50_ep-132047.094_dl-8e-05_ap-1.10 2>&1 | tee logs/partial_dp/20210425/resume/lr0.1_norm1e-3_sigma0.1_seed1111  


# missing digit, baseline, on interaction
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/missing_digits --cuda cuda:3 2>&1 | tee logs/nodp/normalized/20210426/lstm.log
# screen -R nodp2
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/missing_digits --cuda cuda:3 --seed 0 2>&1 | tee logs/nodp/normalized/20210426/lstm_seed0.log
# screen -R nodp3
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/missing_digits --cuda cuda:3 --seed 123 2>&1 | tee logs/nodp/normalized/20210426/lstm_seed123.log
# screen -R nodp4
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/missing_digits --cuda cuda:3 --seed 22 2>&1 | tee logs/nodp/normalized/20210426/lstm_seed22.log
# screen -R nodp5
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/missing_digits --cuda cuda:3 --seed 300 2>&1 | tee logs/nodp/normalized/20210426/lstm_seed300.log

# missing digit, partial dp, on dialog
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b --epochs 100 --seed 1111 2>&1 | tee logs/partial_dp/missed/20210426/lr0.1_sigm0.5_norm1e-3_seed1111_miss10.log
# missing digit, partial dp, on interaction
### screen -R partialdp1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b --epochs 100 --seed 0 2>&1 | tee logs/partial_dp/missed/20210426/lr0.1_sigm0.5_norm1e-3_seed0_miss10.log
### screen -R partialdp2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b --epochs 100 --seed 123 2>&1 | tee logs/partial_dp/missed/20210426/lr0.1_sigm0.5_norm1e-3_seed123_miss10.log
### screen -R partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b --epochs 100 --seed 22 2>&1 | tee logs/partial_dp/missed/20210426/lr0.1_sigm0.5_norm1e-3_seed22_miss10.log
### screen -R partialdp4
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b --epochs 100 --seed 300 2>&1 | tee logs/partial_dp/missed/20210426/lr0.1_sigm0.5_norm1e-3_seed300_miss10.log


# dialog, test, on dialog server
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 1 --sigma 0.5 -norm 1e-3  2>&1 | tee logs/partial_dp/dialog/20210426/sigma0.5_norm1e-3



# resume 50 epochs for sigma=0.5, on interaction
# screen -R resume1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 1e-3  --sigma 0.5 --seed 123 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210421/123511/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-159.3748673_acc-0.33714_epoch-50_ep-5.376_dl-8e-05_ap-3.60 2>&1 | tee logs/partial_dp/20210423/resume/nohidden_lr0.1_norm1e-3_sigma0.5_seed123  
# screen -R resume2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 1e-3  --sigma 0.5 --seed 22 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210421/123522/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-164.0274903_acc-0.32571_epoch-50_ep-5.375_dl-8e-05_ap-3.60 2>&1 | tee logs/partial_dp/20210423/resume/nohidden_lr0.1_norm1e-3_sigma0.5_seed22  
# resume 50 epochs for sigma=0.5, on dialog
# screen -R resume3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 1e-3  --sigma 0.5 --seed 300 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210421/123530/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-187.3671091_acc-0.31429_epoch-50_ep-5.376_dl-8e-05_ap-3.60 2>&1 | tee logs/partial_dp/20210423/resume/nohidden_lr0.1_norm1e-3_sigma0.5_seed300  
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 --seed 300 -resume -resume_from_epoch_num 50 -resume_from model/partialdp/20210421/123530/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-187.3671091_acc-0.31429_epoch-50_ep-5.376_dl-8e-05_ap-3.60 2>&1 | tee logs/partial_dp/20210423/resume/nohidden_lr0.1_norm1e-3_sigma0.5_seed300  


# dialog experiment, baseline
# screen -R dialog_nodp
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:3 2>&1 | tee logs/nodp/dialog/20210427/dialog_bs16.log


# dialog, parameter tuning, on dialog server
# screen -R dialog_partialdp
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.5 -norm 1e-3 --epochs 100 2>&1 | tee logs/partial_dp/dialog/20210428/sigma0.5_norm1e-3
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.5 -norm 1e-2 --epochs 100 2>&1 | tee logs/partial_dp/dialog/20210428/sigma0.5_norm1e-2



# canary, for normalized experiment, baseline, on interaction
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210426/182422 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized/nodp_normalized_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210426/234106 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized/nodp_normalized_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210426/234125 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized/nodp_normalized_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210426/234146 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized/nodp_normalized_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210426/234201 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized/nodp_normalized_seed300.csv

# canary, for param search, on interaction
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210425/150805 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.1_norm0.005_seed1111_resume_50epochs.csv


# canary, dp, on interaction
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210425/151315 --cuda cuda:3 --outputf attacks/canary_insertion/dp/lr0.05_sigma0.5_norm0.1_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210425/151340 --cuda cuda:3 --outputf attacks/canary_insertion/dp/lr0.05_sigma0.5_norm0.1_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210425/151359 --cuda cuda:3 --outputf attacks/canary_insertion/dp/lr0.05_sigma0.5_norm0.1_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210425/151417 --cuda cuda:3 --outputf attacks/canary_insertion/dp/lr0.05_sigma0.5_norm0.1_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210425/151429 --cuda cuda:5 --outputf attacks/canary_insertion/dp/lr0.05_sigma0.5_norm0.1_seed300.csv


# canary insertion, partial dp, on interaction
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210421/123500 --cuda cuda:5 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210421/123511 --cuda cuda:5 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210421/123522 --cuda cuda:5 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210421/123530 --cuda cuda:6 --outputf attacks/canary_insertion/partialdp/lr0.1_sigma0.5_norm0.001_seed300.csv

# canary insertion, partial dp, on interaction, resume
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/213802 --cuda cuda:6 --outputf attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed123_resume50epochs.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/213754 --cuda cuda:6 --outputf attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed22_resume50epochs.csv




# dialog, parameter tuning, on dialog server
# screen -R dialog_partialdp
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 1e-1 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210429/sigma0.7_norm1e-1
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -partial -bs 3 --sigma 0.7 -norm 1e-2 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210429/sigma0.7_norm1e-2
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.6 -norm 1e-1 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210429/sigma0.6_norm1e-1
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.6 -norm 1e-2 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210429/sigma0.6_norm1e-2



# canary, on dialog, resume
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/221533 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed0_resume50epochs.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/213900 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/resume/lr0.1_sigma0.5_norm0.001_seed300_resume50epochs.csv

# dialog, parameter tuning, on dialog server, full dp
# screen -R dialog_dp
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 3 --sigma 0.6 -norm 1e-2 --epochs 50 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 50 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1

# screen -R dialog_partialdp
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210430/sigma0.7_norm5e-3
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.7 -norm 5e-2 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210430/sigma0.7_norm5e-2
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.7 -norm 1e-3 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210430/sigma0.7_norm1e-3
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:6 -dp -partial -bs 3 --sigma 0.7 -norm 5e-4 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210430/sigma0.7_norm5e-4



# dialog, parameter tuning, on dialog server, full dp
# screen -R dialog_dp
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 3 --sigma 0.6 -norm 1e-2 --epochs 50 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 3 --sigma 0.6 -norm 5e-2 --epochs 50 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm5e-2


# dialog, full dp, 100 epochs
# python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 50 -resume -resume_from_epoch_num 50 -resume_from model/dp/20210501/095703/data-simdial_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50260_bs-1_bptt-35_lr-0.1_dp-True_partial-False_0hidden-False_sigma-0.6_norm-0.01_dl-8e-05.pt_ppl-20.8570108_acc-0.61681_epoch-50_ep-2.232_dl-8e-05_ap-5.90 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_resume50epochs
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:1 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs


# on interaction server, canary, for missed partial dp
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/211240 --cuda cuda:1 --outputf attacks/canary_insertion/partialdp_missed/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/211322 --cuda cuda:2 --outputf attacks/canary_insertion/partialdp_missed/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/211339 --cuda cuda:4 --outputf attacks/canary_insertion/partialdp_missed/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210427/211411 --cuda cuda:5 --outputf attacks/canary_insertion/partialdp_missed/lr0.1_sigma0.5_norm0.001_seed300.csv


# dialog experiment, baseline, on interaction server
# screen -R dialog_nodp
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:1 --log-interval 10 --seed 1111 2>&1 | tee logs/nodp/dialog/20210501/dialog_bs16_seed1111.log
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:1 --log-interval 10 --seed 0 2>&1 | tee logs/nodp/dialog/20210501/dialog_bs16_seed0.log
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:2 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210501/dialog_bs16_seed123.log
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:4 --log-interval 10 --seed 22 2>&1 | tee logs/nodp/dialog/20210501/dialog_bs16_seed22.log
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:5 --log-interval 10 --seed 300 2>&1 | tee logs/nodp/dialog/20210501/dialog_bs16_seed300.log

# dialog canary insertion
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210429/234246 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/lr0.1_sigma0.7_norm0.01_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210430/230016 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/partialdp/lr0.1_sigma0.7_norm0.005_seed1111.csv

# dialog, canary insertion
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210501/192118 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/nodp_seed1111.csv 
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210502/124723 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210501/192209 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210501/192230 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210501/192246 --cuda cuda:0 --data_type dial --outputf attacks/canary_insertion/dialog/nodp/nodp_seed300.csv



# dialog, nodp, on interaction
# add 20
python -u main.py -bs 16 --lr 20 --data data/simdial-20 --data_type dial --cuda cuda:0 --log-interval 10 --seed 1111 2>&1 | tee logs/nodp/dialog/20210503/add20/dialog_bs16_seed1111.log
python -u main.py -bs 16 --lr 20 --data data/simdial-20 --data_type dial --cuda cuda:1 --log-interval 10 --seed 0 2>&1 | tee logs/nodp/dialog/20210503/add20/dialog_bs16_seed0.log
python -u main.py -bs 16 --lr 20 --data data/simdial-20 --data_type dial --cuda cuda:2 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210503/add20/dialog_bs16_seed123.log
python -u main.py -bs 16 --lr 20 --data data/simdial-20 --data_type dial --cuda cuda:4 --log-interval 10 --seed 22 2>&1 | tee logs/nodp/dialog/20210503/add20/dialog_bs16_seed22.log
python -u main.py -bs 16 --lr 20 --data data/simdial-20 --data_type dial --cuda cuda:5 --log-interval 10 --seed 300 2>&1 | tee logs/nodp/dialog/20210503/add20/dialog_bs16_seed300.log
# add 30
python -u main.py -bs 16 --lr 20 --data data/simdial-30 --data_type dial --cuda cuda:0 --log-interval 10 --seed 1111 2>&1 | tee logs/nodp/dialog/20210503/add30/dialog_bs16_seed1111.log
python -u main.py -bs 16 --lr 20 --data data/simdial-30 --data_type dial --cuda cuda:1 --log-interval 10 --seed 0 2>&1 | tee logs/nodp/dialog/20210503/add30/dialog_bs16_seed0.log
python -u main.py -bs 16 --lr 20 --data data/simdial-30 --data_type dial --cuda cuda:2 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210503/add30/dialog_bs16_seed123.log
python -u main.py -bs 16 --lr 20 --data data/simdial-30 --data_type dial --cuda cuda:4 --log-interval 10 --seed 22 2>&1 | tee logs/nodp/dialog/20210503/add30/dialog_bs16_seed22.log
python -u main.py -bs 16 --lr 20 --data data/simdial-30 --data_type dial --cuda cuda:6 --log-interval 10 --seed 300 2>&1 | tee logs/nodp/dialog/20210503/add30/dialog_bs16_seed300.log
# add 10 inside
python -u main.py -bs 16 --lr 20 --data data/simdial-in-10 --data_type dial --cuda cuda:0 --log-interval 10 --seed 1111 2>&1 | tee logs/nodp/dialog/20210503/add10inside/dialog_bs16_seed1111.log
python -u main.py -bs 16 --lr 20 --data data/simdial-in-10 --data_type dial --cuda cuda:1 --log-interval 10 --seed 0 2>&1 | tee logs/nodp/dialog/20210503/add10inside/dialog_bs16_seed0.log
python -u main.py -bs 16 --lr 20 --data data/simdial-in-10 --data_type dial --cuda cuda:2 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210503/add10inside/dialog_bs16_seed123.log
python -u main.py -bs 16 --lr 20 --data data/simdial-in-10 --data_type dial --cuda cuda:4 --log-interval 10 --seed 22 2>&1 | tee logs/nodp/dialog/20210503/add10inside/dialog_bs16_seed22.log
python -u main.py -bs 16 --lr 20 --data data/simdial-in-10 --data_type dial --cuda cuda:5 --log-interval 10 --seed 300 2>&1 | tee logs/nodp/dialog/20210503/add10inside/dialog_bs16_seed300.log

################################################
# membership running, wiki
################################################
# nodp
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/181252/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final_fix/nodp_seed1111.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/192226/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final_fix/nodp_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210417/144949/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final_fix/nodp_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210419/052248/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final_fix/nodp_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/210231/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final_fix/nodp_seed300.csv

# dp
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151315 --cuda cuda:0 --outputf attacks/membership_inference/dp/final_fix/lr0.05_sigma0.5_norm0.1_seed1111.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151340 --cuda cuda:0 --outputf attacks/membership_inference/dp/final_fix/lr0.05_sigma0.5_norm0.1_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151359 --cuda cuda:0 --outputf attacks/membership_inference/dp/final_fix/lr0.05_sigma0.5_norm0.1_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151417 --cuda cuda:0 --outputf attacks/membership_inference/dp/final_fix/lr0.05_sigma0.5_norm0.1_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151429 --cuda cuda:0 --outputf attacks/membership_inference/dp/final_fix/lr0.05_sigma0.5_norm0.1_seed300.csv

# partial dp, first 50
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210418/191438 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123500 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123511 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123522 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123530 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/lr0.1_sigma0.5_norm0.001_seed300.csv


# partial dp, resume 50
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/213802 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/213754 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/213900 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed300.csv

# no dp normalized, 
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/182422 --cuda cuda:2 --outputf attacks/membership_inference/nodp_normalized/final_fix/nodp_seed1111.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234106 --cuda cuda:2 --outputf attacks/membership_inference/nodp_normalized/final_fix/nodp_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234125 --cuda cuda:2 --outputf attacks/membership_inference/nodp_normalized/final_fix/nodp_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234146 --cuda cuda:2 --outputf attacks/membership_inference/nodp_normalized/final_fix/nodp_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234201 --cuda cuda:2 --outputf attacks/membership_inference/nodp_normalized/final_fix/nodp_seed300.csv


# missed dp normalized
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211240 --cuda cuda:2 --outputf attacks/membership_inference/partialdp_missed/final_fix/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211322 --cuda cuda:2 --outputf attacks/membership_inference/partialdp_missed/final_fix/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211411 --cuda cuda:2 --outputf attacks/membership_inference/partialdp_missed/final_fix/lr0.1_sigma0.5_norm0.001_seed300.csv




# orange ones, param search
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000742 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.45_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000240 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.45_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000251 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/111003 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000327 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.55_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000344 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.55_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/063839 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm0.0005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134334 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm0.0001_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/064057 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm5e-05_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134345 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm1e-05_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134357 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.5_norm5e-06_seed1111.csv
# more param tunning
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111019 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111038 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.1_norm0.25_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111051 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.05_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111126 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.05_norm0.25_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/134414 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.01_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/183122 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/lr0.1_sigma0.01_norm0.25_seed1111.csv

# more param tunning, resume
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210425/150805 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final_fix/param_search/resume/lr0.1_sigma0.1_norm0.005_seed1111.csv

##################
# on dialog server
##################
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210423/221533 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210423/221514 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final_fix/resume/lr0.1_sigma0.5_norm0.001_seed1111.csv
# missed dp normalized, on dialog server
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210426/223009 --cuda cuda:1 --outputf attacks/membership_inference/partialdp_missed/final_fix/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211339 --cuda cuda:1 --outputf attacks/membership_inference/partialdp_missed/final_fix/lr0.1_sigma0.5_norm0.001_seed300.csv
# param tunning, on dialog server
# dp
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210423/224844 --cuda cuda:1 --outputf attacks/membership_inference/dp/final_fix/lr0.1_sigma0.25_norm0.1_seed1111.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210423/224945 --cuda cuda:1 --outputf attacks/membership_inference/dp/final_fix/lr0.1_sigma0.5_norm0.1_seed1111.csv



# dialog experiments

################################################
# membership running, dialog
################################################
# dialog
python attacks/mem_inference.py  --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000 --checkpoint model/nodp/20210503/220336 --cuda cuda:0 --outputf attacks/membership_inference/dialog/nodp/final_fix/nodp_seed1111.csv
python attacks/mem_inference.py  --data_type dial --data data/simdial --path0 attacks/membership_inference/candidates/dialog/test --path1 attacks/membership_inference/candidates/dialog/train -bs 64 --N 1000 --checkpoint model/partialdp/20210503/230904 --cuda cuda:0 --outputf attacks/membership_inference/dialog/partialdp/final_fix/nodp_seed1111.csv


# canary insertion for partial dp, param search
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/111019 --cuda cuda:0 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/111038 --cuda cuda:1 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.1_norm0.25_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/111051 --cuda cuda:2 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.05_norm0.005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/111126 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.05_norm0.25_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/134414 --cuda cuda:4 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.01_norm0.005_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210423/183122 --cuda cuda:5 --outputf attacks/canary_insertion/partialdp/param_search/lr0.1_sigma0.01_norm0.25_seed1111.csv
#=============================
#membership 
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111019 --cuda cuda:0 --outputf attacks/membership_inference/partialdp/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv   --data_type doc --data data/wikitext-2-add10b
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111038 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/param_search/lr0.1_sigma0.1_norm0.25_seed1111.csv   --data_type doc --data data/wikitext-2-add10b
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111051 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/param_search/lr0.1_sigma0.05_norm0.005_seed1111.csv  --data_type doc --data data/wikitext-2-add10b
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111126 --cuda cuda:3 --outputf attacks/membership_inference/partialdp/param_search/lr0.1_sigma0.05_norm0.25_seed1111.csv  --data_type doc --data data/wikitext-2-add10b
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/134414 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/param_search/lr0.1_sigma0.01_norm0.005_seed1111.csv  --data_type doc --data data/wikitext-2-add10b
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/183122 --cuda cuda:5 --outputf attacks/membership_inference/partialdp/param_search/lr0.1_sigma0.01_norm0.25_seed1111.csv  --data_type doc --data data/wikitext-2-add10b




######################################
# missing digits, miss only 1
######################################
# missing digit, baseline, on interaction
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add1b-normalized/missing_digits --cuda cuda:3 2>&1 | tee logs/nodp/normalized/miss_1/lstm.log
# screen -R nodp2
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add1b-normalized/missing_digits --cuda cuda:3 --seed 0 2>&1 | tee logs/nodp/normalized/miss_1/lstm_seed0.log
# screen -R nodp3
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add1b-normalized/missing_digits --cuda cuda:3 --seed 123 2>&1 | tee logs/nodp/normalized/miss_1/lstm_seed123.log
# screen -R nodp4
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add1b-normalized/missing_digits --cuda cuda:3 --seed 22 2>&1 | tee logs/nodp/normalized/miss_1/lstm_seed22.log
# screen -R nodp5
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add1b-normalized/missing_digits --cuda cuda:3 --seed 300 2>&1 | tee logs/nodp/normalized/miss_1/lstm_seed300.log

# missing digit, partial dp, on interaction
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add1b --epochs 100 --seed 1111 2>&1 | tee logs/partial_dp/missed/miss_1/lr0.1_sigm0.5_norm1e-3_seed1111_miss1.log
# missing digit, partial dp, on interaction
### screen -R partialdp1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add1b --epochs 100 --seed 0 2>&1 | tee logs/partial_dp/missed/miss_1/lr0.1_sigm0.5_norm1e-3_seed0_miss1.log
### screen -R partialdp2
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add1b --epochs 100 --seed 123 2>&1 | tee logs/partial_dp/missed/miss_1/lr0.1_sigm0.5_norm1e-3_seed123_miss1.log
### screen -R partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add1b --epochs 100 --seed 22 2>&1 | tee logs/partial_dp/missed/miss_1/lr0.1_sigm0.5_norm1e-3_seed22_miss1.log
### screen -R partialdp4
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add1b --epochs 100 --seed 300 2>&1 | tee logs/partial_dp/missed/miss_1/lr0.1_sigm0.5_norm1e-3_seed300_miss1.log


# no dp normalized, miss 1
# screen -r d7
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210506/234706 --cuda cuda:0 --outputf attacks/membership_inference/nodp_normalized_miss1/nodp_seed1111.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210507/044232 --cuda cuda:0 --outputf attacks/membership_inference/nodp_normalized_miss1/nodp_seed0.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210507/094710 --cuda cuda:0 --outputf attacks/membership_inference/nodp_normalized_miss1/nodp_seed123.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210507/141548 --cuda cuda:3 --outputf attacks/membership_inference/nodp_normalized_miss1/nodp_seed22.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210507/192746 --cuda cuda:3 --outputf attacks/membership_inference/nodp_normalized_miss1/nodp_seed300.csv

python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/nodp/20210506/234706
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/nodp/20210507/044232
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/nodp/20210507/094710
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/nodp/20210507/141548
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/nodp/20210507/192746

python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210506/234706 --cuda cuda:0 --outputf attacks/canary_insertion/nodp_normalized_miss1/nodp_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210507/044232 --cuda cuda:0 --outputf attacks/canary_insertion/nodp_normalized_miss1/nodp_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210507/094710 --cuda cuda:0 --outputf attacks/canary_insertion/nodp_normalized_miss1/nodp_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210507/141548 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized_miss1/nodp_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210507/192746 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized_miss1/nodp_seed300.csv





######### missing experiments
# 1. append another secret, 
# already run on cuda:0
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b-missed-append10 --epochs 100 --seed 1111 2>&1 | tee logs/partial_dp/missed/append/lr0.1_sigm0.5_norm1e-3_seed1111_miss10.log
### already run screen -r d6
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:2 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b-missed-append10 --epochs 100 --seed 22 2>&1 | tee logs/partial_dp/missed/append/lr0.1_sigm0.5_norm1e-3_seed22_miss10.log
### screen -R partialdp2, 
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b-missed-append10 --epochs 100 --seed 123 2>&1 | tee logs/partial_dp/missed/append/lr0.1_sigm0.5_norm1e-3_seed123_miss10.log

# 2. use both unk and digits as secrets


### screen -R partialdp3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 1e-3  --sigma 0.5 -missing_digits -digits_unk_as_private --data data/wikitext-2-add10b --epochs 100 --seed 22 2>&1 | tee logs/partial_dp/missed/both_unk_digits/lr0.1_sigm0.5_norm1e-3_seed22_miss10.log
# run already on cuda:5
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:5 -partial -norm 1e-3  --sigma 0.5 -missing_digits -digits_unk_as_private --data data/wikitext-2-add10b --epochs 100 --seed 300 2>&1 | tee logs/partial_dp/missed/both_unk_digits/lr0.1_sigm0.5_norm1e-3_seed300_miss10.log
# run already on cuda: 1
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:1 -partial -norm 1e-3  --sigma 0.5 -missing_digits -digits_unk_as_private --data data/wikitext-2-add10b --epochs 100 --seed 1111 2>&1 | tee logs/partial_dp/missed/both_unk_digits/lr0.1_sigm0.5_norm1e-3_seed1111_miss10.log


### missing 20
# screen -r d3
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:4 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add20b --epochs 100 --seed 300 2>&1 | tee logs/partial_dp/missed/miss_20/lr0.1_sigm0.5_norm1e-3_seed300_miss20.log
# screen -r d5
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:6 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add20b --epochs 100 --seed 1111 2>&1 | tee logs/partial_dp/missed/miss_20/lr0.1_sigm0.5_norm1e-3_seed1111_miss20.log



# missing digit, baseline, on interaction
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-missed-append10-normalized/missing_digits --cuda cuda:3 2>&1 | tee logs/nodp/normalized-append10/20210511/lstm.log
# screen -R nodp2
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-missed-append10-normalized/missing_digits --cuda cuda:3 --seed 0 2>&1 | tee logs/nodp/normalized-append10/20210511/lstm_seed0.log
# screen -R nodp3
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-missed-append10-normalized/missing_digits --cuda cuda:3 --seed 123 2>&1 | tee logs/nodp/normalized-append10/20210511/lstm_seed123.log
# screen -R nodp4
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-missed-append10-normalized/missing_digits --cuda cuda:3 --seed 22 2>&1 | tee logs/nodp/normalized-append10/20210511/lstm_seed22.log
# screen -R nodp5
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-missed-append10-normalized/missing_digits --cuda cuda:3 --seed 300 2>&1 | tee logs/nodp/normalized-append10/20210511/lstm_seed300.log



# missing digit, baseline, on interaction
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add20b-normalized/missing_digits --cuda cuda:3 2>&1 | tee logs/nodp/normalized-missing20/20210511/lstm.log
# screen -R nodp2
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add20b-normalized/missing_digits --cuda cuda:3 --seed 0 2>&1 | tee logs/nodp/normalized-missing20/20210511/lstm_seed0.log
# screen -R nodp3
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add20b-normalized/missing_digits --cuda cuda:3 --seed 123 2>&1 | tee logs/nodp/normalized-missing20/20210511/lstm_seed123.log
# screen -R nodp4
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add20b-normalized/missing_digits --cuda cuda:3 --seed 22 2>&1 | tee logs/nodp/normalized-missing20/20210511/lstm_seed22.log
# screen -R nodp5
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add20b-normalized/missing_digits --cuda cuda:3 --seed 300 2>&1 | tee logs/nodp/normalized-missing20/20210511/lstm_seed300.log








# 
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210506/234027 --cuda cuda:3 --outputf attacks/membership_inference/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210506/234042 --cuda cuda:3 --outputf attacks/membership_inference/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210506/234049 --cuda cuda:3 --outputf attacks/membership_inference/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210506/234057 --cuda cuda:3 --outputf attacks/membership_inference/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed300.csv

python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/partialdp/20210506/234027
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/partialdp/20210506/234042
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/partialdp/20210506/234049
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:3 -model_dir model/partialdp/20210506/234057

python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210506/234027 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210506/234042 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210506/234049 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210506/234057 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed300.csv


python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:3 -model_dir model/partialdp/20210507/221838
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210507/221838 --cuda cuda:3 --outputf attacks/membership_inference/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed100.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210507/221838 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_1/lr0.1_sigma0.5_norm0.001_seed100.csv



# manual check
# both
# compare with attacks/canary_insertion/nodp_normalized/nodp_normalized_seed0.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210510/101057/data-wikitext-2-add10b_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-257.9607115_acc-0.28376_epoch-16_ep-4.343_dl-8e-05_ap-3.80 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_both_unk_digits/test/ppl258.csv

# append
# nodp
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/230530/data-missing_digits_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-16_bptt-35_lr-20.0_dp-False_partial-False_0hidden-False.pt_ppl-204.8520007_acc-0.28234_epoch-1_ep-0.000_dl-0_ap-0.00 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized_append10/test/ppl204.csv
# nodp
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210511/230530/data-missing_digits_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-16_bptt-35_lr-20.0_dp-False_partial-False_0hidden-False.pt_ppl-252.3595410_acc-0.25790_epoch-1_ep-0.000_dl-0_ap-0.00 --cuda cuda:3 --outputf attacks/canary_insertion/nodp_normalized_append10/test/ppl252.csv
# sdp
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210511/164815/data-wikitext-2-add10b-missed-append10_model-LSTM_ebd-200_hid-200_bi-False_lay-1_tie-False_tok-50258_bs-7_bptt-35_lr-0.1_dp-True_partial-True_0hidden-False_sigma-0.5_norm-0.001_dl-8e-05.pt_ppl-254.9234443_acc-0.27307_epoch-8_ep-3.624_dl-8e-05_ap-4.00 --cuda cuda:3 --outputf attacks/canary_insertion/partialdp_missed_append/test/ppl252.csv




#### for the missing experiments
# append
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/partialdp/20210511/164815/
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210511/164815/ --cuda cuda:0 --outputf attacks/membership_inference/partialdp_missed_append/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/partialdp/20210511/164815/ --cuda cuda:0 --outputf attacks/canary_insertion/partialdp_missed_append/lr0.1_sigma0.5_norm0.001_seed22.csv


python -u scripts/adjust_ppl_acc.py --cuda cuda:1 -bs 64 -model_dir model/partialdp/20210510/110828/
python attacks/mem_inference.py --cuda cuda:1 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210510/110828/ --outputf attacks/membership_inference/partialdp_missed_append/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/canary_insertion.py --cuda cuda:1 -bs 256 --checkpoint model/partialdp/20210510/110828/ --outputf attacks/canary_insertion/partialdp_missed_append/lr0.1_sigma0.5_norm0.001_seed1111.csv


# both unk and digit
python -u scripts/adjust_ppl_acc.py --cuda cuda:2 -bs 64 -model_dir model/partialdp/20210510/131908/
python attacks/mem_inference.py --cuda cuda:2 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210510/131908/ --outputf attacks/membership_inference/partialdp_missed_both_unk_digits/lr0.1_sigma0.5_norm0.001_seed300.csv
python attacks/canary_insertion.py --cuda cuda:2 -bs 256 --checkpoint model/partialdp/20210510/131908/ --outputf attacks/canary_insertion/partialdp_missed_both_unk_digits/lr0.1_sigma0.5_norm0.001_seed300.csv

python -u scripts/adjust_ppl_acc.py --cuda cuda:4 -bs 64 -model_dir model/partialdp/20210510/101057/
python attacks/mem_inference.py --cuda cuda:4 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210510/101057/ --outputf attacks/membership_inference/partialdp_missed_both_unk_digits/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/canary_insertion.py --cuda cuda:4 -bs 256 --checkpoint model/partialdp/20210510/101057/ --outputf attacks/canary_insertion/partialdp_missed_both_unk_digits/lr0.1_sigma0.5_norm0.001_seed1111.csv


# miss 20
python -u scripts/adjust_ppl_acc.py --cuda cuda:5 -bs 64 -model_dir model/partialdp/20210511/112916/
python attacks/mem_inference.py --cuda cuda:5 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210511/112916/ --outputf attacks/membership_inference/partialdp_missed_20/lr0.1_sigma0.5_norm0.001_seed300.csv
python attacks/canary_insertion.py --cuda cuda:5 -bs 256 --checkpoint model/partialdp/20210511/112916/ --outputf attacks/canary_insertion/partialdp_missed_20/lr0.1_sigma0.5_norm0.001_seed300.csv

python -u scripts/adjust_ppl_acc.py --cuda cuda:6 -bs 64 -model_dir model/partialdp/20210511/113022/
python attacks/mem_inference.py --cuda cuda:6 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210511/113022/ --outputf attacks/membership_inference/partialdp_missed_20/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/canary_insertion.py --cuda cuda:6 -bs 256 --checkpoint model/partialdp/20210511/113022/ --outputf attacks/canary_insertion/partialdp_missed_20/lr0.1_sigma0.5_norm0.001_seed1111.csv


####################################################
# no dp, normalized, baselines
# no dp, normalized, missing 20
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/nodp/20210514/002133/
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210514/002133/ --cuda cuda:0 --outputf attacks/membership_inference/nodp_normalized_miss20/nodp_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210514/002133/ --cuda cuda:0 --outputf attacks/canary_insertion/nodp_normalized_miss20/nodp_seed1111.csv



python -u scripts/adjust_ppl_acc.py --cuda cuda:1 -bs 64 -model_dir model/nodp/20210511/230530/
python attacks/mem_inference.py --cuda cuda:1 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210511/230530/ --outputf attacks/membership_inference/nodp_normalized_append10/nodp_seed0.csv
python attacks/canary_insertion.py --cuda cuda:1 -bs 256 --checkpoint model/nodp/20210511/230530/ --outputf attacks/canary_insertion/nodp_normalized_append10/nodp_seed0.csv


# both unk and digit
python -u scripts/adjust_ppl_acc.py --cuda cuda:2 -bs 64 -model_dir model/nodp/20210512/234722/
python attacks/mem_inference.py --cuda cuda:2 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210512/234722/ --outputf attacks/membership_inference/nodp_normalized_append10/nodp_seed22.csv
python attacks/canary_insertion.py --cuda cuda:2 -bs 256 --checkpoint model/nodp/20210512/234722/ --outputf attacks/canary_insertion/nodp_normalized_append10/nodp_seed22.csv

python -u scripts/adjust_ppl_acc.py --cuda cuda:4 -bs 64 -model_dir model/nodp/20210512/112006/
python attacks/mem_inference.py --cuda cuda:4 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210512/112006/ --outputf attacks/membership_inference/nodp_normalized_append10/nodp_seed123.csv
python attacks/canary_insertion.py --cuda cuda:4 -bs 256 --checkpoint model/nodp/20210512/112006/ --outputf attacks/canary_insertion/nodp_normalized_append10/nodp_seed123.csv


# miss 20
python -u scripts/adjust_ppl_acc.py --cuda cuda:5 -bs 64 -model_dir model/nodp/20210513/114701/
python attacks/mem_inference.py --cuda cuda:5 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210513/114701/ --outputf attacks/membership_inference/nodp_normalized_append10/nodp_seed300.csv
python attacks/canary_insertion.py --cuda cuda:5 -bs 256 --checkpoint model/nodp/20210513/114701/ --outputf attacks/canary_insertion/nodp_normalized_append10/nodp_seed300.csv

python -u scripts/adjust_ppl_acc.py --cuda cuda:6 -bs 64 -model_dir model/nodp/20210511/112425/
python attacks/mem_inference.py --cuda cuda:6 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210511/112425/ --outputf attacks/membership_inference/nodp_normalized_append10/nodp_seed1111.csv
python attacks/canary_insertion.py --cuda cuda:6 -bs 256 --checkpoint model/nodp/20210511/112425/ --outputf attacks/canary_insertion/nodp_normalized_append10/nodp_seed1111.csv




# not missing digit, baseline, sanitization on interaction
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/not_missing_digits --cuda cuda:0 2>&1 | tee logs/nodp/normalized/not_miss/20210515/lstm.log
# screen -R nodp2
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/not_missing_digits --cuda cuda:1 --seed 0 2>&1 | tee logs/nodp/normalized/not_miss/20210515/lstm_seed0.log
# screen -R nodp3
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/not_missing_digits --cuda cuda:2 --seed 123 2>&1 | tee logs/nodp/normalized/not_miss/20210515/lstm_seed123.log
# screen -R nodp4
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/not_missing_digits --cuda cuda:4 --seed 22 2>&1 | tee logs/nodp/normalized/not_miss/20210515/lstm_seed22.log
# screen -R nodp5
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/not_missing_digits --cuda cuda:5 --seed 300 2>&1 | tee logs/nodp/normalized/not_miss/20210515/lstm_seed300.log



python -u main.py -bs 4 --lr 2 --data data/simdial --data_type dial --cuda cuda:3  --seed 1111 --epochs 50 -save_epoch_num 1 --log-interval 50 2>&1 | tee logs/nodp/dialog/20210515/add10_lr2_bs4/dialog_bs4_lr2_seed1111.log
python -u main.py -bs 4 --lr 2 --data data/simdial --data_type dial --cuda cuda:3  --seed 0 --epochs 50 -save_epoch_num 1 --log-interval 50 2>&1 | tee logs/nodp/dialog/20210515/add10_lr2_bs4/dialog_bs4_lr2_seed0.log
python -u main.py -bs 4 --lr 2 --data data/simdial --data_type dial --cuda cuda:3  --seed 123 --epochs 50 -save_epoch_num 1 --log-interval 50 2>&1 | tee logs/nodp/dialog/20210515/add10_lr2_bs4/dialog_bs4_lr2_seed123.log
python -u main.py -bs 4 --lr 2 --data data/simdial --data_type dial --cuda cuda:3  --seed 22 --epochs 50 -save_epoch_num 1 --log-interval 50 2>&1 | tee logs/nodp/dialog/20210515/add10_lr2_bs4/dialog_bs4_lr2_seed22.log
python -u main.py -bs 4 --lr 2 --data data/simdial --data_type dial --cuda cuda:6  --seed 300 --epochs 50 -save_epoch_num 1 --log-interval 50 2>&1 | tee logs/nodp/dialog/20210515/add10_lr2_bs4/dialog_bs4_lr2_seed300.log




##### no dp, normalized, not miss
#### for the missing experiments
model/nodp/20210515/122150
model/nodp/20210515/122210
model/nodp/20210515/122226
model/nodp/20210515/122246
model/nodp/20210515/122306


# append
# screen -r nomiss0
python -u scripts/adjust_ppl_acc.py -bs 64 --cuda cuda:0 -model_dir model/nodp/20210515/122150
python attacks/mem_inference.py --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210515/122150 --cuda cuda:0 --outputf attacks/membership_inference/nodp_normalized_nomiss/nodp_seed1111.csv
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210515/122150 --cuda cuda:0 --outputf attacks/canary_insertion/nodp_normalized_nomiss/nodp_seed1111.csv

# screen -r nomiss1
python -u scripts/adjust_ppl_acc.py --cuda cuda:1 -bs 64 -model_dir model/nodp/20210515/122210
python attacks/mem_inference.py --cuda cuda:1 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210515/122210 --outputf attacks/membership_inference/nodp_normalized_nomiss/nodp_seed0.csv
python attacks/canary_insertion.py --cuda cuda:1 -bs 256 --checkpoint model/nodp/20210515/122210 --outputf attacks/canary_insertion/nodp_normalized_nomiss/nodp_seed0.csv


# screen -r nomiss2
python -u scripts/adjust_ppl_acc.py --cuda cuda:2 -bs 64 -model_dir model/nodp/20210515/122226
python attacks/mem_inference.py --cuda cuda:2 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210515/122226 --outputf attacks/membership_inference/nodp_normalized_nomiss/nodp_seed123.csv
python attacks/canary_insertion.py --cuda cuda:2 -bs 256 --checkpoint model/nodp/20210515/122226 --outputf attacks/canary_insertion/nodp_normalized_nomiss/nodp_seed123.csv

# screen -r nomiss3
python -u scripts/adjust_ppl_acc.py --cuda cuda:3 -bs 64 -model_dir model/nodp/20210515/122246
python attacks/mem_inference.py --cuda cuda:3 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210515/122246 --outputf attacks/membership_inference/nodp_normalized_nomiss/nodp_seed22.csv
python attacks/canary_insertion.py --cuda cuda:3 -bs 256 --checkpoint model/nodp/20210515/122246 --outputf attacks/canary_insertion/nodp_normalized_nomiss/nodp_seed22.csv


# screen -r nomiss4
python -u scripts/adjust_ppl_acc.py --cuda cuda:4 -bs 64 -model_dir model/nodp/20210515/122306
python attacks/mem_inference.py --cuda cuda:4 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210515/122306 --outputf attacks/membership_inference/nodp_normalized_nomiss/nodp_seed300.csv
python attacks/canary_insertion.py --cuda cuda:4 -bs 256 --checkpoint model/nodp/20210515/122306 --outputf attacks/canary_insertion/nodp_normalized_nomiss/nodp_seed300.csv

