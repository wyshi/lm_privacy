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


# dp param search, on nlp
# screen -R dp1
python -u main.py -bs 3 --lr 0.1 -dp --cuda cuda:0 -norm 0.25 --seed 1111 2>&1 | tee logs/dp/20210423/param_search/lr0.1_sigma0.5_norm0.25_seed1111 
# screen -R dp2
python -u main.py -bs 6 --lr 0.1 -dp --cuda cuda:1 -norm 0.5  --seed 1111 2>&1 | tee logs/dp/20210423/param_search/lr0.1_sigma0.5_norm0.5_seed1111 
