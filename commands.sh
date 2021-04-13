python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial -partial_hidden_zero --sigma 0.35 2>&1 | tee logs/partial_dp/20210411/1201/hiddenzero_lr0.1_sigma0.35
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 --sigma 0.5 -norm 0.02 2>&1 | tee logs/dp/20210411/1511/lr0.1_sigma0.5_norm0.02

python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:3 -partial --sigma 0.5 -norm 0.02 --epochs 1 2>&1 | tee logs/partial_dp/20210413/1147/lr0.1_sigma0.5_norm0.02_epoch1