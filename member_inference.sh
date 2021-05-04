################################################
# membership running, wiki
################################################
# nodp
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/181252/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300/nodp_seed1111.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/192226/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300/nodp_seed0.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210417/144949/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300/nodp_seed123.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210419/052248/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300/nodp_seed22.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/210231/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300/nodp_seed300.csv

# dp
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151315 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300/lr0.05_sigma0.5_norm0.1_seed1111.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151340 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300/lr0.05_sigma0.5_norm0.1_seed0.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151359 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300/lr0.05_sigma0.5_norm0.1_seed123.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151417 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300/lr0.05_sigma0.5_norm0.1_seed22.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151429 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300/lr0.05_sigma0.5_norm0.1_seed300.csv

# partial dp, first 50
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210418/191438 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123500 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123511 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123522 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123530 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300/lr0.1_sigma0.5_norm0.001_seed300.csv



# nodp, different candidate
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/181252/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300_diff_candi/nodp_seed1111.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/192226/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300_diff_candi/nodp_seed0.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210417/144949/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300_diff_candi/nodp_seed123.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210419/052248/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300_diff_candi/nodp_seed22.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210418/210231/ --cuda cuda:0 --outputf attacks/membership_inference/nodp/final/seed300_diff_candi/nodp_seed300.csv

# dp
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151315 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300_diff_candi/lr0.05_sigma0.5_norm0.1_seed1111.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151340 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300_diff_candi/lr0.05_sigma0.5_norm0.1_seed0.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151359 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300_diff_candi/lr0.05_sigma0.5_norm0.1_seed123.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151417 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300_diff_candi/lr0.05_sigma0.5_norm0.1_seed22.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/dp/20210425/151429 --cuda cuda:1 --outputf attacks/membership_inference/dp/final/seed300_diff_candi/lr0.05_sigma0.5_norm0.1_seed300.csv

# partial dp, first 50
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210418/191438 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300_diff_candi/lr0.1_sigma0.5_norm0.001_seed1111.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123500 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300_diff_candi/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123511 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300_diff_candi/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123522 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300_diff_candi/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py --seed 300 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210421/123530 --cuda cuda:2 --outputf attacks/membership_inference/partialdp/final/seed300_diff_candi/lr0.1_sigma0.5_norm0.001_seed300.csv








# partial dp, resume 50
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/213802 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final/resume/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/213754 --cuda cuda:4 --outputf attacks/membership_inference/partialdp/final/resume/lr0.1_sigma0.5_norm0.001_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/213900 --cuda cuda:1 --outputf attacks/membership_inference/partialdp/final/resume/lr0.1_sigma0.5_norm0.001_seed300.csv

# no dp normalized, 
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/182422 --cuda cuda:5 --outputf attacks/membership_inference/nodp_normalized/final/nodp_seed1111.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234106 --cuda cuda:5 --outputf attacks/membership_inference/nodp_normalized/final/nodp_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234125 --cuda cuda:5 --outputf attacks/membership_inference/nodp_normalized/final/nodp_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234146 --cuda cuda:5 --outputf attacks/membership_inference/nodp_normalized/final/nodp_seed22.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210426/234201 --cuda cuda:5 --outputf attacks/membership_inference/nodp_normalized/final/nodp_seed300.csv


# missed dp normalized
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211240 --cuda cuda:4 --outputf attacks/membership_inference/partialdp_missed/final/lr0.1_sigma0.5_norm0.001_seed0.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211322 --cuda cuda:4 --outputf attacks/membership_inference/partialdp_missed/final/lr0.1_sigma0.5_norm0.001_seed123.csv
python attacks/mem_inference.py  --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/partialdp/20210427/211411 --cuda cuda:4 --outputf attacks/membership_inference/partialdp_missed/final/lr0.1_sigma0.5_norm0.001_seed300.csv




# orange ones, param search
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000742 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.45_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000240 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.45_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000251 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/111003 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000327 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.55_norm0.01_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210417/000344 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.55_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/063839 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm0.0005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134334 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm0.0001_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/064057 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm5e-05_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134345 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm1e-05_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210419/134357 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.5_norm5e-06_seed1111.csv
# more param tunning
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111019 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.1_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111038 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.1_norm0.25_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111051 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.05_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/111126 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.05_norm0.25_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/134414 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.01_norm0.005_seed1111.csv
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210423/183122 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/lr0.1_sigma0.01_norm0.25_seed1111.csv

# more param tunning, resume
python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint model/partialdp/20210425/150805 --cuda cuda:6 --outputf attacks/membership_inference/partialdp/final/param_search/resume/lr0.1_sigma0.1_norm0.005_seed1111.csv
