# nodp
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/nodp/20210418/181252/
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/nodp/20210418/192226/
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/nodp/20210417/144949/
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/nodp/20210419/052248/
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/nodp/20210418/210231/
# dp
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/dp/20210425/151315
# partial dp, param search, low priority
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/partialdp/20210417/000742
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/partialdp/20210417/000240
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/partialdp/20210417/000251
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/partialdp/20210417/111003
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:1 -model_dir model/partialdp/20210417/000327


#dp
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/dp/20210425/151340
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/dp/20210425/151359
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/dp/20210425/151417
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/dp/20210425/151429
# partialdp
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210418/191438
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210421/123500
# low pri
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210417/000344
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210419/063839
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210419/063840
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210419/063841
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:2 -model_dir model/partialdp/20210419/063842



# partialdp
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210421/123511
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210421/123522
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210421/123530
# partial dp resume
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210427/213802
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210427/213754
# nodp, normalized missed
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/nodp/20210426/182422
# low pri
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210419/063843
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210423/111019
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210423/111038
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210423/111057
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:4 -model_dir model/partialdp/20210423/111076



# nodp, normalized missed
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/nodp/20210426/234106
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/nodp/20210426/234125
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/nodp/20210426/234146
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/nodp/20210426/234201
# partial dp, missed
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/partialdp/20210427/211240
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/partialdp/20210427/211322
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/partialdp/20210427/211411
# low pri
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/partialdp/20210423/111095
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/partialdp/20210423/111114
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/partialdp/20210425/150805
# dp, param search, low priority
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/dp/20210423/224844
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:5 -model_dir model/dp/20210423/224945


#################################
# on dialog server, not run yet
#################################
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210423/221533
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210423/221514/
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210427/213900
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210426/223009
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210427/211339
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210425/152829
python -u scripts/adjust_ppl_acc.py -bs 256 --cuda cuda:0 -model_dir model/partialdp/20210425/152857
