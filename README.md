# misgan


python gain/train_gain.py --data-file data/uce_train_mul3.csv --save-path ./model/gain_uce_mul3.ckpt





for i in 1 2 3 4 5 6 7 8 9 10 ; do python -m gain.train_gain --data-file data/tpc_h_train_mul3.csv --save-path ./model/gain_tpc_h_mul3_${i}.ckpt; done;


for i in 01 02 03 05 12 14 15 24 23 45 ; do python -m gain.train_gain --data-file data_new/data_gamma_1_3_plus_3col_part${i}.csv --save-path ./model_new/gain_gamma_1_3_plus_3col_part${i}.ckpt; done;
for i in 01 02 03 05 12 14 15 24 23 45 ; do python -m gain.train_gain --data-file data_new/data_gamma_2_3_plus_3col_part${i}.csv --save-path ./model_new/gain_gamma_2_3_plus_3col_part${i}.ckpt; done;
for i in 01 02 03 05 12 14 15 24 23 45 ; do python -m gain.train_gain --data-file data_new/gauss_plus_3col_part${i}.csv --save-path ./model_new/gain_gauss_plus_3col_part${i}.ckpt; done;
for i in 01 02 03 05 12 14 15 24 23 45 ; do python -m gain.train_gain --data-file data_new/uniform_plus_3col_part${i}.csv --save-path ./model_new/gain_uniform_plus_3col_part${i}.ckpt; done;



for i in 01 02 03 05 12 14 15 24 23 45 ; do python -m gain.train_gain_random_prior --data-file data_new/data_gamma_1_3_plus_3col_part${i}.csv --save-path ./model_new/gain_random_prior_gamma_1_3_plus_3col_part${i}.ckpt; done;
for i in 01 02 03 05 12 14 15 24 23 45 ; do python -m gain.train_gain_random_prior --data-file data_new/data_gamma_2_3_plus_3col_part${i}.csv --save-path ./model_new/gain_random_prior_gamma_2_3_plus_3col_part${i}.ckpt; done;

