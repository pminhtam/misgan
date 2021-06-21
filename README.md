# misgan

```
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do CUDA_VISIBLE_DEVICES=1 python -m gain.train_gain_notD --save-path model/fd_plus_notD_${i}.ckpt --data-file data/fd_train_plus.csv --p-miss 0.5 --iter 3000; done
```