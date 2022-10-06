#!/bin/bash

for data in eurosat caltech101 ucf101 flowers102 food101 dtd
do
    python main.py --device cuda:7 --dataset ${data} --type visualcocoopv3 --kshot 16 --start_epoch 0 --division base --seed 2021 --train_textprompt n --regularize_vprompt n
    python main.py --device cuda:7 --dataset ${data} --type visualcocoopv3 --kshot 16 --start_epoch 0 --division base --seed 2022 --train_textprompt n --regularize_vprompt n
    python main.py --device cuda:7 --dataset ${data} --type visualcocoopv3 --kshot 16 --start_epoch 0 --division base --seed 2023 --train_textprompt n --regularize_vprompt n
done 