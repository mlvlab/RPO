#!/bin/bash

for data in eurosat caltech101 ucf101 flowers102 food101 dtd 
do
    python main.py --device cuda:5 --dataset ${data} --type visualcocoopv2 --kshot 16 --start_epoch 0 --division base --seed 2021
    python main.py --device cuda:5 --dataset ${data} --type visualcocoopv2 --kshot 16 --start_epoch 0 --division base --seed 2022
    python main.py --device cuda:5 --dataset ${data} --type visualcocoopv2 --kshot 16 --start_epoch 0 --division base --seed 2023
done