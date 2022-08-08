#!/bin/bash

for data in eurosat caltech101 flowers102 ucf101 dtd food101
do   
    python evaluate.py --device cuda:5 --dataset ${data} --epoch 200 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2021
    python evaluate.py --device cuda:5 --dataset ${data} --epoch 200 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2021
    python evaluate.py --device cuda:5 --dataset ${data} --epoch 200 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2022
    python evaluate.py --device cuda:5 --dataset ${data} --epoch 200 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2022
    python evaluate.py --device cuda:5 --dataset ${data} --epoch 200 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2023
    python evaluate.py --device cuda:5 --dataset ${data} --epoch 200 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2023
done