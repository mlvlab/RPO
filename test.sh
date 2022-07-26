#!/bin/sh

for epoch in 50, 100, 200
do
    python evaluate.py --device mps --dataset eurosat --epoch ${epoch} --layer 11 --type text+vision_metanet --division base --kshot 16 --topk 1
done