for data in eurosat caltech101 ucf101 flowers102 food101 dtd
do  
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division base --kshot 16 --topk 1 --seed 2021 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division novel --kshot 16 --topk 1 --seed 2021 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division base --kshot 16 --topk 1 --seed 2022 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division novel --kshot 16 --topk 1 --seed 2022 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division base --kshot 16 --topk 1 --seed 2023 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division novel --kshot 16 --topk 1 --seed 2023 --train_textprompt n --regularize_vprompt y

    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division base --kshot 16 --topk 1 --seed 2021 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division novel --kshot 16 --topk 1 --seed 2021 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division base --kshot 16 --topk 1 --seed 2022 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division novel --kshot 16 --topk 1 --seed 2022 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division base --kshot 16 --topk 1 --seed 2023 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv1 --division novel --kshot 16 --topk 1 --seed 2023 --train_textprompt y --regularize_vprompt y
    
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2021 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2021 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2022 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2022 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2023 --train_textprompt n --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2023 --train_textprompt n --regularize_vprompt y
    
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2021 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2021 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2022 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2022 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division base --kshot 16 --topk 1 --seed 2023 --train_textprompt y --regularize_vprompt y
    python evaluate.py --device cuda:7 --dataset ${data} --epoch 100 --type visualcocoopv2 --division novel --kshot 16 --topk 1 --seed 2023 --train_textprompt y --regularize_vprompt y
done 