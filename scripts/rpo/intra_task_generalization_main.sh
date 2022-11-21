GPU=$1
SHOT=16
EPOCH=15

for dataset in imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
    for seed in 0 1 2
    do

        # training
        sh scripts/rpo/base2new_train.sh ${dataset} ${seed} ${GPU} main_K16 ${SHOT}

        # evaluation
        sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K16 SHOT ${EPOCH} base
        sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K16 SHOT ${EPOCH} new

    done
done