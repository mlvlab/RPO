GPU=$1
SHOT=16
EPOCH=15

for dataset in imagenet eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101 caltech101
do
    for seed in 0 1 2
    do
        for cfg in main_K24
        do
        # training
            sh scripts/rpo/base2new_train.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT}
        # evaluation
        #sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K24 ${SHOT} ${EPOCH} all
            sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} new
        done
    done
done