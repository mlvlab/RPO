GPU=$1
SHOT=16

for dataset in dtd fgvc_aircraft oxford_pets caltech101 stanford_cars oxford_flowers food101 ucf101 sun397 imagenet
do
    for seed in 0 1 2
    do
        # training
        sh scripts/lp/base2new_train.sh ${dataset} ${seed} ${SHOT} ${GPU}
        # evaluation
        # sh scripts/lp/base2new_test.sh ${dataset} ${seed} ${SHOT} base ${GPU}
        sh scripts/lp/base2new_test.sh ${dataset} ${seed} ${SHOT} new ${GPU}
    done
done