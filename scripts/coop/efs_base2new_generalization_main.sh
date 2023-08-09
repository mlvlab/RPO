GPU=$1
SHOT=16

for shot in 1 2 4 8 16
do
    for dataset in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
    do
        for seed in 1 2 3 4 5 6 7 8 9 10
        do
            # training
            sh scripts/coop/base2new_train.sh ${dataset} ${seed} ${shot} ${GPU}
            # evaluation
            sh scripts/coop/base2new_test.sh ${dataset} ${seed} ${shot} base ${GPU}
            sh scripts/coop/base2new_test.sh ${dataset} ${seed} ${shot} new ${GPU}  
        done
    done
done