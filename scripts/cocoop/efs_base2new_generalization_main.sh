GPU=$1

for shot in 8
do
    for dataset in sun397
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
        do
            # training
            sh scripts/cocoop/base2new_train.sh ${dataset} ${seed} ${shot} ${GPU}
            # evaluation
            #sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${shot} base ${GPU}
            sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${shot} new ${GPU}  
        done
    done
done