GPU=$1
SHOT=16

for dataset in sun397
do
    for seed in 0
    do
        # training
        #sh scripts/coop/base2new_train.sh ${dataset} ${seed} ${SHOT} ${GPU}
        # evaluation
        #sh scripts/coop/base2new_test.sh ${dataset} ${seed} ${SHOT} base ${GPU}
        sh scripts/coop/base2new_test.sh ${dataset} ${seed} ${SHOT} new ${GPU}  
    done
done