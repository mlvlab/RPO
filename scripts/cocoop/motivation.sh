GPU=$1
SHOT=16

for dataset in fgvc_aircraft
do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
        # training
        sh scripts/cocoop/base2new_train.sh ${dataset} ${seed} ${SHOT} ${GPU}
        # evaluation
        sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${SHOT} base ${GPU}
        sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${SHOT} new ${GPU} 
    done
done