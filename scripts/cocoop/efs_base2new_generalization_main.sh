GPU=$1

for shot in 1 2 4 8 16
do
    for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101 caltech101
    do
        for seed in 1 2 3 4 5 6 7 8 9 10
        do
            # training
            sh scripts/cocoop/base2new_train.sh ${dataset} ${seed} ${shot} ${GPU}
            # evaluation
            #sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${shot} base ${GPU}
            sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${shot} new ${GPU}  
        done
    done
done