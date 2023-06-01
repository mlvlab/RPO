GPU=$1
EPOCH=10

for seed in 0 1 2
do
    sh scripts/rpo/xd_train.sh imagenet ${seed} ${GPU} imagenet_k24_ep15
    for dataset in imagenet imagenet_a imagenet_r imagenet_sketch imagenetv2
    do
        sh scripts/rpo/xd_test.sh ${dataset} ${seed} ${EPOCH} ${GPU}
    done
done
