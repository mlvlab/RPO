# Read-only-Prompt-Optimization

This repository contains the codebase of RPO (Read-only Prompt Optimization), which can effectively adapt large-scale vision-language models (e.g. CLIP) to downstream image recongition tasks 
through prompt learning in generalizable, robust, and parameter efficient manner.

## About the paper
* [Read-only Prompt Optimization for Vision-Language Few-shot Learning (2023)](https://drive.google.com/file/d/1TMADl2ZqSqzimeHUR7mv37ZthJBiO-oA/view?usp=sharing)

![image](https://github.com/dongdori/Read-only-Prompt-Optimization/assets/70640345/917f6a3b-1925-4128-8a94-a3f2c138649d)

As shown in the figure above, We empirically observed that CLIP adaptation based on linear probe shows more stable performance with respect to 10 random few-shot training data sampling, compared to prompt learning approaches (e.g. CoOp and CoCoOp).

Although it was counterintuitive, We conjectured that preventing internal feature space of pretrained model from being affected by newly introduced prompts can be beneficial in terms of generalizability and robustness after adaptation.

![image](https://github.com/dongdori/Read-only-Prompt-Optimization/assets/70640345/45d75ea2-441d-4243-8494-84e4b2118c42)

Therefore, we propose **Read-only Attention Mechanism** to prevent internal distribution shift occured by learnable prompts.
The **Read-only Attention Mechanism** can be simply implemented by attention masks for visual encoder and text encoder.

The read only prompts read and transform essential information and context from the pre-trained feature space, at the same time do not affect to internal representation of pretrained model.

* RPO shows better base-to-new generalization and domain generalization performance compared to previous baselines.
* RPO acheives better novel accuracy compared to zero-shot CLIP in base-to-new generalization
* Performance of RPO shows smaller variance with respect to randomly sampled few-shot training data, compared to previous baselines. (Robust!)
* RPO requires fewer parameters to train and lighter computational cost for inference, compared to previous baselines.

If you are interested in more details of RPO, Please read our paper!

## Installations & Downloads
1. install Dassl library following instruction from this [link](https://github.com/KaiyangZhou/Dassl.pytorch#installation)
2. After activating dassl and install clip following instruction from this [link](https://github.com/openai/CLIP)
3. Follow [DATASET.md](https://github.com/dongdori/Read-only-Prompt-Optimization/blob/main/DATASETS.md) to download datasets.

## How to Run?

### Main Results
#### Base to new generalization (16shot) on entire 11 datasets

```
scripts/rpo/base2new_generalization_main.sh [gpu_id]

# for individual dataset training & evaluation

## training

sh scripts/rpo/base2new_train.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT}

## evaluation
sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K24 ${SHOT} ${EPOCH} base
sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} new
```

#### Domain generalization (16shot) on 5 benchmarks (ImageNet-Sketch, A, R, V2)
```
scripts/rpo/domain_generalization_main.sh [gpu_id]

# for individual dataset training & evaluation

## training 
sh scripts/rpo/xd_train.sh imagenet ${seed} ${GPU} imagenet_k24_ep15
        
## evaluation (on ImageNetV2, ImageNet-R, ImageNet-A, ImageNet-Sketch)
sh scripts/rpo/xd_test.sh ${dataset} ${seed} ${EPOCH} ${GPU}
```

### Analyses

#### Extreme Few-shot settings (1, 2, 4, 8 shot training & evaluation)
```
scripts/rpo/efs_base2new_generalization_main.sh [gpu_id]
```


