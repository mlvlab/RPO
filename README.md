# Read-only-Prompt-Optimization

This repository contains the codebase of RPO (Read-only Prompt Optimization), which can effectively adapt large-scale vision-language models (e.g. CLIP) to downstream image recongition tasks 
through prompt learning in generalizable, robust, and parameter efficient manner.

## About the paper

## Installations & Downloads
1. install Dassl library following instruction from this [link](https://github.com/KaiyangZhou/Dassl.pytorch#installation)
2. After activating dassl and install clip following instruction from this [link](https://github.com/openai/CLIP)
3. Follow [DATASET.md](https://github.com/dongdori/Read-only-Prompt-Optimization/blob/main/DATASETS.md) to download datasets.

## How to Run?

### Main Results
#### Table 1. Base to new generalization 

```
# Linear Probe
sh scripts/lp/base2new_generalization_main.sh [gpu_id]
# RPO
sh scripts/rpo/base2new_generalization_main.sh [gpu_id]
```

#### Table 2. Domain generalization
```
sh scripts/rpo/domain_generalization_main.sh [gpu_id]
```

### Analyes & Figures

#### Figure 1.
```
# CoOp
sh scripts/coop/motivation.sh [gpu_id]

# CoCoOp
sh scripts/cocoop/motivation.sh [gpu_id]

# Linear Probe
sh scripts/lp/motivation.sh [gpu_id]

```



#### Table 4 & Figure 5
```
# RPO
sh scripts/rpo/efs_base2new_generalization_main.sh [gpu_id]

# CoCoOp
sh scripts/cocoop/efs_base2new_generalization_main.sh [gpu_id]
```

