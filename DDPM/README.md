# SEMU for DDPM
This is the official repository for SEMU for DDPM. The code structure of this project is adapted from the [DDIM](https://github.com/ermongroup/ddim), [SA](https://github.com/clear-nus/selective-amnesia/tree/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm), and [Unlearn-Saliency (SalUn)](https://github.com/OPTML-Group/Unlearn-Saliency) codebases.

# Requirements
Install the requirements using a `conda` environment:
```
conda create --name semu-ddpm python=3.8
conda activate semu-ddpm
pip install -r requirements.txt
```

# Forgetting Training with SEMU

1. First train a conditional DDPM on all 10 CIFAR10/STL10 classes. 

   Specify GPUs using the `CUDA_VISIBLE_DEVICES` environment flag. 

   We demonstrate the code to run SEMU on CIFAR10; the commands can run the STL10 experiments using the same commands but replacing config  and dataset flags accordingly.

   For instance, using two GPUs with IDs 0 and 1 on CIFAR10,

   ```
    CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_train.yml --mode train
   ```

   Similar to the VAE, a checkpoint should be saved under `results/cifar10/yyyy_mm_dd_hhmmss`. 

2. Next, we need to generate model with SVD projections.

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_semu_unlearn.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --label_to_forget 0 --mode generate_svd --explained_variance_ratio 0.95 --explained_variance_ratio_attention 1.0 --use_projection_grad
   ```

   This will create a new model with the set parameters in the directory of your pretrained DDPM model. You can also use other hyperparams from train.py script for generating wanted SVD projections.

3. Forgetting training (withouth remaining dataset) with SEMU

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_semu_unlearn.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --label_to_forget 0 --mode semu_unlearn --alpha 1e-3 --method rl [SVD OPTIONS; same as in step 2]
   ```

   You can experiment with forgetting different class labels using the `--label_to_forget` flag, but we will consider forgetting the 0 (airplane) class here.


4. Forgetting training (with remaining dataset) with SEMU

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_semu_unlearn_retrain.yml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss --label_to_forget 0 --mode saliency_unlearn --alpha 1e-3 --method rl [SVD OPTIONS; same as in step 2]
   ```


   You can experiment with forgetting different class labels using the `--label_to_forget` flag, but we will consider forgetting the 0 (airplane) class here.
