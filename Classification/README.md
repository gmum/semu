# SEMU for Classification
This is the official repository for SEMU for Clasification. The code structure of this project is adapted from the [Unlearn-Saliency (SalUn)](https://github.com/OPTML-Group/Unlearn-Saliency) codebase.


## Requirements
```bash
pip install -r requirements.txt
```

## Scripts
1. Get the origin model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {epochs for training} --lr {learning rate for training} --save_dir {file to save the orgin model}
    ```

    A simple example for ResNet-18 on CIFAR-10.
    ```bash
    python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182
    ```

2. SEMU
    ```bash
    python main_forget.py --save_dir {save_dir} --arch {model name} --model_path {path to pretrained model} --dataset {dataset name} --unlearn {unlearn_method} --num_indexes_to_replace {forgetting data amount} --unlearn_epochs {unlearn_epochs} --batch_size {batch size} --use_projection_grad {if use projection of gradient} --unlearn_lr {unlearning lr} --explained_variance_ratio {explained variance} --early_exit {if use early exit} --early_exit_patience {early exit patience} --early_exit_min_delta {early exit delta}
    ```
   
    A simple example for SEMU on ResNet-18 on CIFAR-10.
   ```bash
    python main_forget.py --save_dir {save_dir} --arch resnet18 --model_path {path to pretrained model} --dataset cifar100 --unlearn own_SVD --num_indexes_to_replace 4500 --unlearn_epochs 10 --batch_size 256 --use_projection_grad --unlearn_lr 1.e-5 --explained_variance_ratio 0.95 --early_exit --early_exit_patience 2 --early_exit_min_delta 0.01
    ```

3. SalUn and Other Baselines
   
*  SalUn
  ```bash
   python generate_mask.py --save_dir ${saliency_map_path} --model_path ${origin_model_path} --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs 1
   ```
   ```bash
   python main_random.py --unlearn RL --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --num_indexes_to_replace ${forgetting data amount} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
   ```

    * Retrain
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn retrain --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * FT
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```
   
    * GA
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn GA --num_indexes_to_replace 4500 --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * IU
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn wfisher --num_indexes_to_replace ${forgetting data amount} --alpha ${alpha}
    ```

    * l1-sparse
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT_prune --num_indexes_to_replace ${forgetting data amount} --alpha ${alpha} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```