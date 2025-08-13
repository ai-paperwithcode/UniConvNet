# Training

We provide ImageNet-1K training here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Multi-node Training
We use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit) for producing the results and models in the paper. Please install:
```
pip install submitit
```
We will give example commands for both multi-node and single-machine training below.

## ImageNet-1K Training 
UniConvNet-A (3.4M) training on ImageNet-1K with 4 8-GPU nodes:
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model UniConvNet_A --drop_path 0.05 \
--batch_size 256 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results

```

- You may need to change cluster-specific arguments in `run_with_submitit.py`.
- You can add `--use_amp true` to train in PyTorch's Automatic Mixed Precision (AMP).
- Use `--resume /path_or_url/to/checkpoint.pth` to resume training from a previous checkpoint; use `--auto_resume true` to auto-resume from latest checkpoint in the specified output folder.
- `--batch_size`: batch size per GPU; `--update_freq`: gradient accumulation steps.
- The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `2*8*256*1 = 4096`. You can adjust these four arguments together to keep the effective batch size at 4096 and avoid OOM issues, based on the model size, number of nodes and GPU memory.

You can use the following command to run this experiment on a single machine: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_A --drop_path 0.05 \
--batch_size 256 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*256*2 = 4096`. Running on one machine, we increased `update_freq` so that the total batch size is unchanged.

To train other UniConvNet variants, `--model` and `--drop_path` need to be changed. Examples are given below, each with both multi-node and single-machine commands:


<details>
<summary>
UniConvNet-P0 (5.2M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model UniConvNet_P0  --mixup 0. --cutmix 0. --reprob 0. --drop_path 0.05 \
--batch_size 256 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_P0  --mixup 0. --cutmix 0. --reprob 0. --drop_path 0.05 \
--batch_size 256 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-P1 (6.1M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_P1  --mixup 0. --cutmix 0. --reprob 0. --drop_path 0.05 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_P1  --mixup 0. --cutmix 0. --reprob 0. --drop_path 0.05 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-P2 (7.6M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_P2  --mixup 0. --cutmix 0. --reprob 0. --drop_path 0.08 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_P2  --mixup 0. --cutmix 0. --reprob 0. --drop_path 0.08 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-N0 (10.2M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_N0 --drop_path 0.08 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_N0 --drop_path 0.08 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-N1 (13.1M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_N1 --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_N1 --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-N2 (15.0M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_N2 --drop_path 0.1 \
--batch_size 64 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_N2 --drop_path 0.1 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-N3 (19.7M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_N3 --drop_path 0.1 \
--batch_size 64 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_N3 --drop_path 0.1 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-T (30.3M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_T --drop_path 0.2 \
--batch_size 64 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_T --drop_path 0.2 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-S (50.0M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_S --drop_path 0.4 \
--batch_size 32 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_S --drop_path 0.4 \
--batch_size 32 --lr 4e-3 --update_freq 16 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-B (97.6M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_B --drop_path 0.6 \
--batch_size 32 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_B --drop_path 0.6 \
--batch_size 32 --lr 4e-3 --update_freq 16 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```


</details>

## ImageNet-22K Pre-training
ImageNet-22K is significantly larger than ImageNet-1K in terms of data size, for example, we can use 16 8-GPU nodes for pre-training on ImageNet-22K.

UniConvNet-L (201.8M) pre-training on ImageNet-22K:

Multi-node
```
python run_with_submitit.py --nodes 16 --ngpus 8 \
--model UniConvNet_L --drop_path 0.2 \
--batch_size 32 --lr 4e-3 --update_freq 1 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model UniConvNet_L --drop_path 0.2 \
--batch_size 32 --lr 4e-3 --update_freq 64 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--output_dir /path/to/save_results
```

<details>
<summary>
UniConvNet-XL (226.7M)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 16 --ngpus 8 \
--model UniConvNet_XL --drop_path 0.2 \
--batch_size 32 --lr 4e-3 --update_freq 1 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model UniConvNet_XL --drop_path 0.2 \
--batch_size 32 --lr 4e-3 --update_freq 64 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--output_dir /path/to/save_results
```


</details>


## ImageNet-1K Fine-tuning
### Finetune from ImageNet-1K pre-training 
The training commands given above for ImageNet-1K use the default resolution (224). We also fine-tune these trained models with a larger resolution (384). Please specify the path or url to the checkpoint in `--finetune`.

UniConvNet-T (30.3M) fine-tuning on ImageNet-1K (384x384):

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_T --drop_path 0.4 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_T --drop_path 0.4 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 8 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

<details>
<summary>
UniConvNet-S (50.0M) (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_S --drop_path 0.6 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_S --drop_path 0.6 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 8 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>
<details>
<summary>
UniConvNet-B (97.6M) (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model UniConvNet_B --drop_path 0.8 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_B --drop_path 0.8 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 8 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

</details>

### Fine-tune from ImageNet-22K pre-training
We finetune from ImageNet-22K pre-trained models, in 384 resolutions.


<details>
<summary>
UniConvNet-L (201.8M) (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model UniConvNet_L --drop_path 0.3 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_L --drop_path 0.3 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 8 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

</details>
<details>
<summary>
UniConvNet-XL (226.7M) (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model UniConvNet_XL --drop_path 0.35 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_XL --drop_path 0.35 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 8 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

