# [UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale](https://arxiv.org/abs/2508.09000)

Official PyTorch implementation of **UniConvNet**, from the following paper:

[UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale](https://arxiv.org/abs/2508.09000). \
ICCV 2025.\
Yuhao Wang, Wei Xi \
Xi'an Jiaotong University\
[[`arXiv`](https://arxiv.org/abs/2508.09000)]

---

<div align="center">
  <img src="UniConvNet.png" width=100% height=100% />
</div>


We propose **UniConvNet**, a pure ConvNet model constructed entirely from standard ConvNet modules. UniConvNet performs well on both lightweight and large-scale models.

## Catalog
- [x] ImageNet-1K Training Code  
- [x] ImageNet-22K Pre-training Code  
- [x] ImageNet-1K Fine-tuning Code  
- [x] Downstream Transfer (Detection, Segmentation) Code (Coming soon ...)


<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### ImageNet-1K trained models

|     name      | resolution | acc@1 | #params | FLOPs  |                                                   model(hugging face)                                                    |                           model(baidu)                           |
|:-------------:|:----------:|:-----:|:-------:|:------:|:------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| UniConvNet-A  |  224x224   | 77.0  |  3.4M   | 0.589G |             [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_a_1k_224.pth)             |[model](https://pan.baidu.com/s/1vHK2bRb82dTHOrvn_oc0Cg?pwd=bqw6) |
| UniConvNet-P0 |  224x224   | 79.1  |  5.2M   | 0.932G |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_p0_1k_224_ema.pth)            |[model](https://pan.baidu.com/s/1kzp_qmFniG78pbYPNF26kQ?pwd=u85r) |
| UniConvNet-P1 |  224x224   | 79.6  |  6.1M   | 0.895G |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_p1_1k_224_ema.pth)            |[model](https://pan.baidu.com/s/1iERGRTI3eSs4lxodCjK9oQ?pwd=qyvg) |
| UniConvNet-P2 |  224x224   | 80.5  |  7.6M   | 1.25G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_p2_1k_224_ema.pth)            |[model](https://pan.baidu.com/s/1q2C27CIf2cqCV_mVjXQb0Q?pwd=p82i) |
| UniConvNet-N0 |  224x224   | 81.6  |  10.2M  | 1.65G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_n0_1k_224_ema.pth)            |[model](https://pan.baidu.com/s/1uwqqQpMcbfUOQgUAlz3h7g?pwd=5zu9) |
| UniConvNet-N1 |  224x224   | 82.2  |  13.1M  | 1.88G  |              [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_n1_1k_224.pth)              |[model](https://pan.baidu.com/s/1xL-ZRnlivt16O3F3k7-61Q?pwd=jmdw) |
| UniConvNet-N2 |  224x224   | 82.7  |  15.0M  | 2.47G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_n2_1k_224_ema.pth)            |[model](https://pan.baidu.com/s/1182n2NWa2bSHXOy70jEc0g?pwd=rxf5) |
| UniConvNet-N3 |  224x224   | 83.2  |  19.7M  | 3.37G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_n3_1k_224_ema.pth)            |[model](https://pan.baidu.com/s/1ApJGDVFTCZdy-DdZw_7V3w?pwd=9r9a) |
| UniConvNet-T  |  224x224   | 84.2  |  30.3M  |  5.1G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_t_1k_224_ema.pth)             |[model](https://pan.baidu.com/s/1ws1Qyjg0kwH2qaSxjztyQg?pwd=ekun) |
| UniConvNet-T  |  384x384   | 85.4  |  30.3M  | 15.0G  |              [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_t_1k_384.pth)               |[model](https://pan.baidu.com/s/19-dK5wum4koFsjxKtzQ7dA?pwd=xg8n) |
| UniConvNet-S  |  224x224   | 84.5  |  50.0M  | 8.48G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_s_1k_224_ema.pth)             |[model](https://pan.baidu.com/s/10UzasYF7TEgbQLRu-tJb1Q?pwd=145f) |
| UniConvNet-S  |  384x384   | 85.7  |  50.0M  | 24.9G  |              [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_s_1k_384.pth)               |[model](https://pan.baidu.com/s/1XLBReKeC_vXQyiwJl9n7tA?pwd=r3em) |
| UniConvNet-B  |  224x224   | 85.0  |  97.6M  | 15.9G  |            [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_b_1k_224_ema.pth)             |[model](https://pan.baidu.com/s/1pj00TQDlLlcAtdAitaDDbA?pwd=d4va) |
| UniConvNet-B  |  384x384   | 85.9  |  97.6M  | 46.6G  |              [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_b_1k_384.pth)               |[model](https://pan.baidu.com/s/15LeK5sxM4_unDkUgYiaEng?pwd=bgdb) |


### ImageNet-22K trained models

| name | resolution | acc@1 | #params | FLOPs  |                                  22k model <br/>(hugging face)                                  |                                  22k model (baidu)                                  |1k model <br/>(hugging face) |22k model (baidu)|
|:---:|:---:|:-----:|:-------:|:------:|:-----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|:----------------------------:|:-----------------------------:|
| ConvNeXt-L | 384x384 | 88.2  | 201.8M  | 100.1G | [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_l_22k_224.pth)  |   [model](https://pan.baidu.com/s/1fRQthCJJZscuLRPjVZNlBw?pwd=g5je)    |[model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_l_22k_1k_384.pth)|[model](https://pan.baidu.com/s/1EgEZRzjSl9Zeqm_EGMN5tA?pwd=emdz)|
| ConvNeXt-XL | 384x384 | 88.4  | 226.7M  | 115.2G | [model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_xl_22k_224.pth) | [model](https://pan.baidu.com/s/10Kz-4quNFhHmpMSdvtV4_g?pwd=bwku) |[model](https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_xl_22k_1k_384.pth)|[model](https://pan.baidu.com/s/1tOn8cuNqle-k6V0aOa153A?pwd=kha8)|




## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for a ImageNet-1K pre-trained UniConvNet-L:

Single-GPU
```
python main.py --model UniConvNet_A --eval true \
--resume https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_a_1k_224.pth \
--input_size 224 --drop_path 0.05 \
--data_path /path/to/imagenet-1k
```


Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model UniConvNet_A --eval true \
--resume https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_a_1k_224.pth \
--input_size 224 --drop_path 0.05 \
--data_path /path/to/imagenet-1k
```

This should give 
```
* Acc@1 77.030 Acc@5 93.364 loss 0.983
```

- For evaluating other model variants, change `--model`, `--resume`, `--input_size` accordingly. You can get the url to pre-trained models from the tables above. 
- Setting model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in timm behaves the same during evaluation; but it is required in training. See [TRAINING.md](TRAINING.md) or our paper for the values used for different models.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [InternImage](https://github.com/OpenGVLab/InternImage) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{wang2025uniconvnet,
  author  = {Yuhao Wang and Wei Xi},
  title   = {UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale},
  journal = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year    = {2025},
}
```
