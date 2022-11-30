# Meta-ticket: Finding optimal subnetworks for few-shot learning within randomly initialized neural networks

Official repository for Meta-ticket paper. ([ArXiv](https://arxiv.org/abs/2205.15619), [OpenReview](https://openreview.net/forum?id=Cr4_3ptitj))

Presented at NeurIPS'22 (Poster): https://neurips.cc/virtual/2022/poster/53373

## Requirements

- Python 3.8
- NVIDIA CUDA 11.3
- torch==1.9.0
- learn2learn==0.1.6
- torchmeta==1.8.0
- pyyaml==5.3.1

## Usage
```
python3.8 main.py <command> <exp_name>
```

- `<command>` is one of `meta_train`, `meta_test`, `meta_cross_test` and `parallel`. The `meta_train`, `meta_test` and `meta_cross_test` commands can be used to meta-train/meta-test/cross-test a single model, and `parallel` can be used to reproduce our experiments with multiple seeds or to search hyperparameters.
- `<exp_name>` is one of the keys defined in the config file `config.yaml`.

## Datasets

We can download datasets for meta-learning by the scripts in `scripts` directory.

Example:

```
python3.8 scripts/download_miniimagenet.py
```

## Reproduce Experimental Results

For each experiment in our paper, we have the corresponding setting in `config.yaml`.
For example, we can run the meta-training for MetaTicket with ResNet12 on miniImageNet (5-shot, 5-way) by `parallel` command:

```
python main.py parallel config.yaml miniimagenet_5s5w_resnet12_ticket
```

In the end of experiments, we can check the final results by:

```
python utils/test_info.py __outputs__/miniimagenet_5s5w_resnet12_ticket/
```

