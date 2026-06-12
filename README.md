# Collaborative-Learning

A TensorFlow/[Tensorpack](https://github.com/tensorpack/tensorpack) implementation of [Collaborative Learning for Deep Neural Networks (NeurIPS 2018)](https://papers.nips.cc/paper/2018/hash/430c3626b879b4005d41b8a46172e0c0-Abstract.html) by Song and Chai, reproducing the DenseNet-40-12 / CIFAR-10 experiment.

Collaborative learning trains **multiple classifier heads of the same network simultaneously on the same data**: each head learns from the ground-truth labels *and* from the consensus (soft labels) of its peers, while the heads share low-level layers. This improves generalization with **no extra inference cost** — at test time only one head is kept.

## Implementation

Everything lives in `main.py`:

- **Backbone** — DenseNet-40-12 (3 dense blocks, growth rate 12, dropout 0.2), with PyTorch-style initialization and standard CIFAR-10 preprocessing/augmentation (pad-and-crop, horizontal flip).
- **Hierarchical head splitting** (`recursive_split_block`) — after the first dense block, the network is split recursively (2-way, depth 2 → 4 heads in a binary tree), so heads share intermediate-level representations (ILR) at different depths.
- **Backpropagation rescaling** (`scale_grad_layer`) — at every split point the gradient flowing back into the shared layers is divided by the number of children, implementing the paper's gradient rescaling so the shared layers receive a properly scaled aggregate gradient.
- **Collaborative loss** — for each head, the objective is `β · hard + (1 − β) · soft` with β = 0.5: a standard cross-entropy against the ground truth, plus a distillation-style term against the soft target `q` = softmax of the *other* heads' averaged logits at temperature T = 2. The targets `q` are wrapped in `stop_gradient`, as in the paper.
- **Training schedule** — SGD with momentum 0.9, weight decay 1e-4, 300 epochs, learning rate 0.1 → 0.01 → 0.001 → 0.0001 at epochs 150/225/290. Classification error is monitored on head 0 only (the head you would keep for deployment).

## Run

Requires TensorFlow 1.x (GPU) and Tensorpack. CIFAR-10 is downloaded automatically by Tensorpack.

```bash
python main.py --gpu 0
```

Logs and checkpoints go to `train_log/`; resume with `--load path/to/checkpoint`.

## Results

**6.09% test error** after 300 epochs — close to, but slightly off, the paper's reported numbers. A likely cause is the split-point ambiguity: here the network splits right after the BatchNorm + ReLU of the transition layers, while the paper does not specify whether the split happens before, after, or inside the transition layers.

## Citation

```bibtex
@inproceedings{song2018collaborative,
  title={Collaborative Learning for Deep Neural Networks},
  author={Song, Guocong and Chai, Wei},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  pages={1837--1846},
  year={2018}
}
```
