# Collaborative-Learning
Tensorflow implementation of NIPS 2018 paper "Collaborative Learning for Deep Neural Networks" [link](https://arxiv.org/pdf/1805.11761.pdf)
## Network and Dataset
We use DenseNet-40-12 and Cifar-10 dataset
## Results
I got 6.09% error rate after 300 epochs which is a slightly different from the paper. Maybe the split point is different from the paper: in my implementation splitting is done right after Batch Normalization and Relu of transition layers while it's not clear whether they split before or after or in the transition layers. Besides, in my implementation, gradients would pass through soft label targets (notation "q" in the paper).
