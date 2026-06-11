# Collaborative-Learning

<!-- README refined by Cursor -->

Tensorflow implementation of "Collaborative Learning for Deep Neural Networks"

## Overview

This repository contains Python code from an older research, course, or prototype project. The README has been refreshed to make the repository easier to scan while preserving the original notes below.

## Repository Contents

- Top-level source files and project assets.

## Setup

- This legacy repo does not pin a full environment. Start from the language/toolchain implied by the source files, then install missing packages as reported by the runtime.

## Usage

- `python main.py`

## Data and Artifacts

No new large artifact is stored in this repository. If a dataset or checkpoint is required, follow the links and notes in the original section below.

## Status

This is a `Batch B` cleanup pass for a legacy repository. Commands may require dependency/version adjustments on a modern machine.

## License

No explicit license file was found in this checkout; check the original project context before reusing code.

## Original Notes

# Collaborative-Learning
Tensorflow implementation of NIPS 2018 paper "Collaborative Learning for Deep Neural Networks" [link](https://arxiv.org/pdf/1805.11761.pdf)
## Network and Dataset
We use DenseNet-40-12 and Cifar-10 dataset
## Results
I got 6.09% error rate after 300 epochs which is a slightly different from the paper. Maybe the split point is different from the paper: in my implementation splitting is done right after Batch Normalization and Relu of transition layers while it's not clear whether they split before or after or in the transition layers. Besides, in my implementation, gradients would pass through soft label targets (notation "q" in the paper).
