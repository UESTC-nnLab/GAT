
# Geometry Attention Transformer with Position-aware LSTMs for Image Captioning
This is the official repo for the paper `Geometry Attention Transformer with Position-aware LSTMs for Image Captioning`.

## Requirements

1. Install the dependancies in `requirements.txt`
2. Install java for evaluation, e.g. CIDEr, BLEU.

## Train
1. `train_GAT3.sh` is for training our GAT model.
2. `train_transformer.sh` is for training the vanilla transformer model.

## Evaluation
The evaluation codes are from [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption) and [Consensus-based Image Description Evaluation](https://github.com/vrama91/cider).