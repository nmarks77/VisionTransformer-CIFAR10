# Vision Transformer for CIFAR-10 Classification

This repository contains a PyTorch implementation of a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset. The Vision Transformer model was proposed in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

This implementation is based on the tutorial found in this video: [Deep Learning (Vision Transformer from Scratch in Python)](https://www.youtube.com/watch?v=ovB0ddFtzzA). Special thanks to the video's creator for providing clear and detailed instructions on how to build the Vision Transformer from scratch.


# Vision Transformer for CIFAR-10 Classification

This repository contains a PyTorch implementation of a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset. The Vision Transformer model was proposed in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

## Model Architecture

The Vision Transformer model consists of several main components:

- Patch Embedding: The input image is split into fixed-size patches, which are linearly transformed to obtain a sequence of patch embeddings.
- Transformer Blocks: Each block contains a multi-head self-attention mechanism and a multi-layer perceptron (MLP). The sequence of patch embeddings is passed through multiple transformer blocks.
- Classification Head: The transformed sequence of patch embeddings is used to compute class probabilities.

The model is trained using cross-entropy loss, AdamW optimizer, and a learning rate scheduler.

## Dataset

This implementation uses the CIFAR-10 dataset, which is a widely used dataset for image classification. The dataset is automatically downloaded and preprocessed. Data augmentation (random crop and flip) can be used for the training set.

## Usage

The main code for training the model is in the file `train.py`. The model is trained for 20 epochs. The training loss, validation loss, and accuracy on the test set are printed after each epoch.

You can run the code using the command:

```bash
python train.py



Requirements

This code is implemented using PyTorch. You need to have PyTorch installed to run the code. Additionally, the code uses the torchvision package to download and preprocess the CIFAR-10 dataset.

