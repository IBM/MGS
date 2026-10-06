# MNIST Deep Learning Example

This directory contains a complete implementation of a backpropagation neural network for MNIST digit classification using MGS.

## Files

- `MNIST.gsl` - Network topology specification (layers, connections)
- `MNIST_gsl.dev` - Development version with additional annotations
- `README.md` - This file

## Architecture

- Input layer: 784 neurons (28×28 MNIST images)
- Hidden layer 1: 256 neurons (ReLU)
- Hidden layer 2: 128 neurons (ReLU)
- Output layer: 10 neurons (softmax)

## Training

- Optimizer: Adam (learning rate 0.001)
- Loss: Cross-entropy
- Batch size: 64
- Epochs: 20

## Usage
```bash
# From MGS root directory
./build_mgs -p LINUX --as-MGS
bin/gslparser examples/ml/MNIST.gsl
```

## Technical Details

See the technical note: [Implementing Backpropagation as Hypergraphs](../../docs/papers/technical-notes/MNIST_backprop_hypergraph.pdf)
