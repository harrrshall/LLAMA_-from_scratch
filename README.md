# LLAMA_-from_scratch
For the sake of learning, I'm implementing LLAMA from the ground up in my own way

Note:-  The current code is in a raw and uncleaned state. At the end of the day (EOD), it will be modified and simplified.


# Llama Language Model Documentation

## Introduction

This code implements a language model using a modified Transformer architecture called "Llama." Llama incorporates three key modifications to the original Transformer model:

1. **RMSNorm for pre-normalization**
2. **Rotary embeddings**
3. **SwiGLU activation function**

The model is trained on a text dataset using PyTorch. The training process includes tokenization, batching, and optimization.

## Tokenization

The input text is read from the 'input.txt' file, and a vocabulary is created based on unique characters in the text. Tokenization is performed by encoding characters into numerical indices using dictionaries (`itos` and `stoi`).

## Model Architecture

### SimpleBrokenModel

This is a baseline model with a simple architecture:

- Embedding layer
- Linear layer with ReLU activation
- Output linear layer

### SimpleModel

A refined version of `SimpleBrokenModel` with an adjustable embedding dimension (`d_model`).

### RoPE (Rotary Positional Embeddings) Models

The following components are added to the model:

- `RMSNorm` layer for pre-normalization
- Rotary positional embeddings in the attention heads
- SwiGLU activation function

### RoPEAttentionHead

A single attention head with rotary positional embeddings.

### RoPEMultiheadAttention

Multiple attention heads combined with linear layers and dropout.

### RopeModel

A model combining RMSNorm, RoPEMultiheadAttention, linear layers, and SwiGLU activation.

### SwiGLU

A custom activation function called SwiGLU, incorporating a gated linear unit.

### LlamaBlock

A block within the Llama model, consisting of RMSNorm, RoPEMaskedMultiheadAttention, and feedforward layers.

### Llama

The main Llama model, consisting of embeddings, multiple LlamaBlocks, and a final feedforward layer.

## Training

The training process involves iterating through epochs, computing losses, and optimizing model parameters using the Adam optimizer. The `train` function handles this process.

## Generation

The `generate` function generates new text using the trained model.

## Usage

1. Load and preprocess the data.
2. Choose a model architecture (e.g., `SimpleModel`, `RopeModel`, `Llama`).
3. Initialize the model with specified configurations.
4. Train the model using the `train` function.
5. Generate new text with the `generate` function.

Example usage is provided at the end of the code, demonstrating the training and generation steps for the Llama model.

For more details on Llama and its components, refer to the code comments and associated research documentation.



