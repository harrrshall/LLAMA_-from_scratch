# LLAMA_from_scratch
For the sake of learning, I'm implementing LLAMA from the ground up in my own way

Note:-  The current code is in a just clean, simple and minimalistic implementation of [llama](https://github.com/facebookresearch/llama/blob/main/llama/model.py)



# Llama Language Model Documentation

## sd

This code implements a language model using a modified Transformer architecture called "Llama." Llama incorporates three key modifications to the original Transformer model:

1. **RMSNorm for pre-normalization**
2. **Rotary embeddings**
3. **SwiGLU activation function**

The model is trained on a text dataset using PyTorch. The training process includes tokenization, batching, and optimization.

## Tokenization

The input text is read from the 'input.txt' file, and a vocabulary is created based on unique characters in the text. Tokenization is performed by encoding characters into numerical indices using dictionaries (`itos` and `stoi`).

## Llama Model Implementation
This code implements a scaled-down version of the Llama model architecture described in the paper "Llama: Efficient Sparse and Context-Aware Models" (https://arxiv.org/abs/2102.05055). It trains on the TinyShakespeare dataset of all of Shakespeare's works tokenized at the character level.

The key classes and functions are:

Dataset
get_batches() - Returns batches of inputs and targets from the dataset for training.
Modules
RMSNorm() - Implements RMS normalization layer.
RoPEAttentionHead() - Self-attention head with Rotary positional embeddings.
RoPEMaskedAttentionHead() - Adds causal masking to self-attention.
RoPEMultiheadAttention() - Multi-headed self-attention with RoPE.
SwiGLU() - SwiGLU activation function layer.
LlamaBlock() - Single transformer block with RMSNorm and attention/feedforward.
Llama() - Full Llama model with embeddings, blocks, and output projection.
Training
evaluate_loss() - Evaluates validation loss during training.
train() - Performs model training with logging.
generate() - Generates text from the trained model.
The model is implemented incrementally, starting from a basic RNN and adding components from Llama one by one: RMSNorm, RoPE, causal masking, multi-head attention, SwiGLU, and stacked blocks. Training curves are plotted to check for improvements at each step.

Key techniques used include:

Incremental paper implementation with iterative testing and debugging
Ensuring dimensionality matches at each layer
Checking attention masks and gradients flow properly
Tracking validation loss to evaluate model quality
