"""
Attention Implementations for Lab 2 — Attention in Action

STUDENTS: Copy your implementations from the notebook into this file.
The auto-grading tests import from here.

Three functions to implement:
  1. scaled_dot_product_attention(Q, K, V, mask=None)
  2. create_causal_mask(seq_len)
  3. multi_head_attention(X, num_heads, d_model, mask=None)
"""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax (provided — do not modify)."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        mask: Optional boolean mask, shape (seq_len_q, seq_len_k).
              True = masked (should not attend).

    Returns:
        output: shape (seq_len_q, d_v)
        weights: shape (seq_len_q, seq_len_k) — attention probabilities
    """
    # TODO: Paste your implementation from the notebook here.

    raise NotImplementedError(
        "scaled_dot_product_attention() not yet implemented. "
        "Copy your code from the notebook."
    )


def create_causal_mask(seq_len):
    """
    Create a causal (auto-regressive) mask.
    Position i can only attend to positions 0..i.

    Args:
        seq_len: Length of the sequence.

    Returns:
        Boolean mask of shape (seq_len, seq_len).
        True = MASKED (cannot attend), False = allowed.
    """
    # TODO: Paste your implementation from the notebook here.

    raise NotImplementedError(
        "create_causal_mask() not yet implemented. "
        "Copy your code from the notebook."
    )


def multi_head_attention(X, num_heads, d_model, mask=None):
    """
    Compute multi-head attention.

    Args:
        X: Input embeddings, shape (seq_len, d_model)
        num_heads: Number of attention heads
        d_model: Embedding dimension (must be divisible by num_heads)
        mask: Optional causal mask

    Returns:
        output: shape (seq_len, d_model)
        all_weights: List of attention weight matrices (one per head)
    """
    # TODO: Paste your implementation from the notebook here.

    raise NotImplementedError(
        "multi_head_attention() not yet implemented. "
        "Copy your code from the notebook."
    )
