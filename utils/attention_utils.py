"""
Attention Utilities for Lab 2 — Attention in Action

Provided helper functions (fully implemented — students use these as tools).
  - softmax(): Numerically stable softmax
  - plot_attention(): Visualize attention weight heatmap
  - plot_attention_comparison(): Side-by-side heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x, axis=-1):
    """
    Numerically stable softmax.

    Args:
        x: Input array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax probabilities (same shape as x).
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def plot_attention(weights, tokens_q, tokens_k, title="Attention Weights",
                   figsize=(8, 6), cmap="Blues", annotate=True, ax=None):
    """
    Visualize an attention weight matrix as a heatmap.

    Args:
        weights: 2D array of shape (len(tokens_q), len(tokens_k)).
        tokens_q: List of query token labels (rows).
        tokens_k: List of key token labels (columns).
        title: Chart title.
        figsize: Figure size tuple (only used if ax is None).
        cmap: Matplotlib colormap name.
        annotate: Whether to annotate cells with numeric values.
        ax: Optional matplotlib Axes to draw on. If None, creates new figure.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=weights.max())

    ax.set_xticks(range(len(tokens_k)))
    ax.set_xticklabels(tokens_k, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(tokens_q)))
    ax.set_yticklabels(tokens_q, fontsize=10)
    ax.set_xlabel("Keys (attending TO)", fontsize=11)
    ax.set_ylabel("Queries (attending FROM)", fontsize=11)
    ax.set_title(title, fontsize=13)

    if annotate:
        for i in range(len(tokens_q)):
            for j in range(len(tokens_k)):
                val = weights[i, j]
                color = "white" if val > weights.max() * 0.6 else "black"
                fontsize = 9 if len(tokens_q) <= 10 else 7
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=fontsize, color=color)

    plt.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)

    if ax.figure is not None:
        plt.tight_layout()

    return ax


def plot_attention_comparison(weights_list, tokens_list, titles,
                              figsize=None, cmap="Blues"):
    """
    Plot multiple attention heatmaps side by side.

    Args:
        weights_list: List of 2D weight arrays.
        tokens_list: List of (tokens_q, tokens_k) tuples.
        titles: List of subplot titles.
        figsize: Figure size. Auto-computed if None.
        cmap: Colormap name.

    Returns:
        Figure and axes.
    """
    n = len(weights_list)
    if figsize is None:
        figsize = (7 * n, 6)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    vmax = max(w.max() for w in weights_list)

    for ax, w, (tq, tk), title in zip(axes, weights_list, tokens_list, titles):
        im = ax.imshow(w, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(tk)))
        ax.set_xticklabels(tk, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(tq)))
        ax.set_yticklabels(tq, fontsize=9)
        ax.set_title(title, fontsize=12)

        annotate = len(tq) <= 8
        if annotate:
            for i in range(len(tq)):
                for j in range(len(tk)):
                    val = w[i, j]
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, color=color)

    plt.tight_layout()
    return fig, axes


def print_attention_distribution(weights_row, tokens, label=""):
    """
    Print a single token's attention distribution as a horizontal bar.

    Args:
        weights_row: 1D array of attention weights for one query token.
        tokens: List of key token labels.
        label: Optional label for the distribution.
    """
    if label:
        print(f"\nAttention distribution for '{label}':")
    print(f"{'Token':<14} {'Weight':>8}  Bar")
    print("-" * 45)
    for tok, w in zip(tokens, weights_row):
        bar = "█" * int(w * 40)
        print(f"{tok:<14} {w:>8.3f}  {bar}")
