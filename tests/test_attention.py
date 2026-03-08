"""
Auto-grading tests for Lab 2 — Attention Implementations

Tests verify that students correctly implemented:
  1. scaled_dot_product_attention(Q, K, V, mask=None)
  2. create_causal_mask(seq_len)
  3. multi_head_attention(X, num_heads, d_model, mask=None)

Students must copy their implementations to utils/attention_impl.py.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try to import student implementations
SDPA_AVAILABLE = False
MASK_AVAILABLE = False
MHA_AVAILABLE = False

try:
    from utils.attention_impl import scaled_dot_product_attention
    try:
        Q = np.random.randn(2, 3)
        K = np.random.randn(2, 3)
        V = np.random.randn(2, 3)
        scaled_dot_product_attention(Q, K, V)
        SDPA_AVAILABLE = True
    except NotImplementedError:
        pass
except ImportError:
    pass

try:
    from utils.attention_impl import create_causal_mask
    try:
        create_causal_mask(3)
        MASK_AVAILABLE = True
    except NotImplementedError:
        pass
except ImportError:
    pass

try:
    from utils.attention_impl import multi_head_attention
    try:
        X = np.random.randn(3, 4)
        multi_head_attention(X, num_heads=2, d_model=4)
        MHA_AVAILABLE = True
    except NotImplementedError:
        pass
except ImportError:
    pass


# ============================================================
# Tests: scaled_dot_product_attention
# ============================================================


@pytest.mark.skipif(not SDPA_AVAILABLE, reason="scaled_dot_product_attention not implemented")
class TestScaledDotProductAttention:

    def test_output_shape_square(self):
        """Q, K, V all same seq_len should produce correct shapes."""
        Q = np.random.randn(4, 3)
        K = np.random.randn(4, 3)
        V = np.random.randn(4, 5)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (4, 5), f"Output shape {output.shape}, expected (4, 5)"
        assert weights.shape == (4, 4), f"Weights shape {weights.shape}, expected (4, 4)"

    def test_output_shape_rectangular(self):
        """Different seq_len for Q vs K/V."""
        Q = np.random.randn(3, 4)
        K = np.random.randn(5, 4)
        V = np.random.randn(5, 6)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (3, 6), f"Output shape {output.shape}, expected (3, 6)"
        assert weights.shape == (3, 5), f"Weights shape {weights.shape}, expected (3, 5)"

    def test_weights_sum_to_one(self):
        """Each row of attention weights must sum to 1."""
        np.random.seed(42)
        Q = np.random.randn(5, 4)
        K = np.random.randn(5, 4)
        V = np.random.randn(5, 4)
        _, weights = scaled_dot_product_attention(Q, K, V)
        row_sums = weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Attention weight rows must sum to 1, got {row_sums}"

    def test_weights_non_negative(self):
        """All attention weights must be >= 0."""
        Q = np.random.randn(3, 4)
        K = np.random.randn(3, 4)
        V = np.random.randn(3, 4)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert np.all(weights >= -1e-9), "Attention weights must be non-negative"

    def test_scaling_applied(self):
        """Verify that scaling by sqrt(d_k) is applied (not overly peaked)."""
        np.random.seed(42)
        d_k = 64
        Q = np.random.randn(4, d_k)
        K = np.random.randn(4, d_k)
        V = np.random.randn(4, d_k)
        _, weights = scaled_dot_product_attention(Q, K, V)
        # Entropy should be reasonable (not completely peaked)
        entropy = -np.sum(weights * np.log(weights + 1e-10), axis=1)
        assert np.mean(entropy) > 0.1, \
            f"Attention too peaked — likely missing sqrt(d_k) scaling. Mean entropy: {np.mean(entropy):.4f}"

    def test_mask_zeros_future(self):
        """Masked positions should have ~0 attention weight."""
        np.random.seed(123)
        Q = np.random.randn(4, 3)
        K = np.random.randn(4, 3)
        V = np.random.randn(4, 3)
        mask = np.array([
            [False, True,  True,  True],
            [False, False, True,  True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        for i in range(4):
            for j in range(i + 1, 4):
                assert weights[i, j] < 1e-4, \
                    f"Masked position ({i},{j}) should be ~0, got {weights[i, j]:.6f}"

    def test_mask_preserves_row_sum(self):
        """Even with mask, each row should still sum to 1."""
        Q = np.random.randn(4, 3)
        K = np.random.randn(4, 3)
        V = np.random.randn(4, 3)
        mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        row_sums = weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Masked attention rows must still sum to 1, got {row_sums}"

    def test_no_mask_returns_valid(self):
        """Without mask, should still work correctly."""
        Q = np.random.randn(3, 4)
        K = np.random.randn(3, 4)
        V = np.random.randn(3, 4)
        output, weights = scaled_dot_product_attention(Q, K, V, mask=None)
        assert output is not None
        assert weights is not None


# ============================================================
# Tests: create_causal_mask
# ============================================================


@pytest.mark.skipif(not MASK_AVAILABLE, reason="create_causal_mask not implemented")
class TestCausalMask:

    def test_shape(self):
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5), f"Shape {mask.shape}, expected (5, 5)"

    def test_dtype_bool(self):
        mask = create_causal_mask(4)
        assert mask.dtype == bool, f"Dtype {mask.dtype}, expected bool"

    def test_diagonal_is_false(self):
        """Token i should be able to attend to itself."""
        mask = create_causal_mask(6)
        for i in range(6):
            assert mask[i, i] == False, \
                f"Diagonal ({i},{i}) should be False (can attend to self)"

    def test_lower_triangle_false(self):
        """Token i should be able to attend to tokens 0..i-1."""
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i):
                assert mask[i, j] == False, \
                    f"Lower triangle ({i},{j}) should be False (can attend to past)"

    def test_upper_triangle_true(self):
        """Token i should NOT be able to attend to tokens i+1..n."""
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j] == True, \
                    f"Upper triangle ({i},{j}) should be True (cannot attend to future)"

    def test_size_1(self):
        """Single token: no masking needed."""
        mask = create_causal_mask(1)
        assert mask.shape == (1, 1)
        assert mask[0, 0] == False


# ============================================================
# Tests: multi_head_attention
# ============================================================


@pytest.mark.skipif(not MHA_AVAILABLE, reason="multi_head_attention not implemented")
class TestMultiHeadAttention:

    def test_output_shape(self):
        """Output should have same shape as input."""
        X = np.random.randn(6, 8)
        output, _ = multi_head_attention(X, num_heads=2, d_model=8)
        assert output.shape == (6, 8), f"Output shape {output.shape}, expected (6, 8)"

    def test_correct_number_of_heads(self):
        """Should return one weight matrix per head."""
        X = np.random.randn(4, 8)
        _, weights = multi_head_attention(X, num_heads=4, d_model=8)
        assert len(weights) == 4, f"Expected 4 weight matrices, got {len(weights)}"

    def test_head_weights_shape(self):
        """Each head's weight matrix should be (seq_len, seq_len)."""
        X = np.random.randn(5, 6)
        _, weights = multi_head_attention(X, num_heads=3, d_model=6)
        for h, w in enumerate(weights):
            assert w.shape == (5, 5), \
                f"Head {h} weights shape {w.shape}, expected (5, 5)"

    def test_head_weights_valid_distributions(self):
        """Each head's attention weights should be valid probability distributions."""
        X = np.random.randn(4, 8)
        _, weights = multi_head_attention(X, num_heads=2, d_model=8)
        for h, w in enumerate(weights):
            row_sums = w.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-5), \
                f"Head {h} weights must sum to 1, got {row_sums}"
            assert np.all(w >= -1e-9), \
                f"Head {h} has negative weights"

    def test_heads_differ(self):
        """Different heads should produce different attention patterns."""
        np.random.seed(42)
        X = np.random.randn(6, 8)
        _, weights = multi_head_attention(X, num_heads=2, d_model=8)
        diff = np.abs(weights[0] - weights[1]).mean()
        assert diff > 0.01, \
            f"Heads produce identical patterns (mean diff={diff:.6f}). Are you using different W matrices?"

    def test_with_mask(self):
        """Should work with causal mask if create_causal_mask is available."""
        if not MASK_AVAILABLE:
            pytest.skip("create_causal_mask not available")
        X = np.random.randn(4, 8)
        mask = create_causal_mask(4)
        output, weights = multi_head_attention(X, num_heads=2, d_model=8, mask=mask)
        assert output.shape == (4, 8)
        # Check upper triangle is ~0 for each head
        for h, w in enumerate(weights):
            for i in range(4):
                for j in range(i + 1, 4):
                    assert w[i, j] < 1e-4, \
                        f"Head {h}: masked position ({i},{j}) should be ~0, got {w[i, j]:.6f}"
