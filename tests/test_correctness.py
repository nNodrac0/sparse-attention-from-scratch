import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import dense_attention, sliding_window_mask, block_sparse_mask


def brute_force_masked_attention(Q, K, V, mask):
    seq_len = Q.shape[-2]
    d_k = Q.shape[-1]
    output = torch.zeros_like(V)

    for i in range(seq_len):
        allowed_j = (mask[i] == 0).nonzero(as_tuple=True)[0]

        q_i = Q[..., i:i+1, :]                
        k_allowed = K[..., allowed_j, :]      
        v_allowed = V[..., allowed_j, :]

        scores = (q_i @ k_allowed.transpose(-2, -1)) / (d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out_i = weights @ v_allowed           
        output[..., i, :] = out_i.squeeze(-2)

    return output

def check_pattern(name, Q, K, V, mask, tol=1e-5):
    fast_out, _ = dense_attention(Q, K, V, mask=mask)
    reference_out = brute_force_masked_attention(Q, K, V, mask)

    diff = (fast_out - reference_out).abs().max().item()
    passed = diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"{name} max diff = {diff:.8f} -> {status}")
    return passed

def main():
    torch.manual_seed(0)
    batch, heads, seq_len, d_k = 2, 2, 16, 8
    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_k)

    results = []

    sw_mask = sliding_window_mask(seq_len, window_size=3, causal=True)
    results.append(check_pattern("Sliding window", Q, K, V, sw_mask))

    bs_mask = block_sparse_mask(seq_len, window_size=2, num_global=2, num_random=2, causal=True, seed=42)
    results.append(check_pattern("Block-sparse (BigBird)", Q, K, V, bs_mask))

    print()
    if all(results):
        print("All correctness checks PASSED")
    else:
        print("Some correctness checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()