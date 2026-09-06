import torch
import torch.nn.functional as F # Using only for checking, will remove it later

import torch

def dense_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores + mask

    weights = torch.softmax(scores, dim=-1)

    output = weights @ V

    return output, weights

torch.manual_seed(0)

#example: batch=2, heads=2, seq_len=5, d_k=8
batch, heads, seq_len, d_k = 2, 2, 5, 8
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

#No mask
my_output, my_weights = dense_attention(Q, K, V, mask=None)
ref_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None)

diff = (my_output - ref_output).abs().max().item()
print(f"Max difference (no mask): {diff:.8f}")
print("Match!" if diff < 1e-5 else "MISMATCH — something's wrong")

#With a causal mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
additive_mask = torch.zeros(seq_len, seq_len)
additive_mask.masked_fill_(causal_mask, float('-inf'))

my_output_causal, _ = dense_attention(Q, K, V, mask=additive_mask)
ref_output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

diff_causal = (my_output_causal - ref_output_causal).abs().max().item()
print(f"Max difference (causal): {diff_causal:.8f}")
print("Match!" if diff_causal < 1e-5 else "MISMATCH")