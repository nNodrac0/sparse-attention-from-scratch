import torch
import torch.nn.functional as F # Using only for checking, will remove it later

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
# print(f"Max difference (no mask): {diff:.8f}")
# print("Match!" if diff < 1e-5 else "MISMATCH — something's wrong")

#With a causal mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
additive_mask = torch.zeros(seq_len, seq_len)
additive_mask.masked_fill_(causal_mask, float('-inf'))

my_output_causal, _ = dense_attention(Q, K, V, mask=additive_mask)
ref_output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

diff_causal = (my_output_causal - ref_output_causal).abs().max().item()
# print(f"Max difference (causal): {diff_causal:.8f}")
# print("Match!" if diff_causal < 1e-5 else "MISMATCH")


# Item 1.2

def sliding_window_mask(seq_len, window_size, causal=True):
    # i, j will be (seq_len, seq_len) grids of rows and columns
    i = torch.arange(seq_len).unsqueeze(1)  # shape (seq_len, 1)
    j = torch.arange(seq_len).unsqueeze(0)  # shape (1, seq_len)

    distance = i - j

    if causal:
        allowed = (distance >= 0) & (distance <= window_size)
    else:
        allowed = distance.abs() <= window_size

    mask = torch.zeros(seq_len, seq_len)
    mask.masked_fill_(~allowed, float('-inf'))
    return mask



mask = sliding_window_mask(seq_len=8, window_size=2, causal=True)

# Print it as a readable grid: "." = allowed, "X" = forbidden
for row in mask:
    print("".join("." if val == 0 else "X" for val in row))
