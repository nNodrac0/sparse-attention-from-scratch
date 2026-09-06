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




# Item 1.2

#Patter 1 
def sliding_window_mask(seq_len, window_size, causal=True):
    i = torch.arange(seq_len).unsqueeze(1) 
    j = torch.arange(seq_len).unsqueeze(0) 

    distance = i - j

    if causal:
        allowed = (distance >= 0) & (distance <= window_size)
    else:
        allowed = distance.abs() <= window_size

    mask = torch.zeros(seq_len, seq_len)
    mask.masked_fill_(~allowed, float('-inf'))
    return mask




# Pattern 2: Big Bird
def block_sparse_mask(seq_len, window_size, num_global, num_random, causal=True, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    distance = i - j

    #local window
    if causal:
        local = (distance >= 0) & (distance <= window_size)
    else:
        local = distance.abs() <= window_size

    #global tokens
    is_global = torch.zeros(seq_len, dtype=torch.bool)
    is_global[:num_global] = True
    global_allowed = is_global.unsqueeze(1) | is_global.unsqueeze(0)


    #random connections
    random_allowed = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if num_random > 0:
        for row in range(seq_len):
            # If causal, only sample from valid preceding positions
            valid_pool = torch.arange(row + 1) if causal else torch.arange(seq_len)
            k = min(num_random, len(valid_pool))
            if k > 0:
                picked = valid_pool[torch.randperm(len(valid_pool))[:k]]
                random_allowed[row, picked] = True



    allowed = local | global_allowed | random_allowed

    if causal:
        causal_allowed = distance >= 0
        allowed = allowed & causal_allowed

    mask = torch.zeros(seq_len, seq_len)
    mask.masked_fill_(~allowed, float('-inf'))
    return mask






# for testing


def show_mask(mask):
    for row in mask:
        print("".join("." if val == 0 else "X" for val in row))
    print()


torch.manual_seed(0)

sw = sliding_window_mask(8, 2)
show_mask(sw)

bs = block_sparse_mask(8, 1, 1, 1, seed=42)
show_mask(bs)

x = torch.randn(1, 2, 8, 16)
out, _ = dense_attention(x, x, x, mask=bs)
print("Output shape:", out.shape)