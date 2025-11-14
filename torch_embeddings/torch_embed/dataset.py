import torch

def generate_skipgram_pairs(token_ids, window_size):
    pairs = []
    for i, center in enumerate(token_ids):
        for j in range(i-window_size, i+window_size+1):
            if j == i or j < 0 or j >= len(token_ids):
                continue
            pairs.append((center, token_ids[j]))
    return pairs

def make_batches(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i+batch_size]
        centers = torch.tensor([c for c,_ in batch], dtype=torch.long)
        contexts = torch.tensor([o for o,_ in batch], dtype=torch.long)
        yield centers, contexts