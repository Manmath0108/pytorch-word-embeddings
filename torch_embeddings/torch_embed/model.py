import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)
        self.context = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, center_ids, context_ids):
        c = self.center(center_ids)
        o = self.context(context_ids)
        dot = (c*o).sum(dim=1)
        return -F.logsigmoid(dot).mean()