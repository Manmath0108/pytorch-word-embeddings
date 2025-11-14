import torch
from .config import LR, EPOCHS, DEVICE
from .model import SkipGramModel

def train(model, pairs, batch_gen):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for centers, contexts in batch_gen(pairs):
            centers = centers.to(DEVICE)
            contexts = contexts.to(DEVICE)

            loss = model(centers, contexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")