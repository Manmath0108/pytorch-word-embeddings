import numpy as np
from pathlib import Path
import math

MODELS_DIR = Path("torch_embeddings/torch_embed/models")

def load():
    center = np.load(MODELS_DIR / "center_embeddings.npy")
    mapping = np.load(MODELS_DIR / "word_to_ix.npy", allow_pickle=True).item()
    # build inverse
    ix_to_word = {i: w for w, i in mapping.items()}
    return center, mapping, ix_to_word

def cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def nearest(center, ix_to_word, mapping, word, topn=5):
    word = word.lower()
    if word not in mapping:
        print(f"'{word}' not in vocab")
        return []
    idx = mapping[word]
    vec = center[idx]
    sims = []
    for j in range(center.shape[0]):
        sims.append((j, cosine(vec, center[j])))
    sims = sorted(sims, key=lambda x: -x[1])[1:topn+1]  # skip itself
    return [(ix_to_word[i], round(score, 4)) for i, score in sims]

if __name__ == "_main_":
    center, mapping, ix_to_word = load()
    print("center shape:", center.shape)
    print("vocab size:", len(mapping))
    # show first 5 words & first 5 dims
    for w, i in list(mapping.items())[:5]:
        print(f"{w:12s} id={i} vec[:5]={center[i][:5].tolist()}")
    # test nearest for a few words (change as per your vocab)
    tests = ["king", "queen", "nlp", "dog", "cat"]
    for t in tests:
        print("\nNearest to:", t)
        print(nearest(center, ix_to_word, mapping, t, topn=5))