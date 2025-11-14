
import numpy as np
from pathlib import Path

def set_seed(seed=42):
    import random, torch
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_embeddings(model, id_to_word, out_dir="torch_embeddings/torch_embed/models", text_fname="embeddings.txt"):
    """
    Save embeddings safely:
     - creates out_dir if missing
     - saves center embeddings as .npy and a simple text file
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # center embeddings (PyTorch tensor -> numpy)
    emb = model.center.weight.detach().cpu().numpy()
    npy_path = out / "center_embeddings.npy"
    np.save(npy_path, emb)

    # context embeddings too (optional)
    try:
        ctx = model.context.weight.detach().cpu().numpy()
        np.save(out / "context_embeddings.npy", ctx)
    except Exception:
        pass

    # save human readable text file
    txt_path = out / text_fname
    with txt_path.open("w", encoding="utf8") as f:
        for idx, word in id_to_word.items():
            vec = emb[idx]
            vec_str = " ".join(map(str, vec.tolist()))
            f.write(f"{word} {vec_str}\n")

    # also persist the mapping for easy loading later
    mapping_path = out / "word_to_ix.npy"
    import numpy as _np
    _np.save(mapping_path, id_to_word)

    print(f"Saved embeddings: {npy_path}, {txt_path}, mapping: {mapping_path}")
