from .config import set_seed, DEVICE, EMBED_DIM, WINDOW_SIZE, BATCH_SIZE
from .vocab import build_vocab
from .dataset import generate_skipgram_pairs, make_batches
from .model import SkipGramModel
from .train import train
from .utils import save_embeddings
from pathlib import Path

def read_corpus(path = Path(__file__).resolve().parent / "data" / "corpus.txt"):
    with open(path, "r") as f:
        text = f.read().lower().split()
    return text

if __name__ == "__main__":
    set_seed(42)

    tokens = read_corpus()
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = [word_to_id.get(w.lower(), word_to_id.get('<UNK>', 0)) for w in tokens]

    pairs = generate_skipgram_pairs(token_ids, WINDOW_SIZE)

    model = SkipGramModel(vocab_size=len(word_to_id), embed_dim=EMBED_DIM).to(DEVICE)

    train(model, pairs, lambda p: make_batches(p, BATCH_SIZE))

    save_embeddings(model, id_to_word)