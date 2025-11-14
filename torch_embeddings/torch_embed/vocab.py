from collections import Counter
from typing import List, Tuple, Union

def build_vocab(tokens: Union[List[str], List[List[str]]], min_freq: int = 1) -> Tuple[dict, dict]:
    """Build a robust vocabulary mapping.

    - tokens can be a flat list of strings (['i','love','nlp',...])
      or a list of token lists ([['i','love'], ['nlp','is']])
    - lowercases tokens
    - includes an <UNK> token with id 0
    - filters by min_freq
    Returns: (word_to_id, id_to_word)
    """
    counter = Counter()

    if len(tokens) == 0:
        word_to_id = {"<UNK>": 0}
        id_to_word = {0: "<UNK>"}
        return word_to_id, id_to_word

    if isinstance(tokens[0], list) or isinstance(tokens[0], tuple):
        for sent in tokens:
            for w in sent:
                counter[w.lower()] += 1
    else:
        for w in tokens:
            counter[w.lower()] += 1

    word_to_id = {"<UNK>": 0}

    for word, freq in counter.items():
        if freq >= min_freq and word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    id_to_word = {i: w for w, i in word_to_id.items()}
    return word_to_id, id_to_word
