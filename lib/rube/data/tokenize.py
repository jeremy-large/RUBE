import collections
import numpy as np


def index_words(words, vocab_size):
    """
    :param words: list of words, the 'stream of consciousness'
    :param vocab_size: number of words that we want to encode as integers.
    We'll encode the most frequent words. We'll encode them with their frequency ranking.
    :return: (An array of integers < vocab_size. Each is the ranking of the word that was at that place in 'words',
             A dictionary mapping the original words to their counts in the data.)
    """
    counts = collections.OrderedDict({'UNK': 0}) # assume UNK the most frequent
    counts.update(collections.Counter(words).most_common(vocab_size - 1))
    data = list()
    _index = {word: i for i, word in enumerate(counts)}

    for word in words:
        data.append(_index.get(word, 0))
        if word not in _index:
            counts['UNK'] += 1

    return np.array(data), counts
