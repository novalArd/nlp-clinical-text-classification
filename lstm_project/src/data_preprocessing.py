import re
from collections import Counter

MAX_LEN = 40

def tokenize(text):
    """Tokenisasi sederhana berbasis regex untuk teks klinis."""
    return re.findall(r'[a-zA-Z]+', str(text).lower())

def build_vocab(texts, min_freq=1):
    """Membangun vocabulary dari korpus teks training."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=MAX_LEN):
    """Mengubah teks menjadi sekuens ID token dengan padding/truncation."""
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    
    # Truncate jika melebihi max_len
    ids = ids[:max_len]
    # Padding jika kurang dari max_len
    ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids
