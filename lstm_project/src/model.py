import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        # Layer 1: Embedding - mengubah token ID ke dense vector
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # token <PAD> tidak ikut training
        )
        # Layer 2: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # membaca kiri->kanan dan kanan->kiri
        )
        # Layer 3: Dropout regularisasi
        self.dropout = nn.Dropout(dropout)
        # Layer 4: Fully connected classifier
        # hidden_size * 2 karena bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        out, (hn, _) = self.lstm(emb)     # hn: (2, batch, hidden_size)
        # Gabungkan hidden state forward dan backward
        hn = torch.cat([hn[0], hn[1]], dim=1)  # (batch, hidden*2)
        hn = self.dropout(hn)
        return self.fc(hn)                # (batch, num_classes)

import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, embed_dim)
        
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
                 num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, dropout=dropout)
        # Stack of Transformer Encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Buat padding mask agar token <PAD> diabaikan attention
        pad_mask = (x == 0)
        emb = self.embedding(x)      # (B, seq_len, embed_dim)
        emb = self.pos_enc(emb)      # tambahkan positional encoding
        enc = self.encoder(emb, src_key_padding_mask=pad_mask)  # (B, seq, embed)
        # Mean pooling (abaikan token padding)
        mask = (~pad_mask).unsqueeze(-1).float()  # (B, seq, 1)
        pooled = (enc * mask).sum(1) / mask.sum(1)  # (B, embed_dim)
        return self.fc(pooled)       # (B, num_classes)
