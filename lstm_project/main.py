import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import build_vocab, encode
from model import LSTMClassifier, TransformerClassifier
from train import train_model
from evaluate import evaluate_model
import time

def main():

    from sklearn.preprocessing import LabelEncoder
    
    import os
    
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "data", "dataset_hasil_feature_extraction.xlsx")
    
    df = pd.read_excel(dataset_path)
    
    # Asumsi kolom fitur bernama 'Combined_Text' dan kolom target bernama 'Label'
    X = df['Combined_Text'].fillna('').astype(str).tolist()
    y_raw = df['Label'].astype(str).tolist()
    
    # Encode label string menjadi integer (0, 1, 2, ...)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"Kelas Label: {label_encoder.classes_}")
    
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Tokenisasi & Vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(X_train)
    print(f"Ukuran Vocabulary: {len(vocab)} token unik")

    # 3. Encoding Sekuens
    print("Encoding sequences...")
    X_train_enc = torch.tensor([encode(t, vocab) for t in X_train], dtype=torch.long)
    X_test_enc  = torch.tensor([encode(t, vocab) for t in X_test],  dtype=torch.long)
    y_train_t   = torch.tensor(y_train_enc, dtype=torch.long)
    y_test_t    = torch.tensor(y_test_enc,  dtype=torch.long)

    print(f"Shape X_train: {X_train_enc.shape}")
    print(f"Shape X_test : {X_test_enc.shape}")

    # 4. Inisialisasi Model
    VOCAB_SIZE  = len(vocab)
    EMBED_DIM   = 64
    HIDDEN_SIZE = 64
    NUM_CLASSES = len(label_encoder.classes_)
    
    print("Initializing LSTM model...")
    model_lstm = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES)
    
    print("Initializing Transformer model...")
    model_transformer = TransformerClassifier(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=4,
        ff_dim=128, num_layers=2, num_classes=NUM_CLASSES
    )

    # 5. Training
    print("\nStarting Training LSTM...")
    torch.manual_seed(42)
    hist_lstm, time_lstm = train_model(
        model_lstm, X_train_enc, y_train_t, X_test_enc, y_test_t,
        lr=1e-3, epochs=25, batch_size=16
    )
    print(f"Training LSTM selesai dalam {time_lstm:.2f} detik")

    print("\nStarting Training Transformer...")
    torch.manual_seed(42)
    hist_transformer, time_transformer = train_model(
        model_transformer, X_train_enc, y_train_t, X_test_enc, y_test_t,
        lr=5e-4, epochs=25, batch_size=16
    )
    print(f"Training Transformer selesai dalam {time_transformer:.2f} detik")

    # 6. Evaluasi
    print("\nEvaluating Models...")
    acc_l, prec_l, rec_l, f1_l = evaluate_model(model_lstm, X_test_enc, y_test_t)
    print(f"LSTM       : Acc={acc_l:.4f}  P={prec_l:.4f}  R={rec_l:.4f}  F1={f1_l:.4f}")

    acc_t, prec_t, rec_t, f1_t = evaluate_model(model_transformer, X_test_enc, y_test_t)
    print(f"Transformer: Acc={acc_t:.4f}  P={prec_t:.4f}  R={rec_t:.4f}  F1={f1_t:.4f}")

if __name__ == "__main__":
    main()
