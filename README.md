# Dokumentasi Implementasi Deep Learning (LSTM & Transformer)

Repositori ini berisi implementasi dari arsitektur Deep Learning **Bidirectional LSTM** dan **Transformer Encoder** yang dirancang untuk tugas klasifikasi teks rekam medis klinis, khususnya untuk memprediksi kode ICD-10 (J13, J15.0–J15.8) dari data berbahasa Indonesia. Implementasi model diekstraksi dari **Laporan Deep Learning** dan dimodularisasi ke dalam beberapa berkas terpisah untuk mempermudah pengembangan dan eksperimen. Hal ini sekaligus memenuhi kriteria penilaian untuk menerapkan minimal satu arsitektur RNN/LSTM dan satu arsitektur Transformer pada studi kasus yang sama.

## Struktur Direktori dan File

Proyek ini telah disusun ke dalam struktur *folder* sebagai berikut:

```text
lstm_project/
│
├── main.py
│
└── src/
    ├── data_preprocessing.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

### 1. `main.py`
**Fungsi:** Merupakan *entry point* atau skrip utama yang menghubungkan keseluruhan tahapan secara ujung-ke-ujung (End-to-End).
**Deskripsi:** 
- Berperan sebagai konduktor yang memuat dataset sesungguhnya (`dataset_hasil_feature_extraction.xlsx`).
- Menjalankan preprocessing teks, membagi dataset menjadi set *training* dan *testing* (`train_test_split`).
- Melakukan *encoding* label (menggunakan `LabelEncoder` dari scikit-learn).
- Menginisialisasi kedua arsitektur model (LSTM dan Transformer) dengan hiperparameter yang disesuaikan.
- Menjalankan fungsi training secara bergantian untuk masing-masing model, lalu mengevaluasi dan membandingkan performanya (Akurasi, Presisi, Recall, F1).

### 2. `src/data_preprocessing.py`
**Fungsi:** Menangani seluruh tahap pra-pemrosesan teks dari *raw text* hingga menjadi representasi tensor PyTorch.
**Deskripsi:**
- **`tokenize(text)`**: Melakukan pemotongan kalimat menjadi kata (token) sederhana memanfaatkan *regex* untuk menyaring hanya karakter alfanumerik (membuang tanda baca dan mengubah ke *lowercase*).
- **`build_vocab(texts)`**: Membangun kamus vocabulary berisi mapping kata ke ID *integer*, dengan menambahkan dua token spesial, yaitu `<PAD>` (untuk penyeragaman panjang) dan `<UNK>` (untuk token di luar vocabulary *training*).
- **`encode(text, vocab, max_len)`**: Mengonversi teks utuh menjadi *sequence of IDs*. Termasuk mengaplikasikan strategi *padding* (menambahkan ID `0` jika teks kependekan) dan *truncation* (memotong dari kanan jika teks terlalu panjang, secara *default* `MAX_LEN=40`).

### 3. `src/model.py`
**Fungsi:** Tempat definisi arsitektur jaringan saraf (Neural Network).
**Deskripsi:**
- Memuat *class* **`LSTMClassifier`** dan **`TransformerClassifier`** yang diturunkan dari `torch.nn.Module`.
- **Arsitektur LSTM:** Terdiri dari lapisan Embedding, Bidirectional LSTM, Dropout, dan Linear (Fully Connected).
- **Arsitektur Transformer:** Dibangun *from scratch* mengikuti paper "Attention Is All You Need", meliputi `PositionalEncoding`, `TransformerEncoderLayer`, dan `TransformerEncoder` menggunakan mekanisme *self-attention*, yang diakhiri dengan *mean pooling* dan Linear layer.

### 4. `src/train.py`
**Fungsi:** Memuat fungsionalitas dan logika perulangan fase *training* (Training Loop) secara spesifik.
**Deskripsi:**
- Memuat fungsi **`train_model`** yang menjalankan iterasi per-epoch dari dataset menggunakan `DataLoader`.
- Menggunakan *Optimizer* **Adam** dan fungsi kerugian **CrossEntropyLoss** (cocok untuk multikelas).
- Dilengkapi dengan fitur **Gradient Clipping** (`clip_grad_norm_`) untuk mengontrol dan memastikan agar kalkulasi gradien pada RNN tidak "meledak" (menghindari *Exploding Gradient*).
- Menghitung akurasi dan kerugian (*loss*) secara bersamaan untuk data uji dan latih.

### 5. `src/evaluate.py`
**Fungsi:** Mengompilasi dan mengkalkulasi matriks performa (metrik evaluasi) model pasca-training.
**Deskripsi:**
- Memuat fungsi **`evaluate_model`** dengan menggunakan API dari scikit-learn untuk mengeluarkan hasil berupa *Accuracy*, *Precision*, *Recall*, dan *F1-score* (secara *weighted*).
- Menghentikan proses gradient tracking lewat baris blok `torch.no_grad()` agar memori tidak bocor dan proses inferensi berjalan lebih cepat dan efisien.

## Cara Penggunaan
Untuk menjalankan pipeline kode secara penuh, pastikan Anda berada di direktori `kode_program_penugasaan6` dan eksekusi file utama:
```bash
python lstm_project/main.py
```
