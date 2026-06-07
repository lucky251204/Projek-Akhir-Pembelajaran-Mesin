# Analisis Kapasitas Ruang Laten pada Convolutional Autoencoder dengan Penyelarasan Representasi Lintas Modalitas Berbasis RNN

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Penelitian ini menganalisis pengaruh kapasitas ruang laten (`latent_dim`) pada Convolutional Autoencoder (CAE) terhadap kualitas rekonstruksi citra, sekaligus mengeksplorasi penyelarasan representasi lintas modalitas (gambar dan teks) menggunakan arsitektur RNN berbasis custom `RNNCell`. Dataset berupa pasangan gambar hewan dan caption teks yang dikumpulkan dari API Pexels.

---

## Project Structure

```
.
├── pixels_scraper.py        # Async scraper dari Pexels API
├── post_processing.py       # Filtering caption (copyright, noise, dll)
├── split_dataset.py         # Split train/val/test per keyword
├── eda_dataset.ipynb        # Exploratory Data Analysis dataset
├── pembelajaran-mesin-project_2_0.ipynb  # Training pipeline utama
├── pexels_dataset/
│   ├── metadata.csv         # Raw metadata hasil scraping
│   ├── metadata_clean.csv   # Metadata setelah cleaning
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── .env.example             # Template environment variables
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install torch torchvision torchmetrics
pip install pandas scikit-learn tqdm
pip install aiohttp aiofiles requests
pip install matplotlib seaborn jupyter
```

Atau:

```bash
pip install -r requirements.txt
```

> Training dilakukan pada Kaggle dengan dual T4 GPU. Untuk lokal, sesuaikan `BATCH_SIZE` dan `EPOCHS`.

---

## Usage

### 1. Setup environment

Salin `.env.example` ke `.env` dan isi dengan API key Pexels:

```bash
cp .env.example .env
```

```env
PEXELS_SECRET_KEY=your_key_here
```

### 2. Scraping dataset

```bash
python pixels_scraper.py
```

Menghasilkan `pexels_dataset/metadata.csv` berisi ~50k pasangan gambar + caption dari 115 keyword hewan.

### 3. Preprocessing

```bash
# Filtering caption
python post_processing.py

# Split train/val/test (80/10/10 per keyword)
python split_dataset.py
```

### 4. EDA

Jalankan `eda_dataset.ipynb` untuk eksplorasi dataset: distribusi keyword, panjang caption, distribusi warna RGB, dan statistik resolusi gambar.

### 5. Training

Buka dan jalankan `pembelajaran-mesin-project_2_0.ipynb`. Pipeline training terdiri dari tiga tahap terpisah:

| Tahap        | Komponen                    | Loss                       |
| ------------ | --------------------------- | -------------------------- |
| Train First  | ImageEncoder + ImageDecoder | MSE rekonstruksi citra     |
| Train Second | RNN TextEncoder             | MSE vs latent ImageEncoder |
| Train Third  | RNN TextDecoder             | Cross-entropy caption      |

Hyperparameter default:

```python
LATENT_DIMS = [4, 8, 16, 32, 64, 128, 256]  # sweep
EPOCHS      = 30
LR          = 1e-3
BATCH_SIZE  = 1280
```

---

## Architecture

### ImageEncoder

```
Input [3 × 64 × 64]
→ Conv2d(3, 32) + BN + ReLU + MaxPool2d(2)     # 32×32
→ Conv2d(32, 64) + BN + ReLU + MaxPool2d(2)    # 16×16
→ Conv2d(64, 128) + BN + ReLU + AdaptiveAvgPool2d(4)  # 4×4
→ Linear(2048, latent_dim)
```

### ImageDecoder

```
Input [latent_dim]
→ Linear(latent_dim, 2048) → reshape(128, 4, 4)
→ ConvTranspose2d(128, 64) + BN + ReLU          # 8×8
→ ConvTranspose2d(64, 32) + BN + ReLU           # 16×16
→ ConvTranspose2d(32, 16) + BN + ReLU           # 32×32
→ ConvTranspose2d(16, 3) + Tanh                 # 64×64
```

### RNN TextEncoder

```
Input [caption tokens, max_len=32]
→ Embedding(vocab_size, embed_dim=128)
→ RNNCell: h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)  [hidden_dim=256]
→ Linear(256, latent_dim)
```

### RNN TextDecoder

```
Input [latent_dim]
→ Linear(latent_dim, hidden_dim) + Tanh   # init hidden state
→ Embedding + RNNCell (autoregressive, teacher forcing)
→ Linear(hidden_dim, vocab_size)          # logits per token
```

---

## Results

### Train First — CNN Autoencoder (64×64, torchmetrics SSIM)

|  Latent Dim  |       MSE ↓       |     SSIM ↑     |  Params  |
| :-----------: | :----------------: | :--------------: | :-------: |
|       4       |      0.117904      |      0.1138      |  285,271  |
|       8       |      0.099795      |      0.1294      |  301,659  |
|      16      |      0.073642      |      0.1675      |  334,435  |
|      32      |      0.058328      |      0.2075      |  399,987  |
| **64** | **0.047531** |      0.2486      |  531,091  |
|      128      |      0.050252      |      0.2444      |  793,299  |
| **256** |      0.048898      | **0.2523** | 1,317,715 |

> Catatan: SSIM menggunakan `torchmetrics.StructuralSimilarityIndexMeasure` (Gaussian window 11×11). Nilai berbeda dengan iterasi sebelumnya yang menggunakan implementasi custom berbasis statistik global.

### Train Second — RNN TextEncoder (Text → Image Latent)

|  Latent Dim  |    Val Loss ↓    |    Test Loss ↓    | Params |
| :-----------: | :----------------: | :----------------: | :-----: |
|       4       |      5.022590      |      5.000492      | 802,052 |
|      32      |      0.845218      |      0.856818      | 809,248 |
|      64      |      0.657929      |      0.665728      | 817,472 |
| **256** | **0.460341** | **0.464590** | 866,816 |

### Train Third — RNN TextDecoder (Image Latent → Text)

| Latent Dim | Val Loss ↓ | Test Loss ↓ |
| :--------: | :---------: | :----------: |
|     4     |  2.739820  |   2.751476   |
|    128    |  2.710046  |   2.718260   |
|    256    |  2.715880  |   2.726520   |

> Loss plateau di ~2.71 untuk semua latent_dim mengindikasikan adanya semantic gap antara representasi visual dan tekstual ketika dilatih secara terpisah.

---

## Key Findings

* **Optimal latent_dim** : z=64 untuk MSE, z=256 untuk SSIM pada Train First
* **Diminishing returns** : peningkatan latent_dim di atas z=64 tidak memberikan perbaikan MSE yang signifikan
* **RNN vs FNN** : RNN TextEncoder mencapai performa alignment sebanding dengan FNN iterasi sebelumnya (test loss ~0.46 di z=256) dengan keunggulan pemodelan urutan kata
* **Semantic gap** : Train Third mengungkap bahwa latent space yang dioptimasi untuk rekonstruksi piksel tidak secara otomatis encode informasi semantik yang cukup untuk language generation

---

## Authors

| Nama                  | NIM         |
| --------------------- | ----------- |
| Lucky Surya Revansyah | 24031554090 |
| Fauzan Rafingudin     | 24031554111 |

Program Studi S1 Sains Data — Universitas Negeri Surabaya (UNESA)
Mata Kuliah Pembelajaran Mesin 2025/2026 Genap
