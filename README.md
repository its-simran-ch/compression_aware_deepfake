# 🔍 Compression-Aware Video Deepfake Detection

> **MSc Research Project** — A hybrid spatial-frequency framework for robust deepfake detection under video compression.

## Architecture

```
Video → Frame Sampling (5fps) → MTCNN Face Detection → 224×224 RGB Crops
                                                           │
                         ┌─────────────────────────────────┤
                         ▼                                 ▼
              ┌──────────────────┐              ┌──────────────────┐
              │  Spatial Branch  │              │ Frequency Branch │
              │  EfficientNet-B0 │              │  Grayscale→DWT   │
              │  (ImageNet)      │              │  Haar Wavelet    │
              │  → 1280-D       │              │  → Small CNN     │
              └────────┬─────────┘              │  → 128-D        │
                       │                        └────────┬─────────┘
                       │        Concatenate              │
                       └───────────┬─────────────────────┘
                                   ▼
                          ┌────────────────┐
                          │   Fusion MLP   │
                          │ 1408→512→256→1 │
                          └───────┬────────┘
                                  ▼
                    Frame Probability (Sigmoid)
                                  ▼
                      Mean Aggregation → Video Label
                     (REAL / FAKE / UNCERTAIN)
```

## Quick Start

### 1. Clone & Install

```bash
# On Mac (local)
git clone https://github.com/YOUR_USERNAME/compression_aware_deepfake.git
cd compression_aware_deepfake
pip install -r requirements.txt
```

```python
# On Colab / Kaggle
!git clone https://github.com/YOUR_USERNAME/compression_aware_deepfake.git
%cd compression_aware_deepfake
!pip install -q -r requirements.txt
```

### 2. Dataset Setup

**Your FF++ data is already on Google Drive** at `/content/drive/MyDrive/FFPP_raw/` with this structure:

```
FFPP_raw/
├── original_sequences/youtube/{raw,c23,c40}/videos/
└── manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/{raw,c23,c40}/videos/
```

**Celeb-DF v2** is available as a zip file — extract it:
```bash
python scripts/download_celebdf.py --zip_path /path/to/Celeb-DF-v2.zip --output_dir data/celeb_df
```

### 3. Preprocessing Pipeline

```bash
# Step 1: Generate train/val/test splits
python scripts/prepare_ffpp_splits.py \
    --data_root /content/drive/MyDrive/FFPP_raw \
    --output data/faceforensics/splits.json

# Step 2: Extract face crops (run on Colab with GPU)
python scripts/extract_faces_ffpp.py \
    --data_root /content/drive/MyDrive/FFPP_raw \
    --output_dir /content/drive/MyDrive/ffpp_faces \
    --splits_json data/faceforensics/splits.json \
    --compressions c0 c23 c40 \
    --manipulations Deepfakes FaceSwap \
    --target_fps 5 --max_frames 50 \
    --device cuda
```

### 4. Training

```bash
# Hybrid model (recommended) — run on Colab/Kaggle GPU
python src/training/train_ffpp.py \
    --metadata_csv /content/drive/MyDrive/ffpp_faces/metadata.csv \
    --data_root /content/drive/MyDrive/ffpp_faces \
    --mode hybrid \
    --compressions c23 c40 \
    --epochs 15 --batch_size 16
```

### 5. Evaluation

```bash
# Per-compression evaluation
python src/training/evaluate_compression_levels.py \
    --checkpoint results/checkpoints/best_hybrid_c23_c40.pth \
    --metadata_csv /content/drive/MyDrive/ffpp_faces/metadata.csv \
    --data_root /content/drive/MyDrive/ffpp_faces \
    --mode hybrid

# Ablation experiments (spatial vs frequency vs hybrid)
python src/training/run_ablations.py \
    --metadata_csv /content/drive/MyDrive/ffpp_faces/metadata.csv \
    --data_root /content/drive/MyDrive/ffpp_faces \
    --epochs 10

# Cross-dataset evaluation (on Celeb-DF v2)
python src/training/cross_dataset_eval.py \
    --checkpoint results/checkpoints/best_hybrid_c23_c40.pth \
    --celeb_root /content/drive/MyDrive/celeb_df_faces \
    --mode hybrid
```

### 6. Generate Plots

```bash
python scripts/plot_results.py --results_dir results/csv --output_dir results/plots
```

### 7. Run Demo

```bash
# Local (Mac)
streamlit run src/inference/streamlit_app.py

# On Colab (with ngrok for public URL)
!pip install pyngrok
from pyngrok import ngrok
# !streamlit run src/inference/streamlit_app.py &>/dev/null &
# public_url = ngrok.connect(8501)
# print(f"Demo URL: {public_url}")
```

---

## Project Structure

```
compression_aware_deepfake/
├── scripts/                          # CLI scripts
│   ├── download_faceforensics.py     # FF++ download reference
│   ├── download_celebdf.py           # Celeb-DF extraction
│   ├── prepare_ffpp_splits.py        # Train/val/test splits
│   ├── extract_faces_ffpp.py         # Face crop extraction
│   └── plot_results.py               # Paper-ready plots
├── src/
│   ├── datasets/
│   │   ├── ffpp_dataset.py           # FF++ PyTorch dataset
│   │   └── celebdf_dataset.py        # Celeb-DF dataset
│   ├── models/
│   │   ├── spatial_efficientnet.py   # EfficientNet-B0 branch
│   │   ├── frequency_dwt_branch.py   # DWT CNN branch
│   │   └── fusion_classifier.py      # Hybrid classifier
│   ├── training/
│   │   ├── train_ffpp.py             # Main training script
│   │   ├── evaluate_compression_levels.py
│   │   ├── run_ablations.py          # Systematic ablations
│   │   └── cross_dataset_eval.py     # FF++ → Celeb-DF
│   ├── utils/
│   │   ├── video_utils.py            # Frame sampling
│   │   ├── face_detection.py         # MTCNN face cropping
│   │   ├── dwt_utils.py              # DWT computation
│   │   └── metrics.py                # Eval metrics
│   └── inference/
│       ├── video_inference.py        # Video → prediction pipeline
│       └── streamlit_app.py          # Demo web app
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Colab: preprocess
│   ├── 02_training_experiments.ipynb  # Colab: train
│   └── 03_results_plots.ipynb        # Colab/local: plots
├── paper/
│   ├── outline.md                    # IEEE paper outline
│   └── draft_skeleton.tex            # LaTeX skeleton
├── results/
│   ├── csv/                          # Experiment metrics
│   └── plots/                        # Paper-ready figures
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Workflow Summary

| Step | What | Where | Time |
|------|------|-------|------|
| 1. Clone repo | `git clone` | Mac | 1 min |
| 2. Prepare splits | `prepare_ffpp_splits.py` | Colab | 1 min |
| 3. Extract faces | `extract_faces_ffpp.py` | Colab (GPU) | 1–3 hrs |
| 4. Train hybrid | `train_ffpp.py` | Colab/Kaggle (GPU) | 1–2 hrs |
| 5. Evaluate | `evaluate_compression_levels.py` | Colab (GPU) | 15 min |
| 6. Ablations | `run_ablations.py` | Colab (GPU) | 3–5 hrs |
| 7. Cross-dataset | `cross_dataset_eval.py` | Colab (GPU) | 15 min |
| 8. Plots | `plot_results.py` | Mac or Colab | 1 min |
| 9. Demo | `streamlit_app.py` | Mac or Colab | N/A |

## Key Design Decisions

- **EfficientNet-B0** chosen for balance of accuracy vs memory (fits free Colab T4)
- **Haar DWT** is simple, fast, and well-suited for compression artifact detection
- **128-D frequency features** add <5% compute overhead but measurable improvement under heavy compression
- **No temporal modeling** (LSTM/GRU) — keeps scope MSc-appropriate and explainable
- **Mixed-compression training** (c23+c40) provides best real-world robustness

## Citation

If you use this work, please cite:
```
@mastersthesis{chaudhary2026compression,
  title={A Compression-Aware Video Deepfake Detection Framework Using Spatial and Frequency-Domain Features},
  author={Chaudhary, Simran},
  year={2026},
  school={Your University}
}
```

## License

MIT License — see [LICENSE](LICENSE).
