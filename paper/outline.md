# Compression-Aware Deepfake Detection — Paper Outline

## IEEE-Style Structure for MSc Research Paper

---

## Abstract (~200 words)
- **Problem:** Video deepfake detection degrades significantly under video compression (H.264, varying quality factors), which is the dominant distribution format
- **Gap:** Existing CNN-based detectors trained on raw/lightly-compressed data suffer dramatic performance drops on heavily compressed videos
- **Approach:** We propose a compression-aware hybrid framework combining:
  - Spatial features (EfficientNet-B0) for semantic face analysis
  - Frequency features (2D DWT + lightweight CNN) for compression artifact detection
  - Late fusion via concatenation + MLP
- **Experiments:** Systematic evaluation on FaceForensics++ at c0/c23/c40 compression levels, cross-dataset testing on Celeb-DF v2
- **Key finding:** The hybrid model consistently outperforms spatial-only and frequency-only baselines, especially under heavy compression (c40), demonstrating improved robustness
- **Contribution:** MSc-level research demonstrating that combining spatial and frequency-domain features leads to more compression-robust deepfake detection

---

## 1. Introduction
- Rise of deepfakes and societal implications (misinformation, fraud)
- Video compression is ubiquitous (social media, messaging, streaming) — detectors must handle this
- Problem: Most detectors evaluated on raw/lightly-compressed data; real-world performance on compressed video is understudied
- Our contribution:
  1. A hybrid spatial+frequency architecture for deepfake detection
  2. Systematic compression robustness evaluation (c0 vs c23 vs c40)
  3. Cross-dataset generalization study (FF++ → Celeb-DF v2)
  4. Open-source implementation with interactive demo

---

## 2. Related Work (organize by theme)

### 2.1 Face Manipulation Detection
- Early methods: hand-crafted features (blending artifacts, head pose inconsistencies)
- CNN-based: XceptionNet [Rössler et al., 2019], EfficientNet-based detectors
- Reference: FaceForensics++ benchmark [Rössler et al., 2019]

### 2.2 Frequency-Domain Analysis
- Frequency artifacts in GAN-generated images [Durall et al., 2020]
- Spectral analysis for face forgery detection [Li et al., 2021]
- DCT-based features [Qian et al., 2020, F3-Net]
- DWT in image forensics

### 2.3 Compression Robustness
- Impact of JPEG/H.264 compression on detection performance
- Compression-aware training strategies
- Gap: limited systematic study across multiple compression levels with hybrid approaches

### 2.4 Multi-Branch / Fusion Approaches
- Two-stream networks combining RGB and frequency/noise
- Feature-level vs decision-level fusion

---

## 3. Methodology

### 3.1 Problem Formulation
- Binary classification: Real (0) vs Fake (1) at frame level
- Video-level prediction via aggregation of frame probabilities
- Formally: P(fake | video) = mean(P(fake | frame_i))

### 3.2 Architecture Overview
- *Include architecture diagram*
- Spatial Branch: EfficientNet-B0 (ImageNet pretrained) → 1280-D feature vector
- Frequency Branch: Grayscale → 2D DWT (Haar, level 1) → 4-channel tensor (cA, cH, cV, cD) → Small CNN → 128-D feature vector
- Fusion: Concat(1280, 128) = 1408-D → FC(512) → ReLU → Dropout → FC(256) → ReLU → Dropout → FC(1) → Sigmoid

### 3.3 Preprocessing Pipeline
- Face detection: MTCNN with margin=40px
- Face crops resized to 224×224
- Frame sampling at 5 fps (max 100 frames/video)
- Data augmentation: horizontal flip, color jitter, rotation (±5°)

### 3.4 DWT Feature Extraction
- Color conversion: RGB → Grayscale
- Resize to 224×224 so DWT output = 112×112 per subband
- Haar wavelet, single-level decomposition
- 4 subbands stacked as 4-channel input: (4, 112, 112)
- Rationale: DWT captures frequency-domain artifacts that differ between real (natural noise) and fake (GAN artifacts) faces, and these artifacts are affected differently by compression

### 3.5 Training Strategy
- Loss: Binary Cross-Entropy with Logits
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: Cosine annealing
- Batch size: 16 (fits free T4 GPU)
- Epochs: 15
- Compression-aware training: model trained on mixed c0+c23+c40 data

---

## 4. Experiments

### 4.1 Datasets
- **FaceForensics++ (FF++)**: 1000 original + 4000 manipulated videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures), at three compression levels: c0 (raw), c23, c40
  - Split: 720 train / 140 val / 140 test (official)
  - Focus subsets: Deepfakes + FaceSwap for practicality
- **Celeb-DF v2**: 590 real + 5639 synthesized videos (higher quality deepfakes)
  - Used only for cross-dataset testing

### 4.2 Evaluation Metrics
- Frame-level: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Video-level: Mean frame probability → threshold-based classification

### 4.3 Experimental Setup
- Hardware: Google Colab (NVIDIA T4, 16GB VRAM) and Kaggle (P100)
- Framework: PyTorch 2.0+
- Training time: ~X hours for hybrid model on full FF++ subset

---

## 5. Results

### 5.1 Compression Robustness (Main Result)
- **Table:** AUC scores for each model variant × compression level
  | Model     | c0    | c23   | c40   |
  |-----------|-------|-------|-------|
  | Spatial   | <AUC> | <AUC> | <AUC> |
  | Frequency | <AUC> | <AUC> | <AUC> |
  | Hybrid    | <AUC> | <AUC> | <AUC> |

- **Key observation:** All models degrade with heavier compression, but hybrid degrades least
- **Figure:** Line plot of AUC vs compression (Fig. 1)

### 5.2 Ablation Study
- Spatial-only vs Frequency-only vs Hybrid
- Hybrid outperforms both individual branches at every compression level
- Frequency branch contributes most under heavy compression (c40)
- **Figure:** Bar chart comparing branches (Fig. 2)

### 5.3 Training Compression Strategy
- Trained on c0-only vs c23-only vs c0+c23+c40 (mixed)
- Mixed-compression training provides best generalization
- Single-compression training overfits to that quality level

### 5.4 Cross-Dataset Generalization (FF++ → Celeb-DF v2)
- AUC on Celeb-DF: <AUC_celeb>
- Expected degradation (domain shift), but still reasonable
- **Table:** Comparison of in-dataset vs cross-dataset

---

## 6. Discussion
- Why the hybrid approach helps: spatial captures semantic artifacts, frequency captures spectral/compression-level artifacts — complementary information
- DWT effectiveness: Haar wavelet decomposes images into frequency subbands that reveal manipulation traces differently under compression
- Limitations:
  - Limited to face-swap type manipulations (not reenactment-only)
  - Subset of FF++ used due to compute constraints
  - Single-level DWT (could explore multi-scale)
  - No temporal modeling (frame-level only)
- Practical insight: the 128-D frequency branch adds minimal compute (<5% overhead) but measurable improvement

---

## 7. Conclusion
- We presented a compression-aware hybrid deepfake detection framework
- Systematic evaluation shows that combining spatial and frequency features improves detection robustness across compression levels
- The hybrid model achieves <AUC_hybrid_c40> AUC even on heavily compressed (c40) videos, vs <AUC_spatial_c40> for spatial-only
- Cross-dataset evaluation confirms reasonable generalization
- Contributions are reproducible and MSc-level appropriate

---

## 8. Future Work
- Multi-scale DWT (multi-level decomposition)
- Temporal modeling (sequential frame analysis)
- Additional manipulation types (full FF++ and beyond)
- WhatsApp/YouTube re-compression simulation
- Explainability: Grad-CAM visualization on spatial and frequency branches
- Larger backbone models (EfficientNet-B3/B4)

---

## References (key papers to cite)
1. Rössler, A., et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV, 2019.
2. Li, Y., et al. "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics." CVPR, 2020.
3. Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML, 2019.
4. Durall, R., et al. "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions." CVPR, 2020.
5. Qian, Y., et al. "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues." ECCV, 2020.
6. Frank, J., et al. "Leveraging Frequency Analysis for Deep Fake Image Recognition." ICML, 2020.
7. Chollet, F. "Xception: Deep Learning with Depthwise Separable Convolutions." CVPR, 2017.
