# CAFuNet: Context-Aligned Topic Conditioning and Calibrated Fusion for Multimodal Crisis Informatics

<p align="center">
  <a href="https://shahid-135.github.io"><img src="https://img.shields.io/badge/Project%20Page-Live-blue?style=flat-square&logo=github" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&logo=arxiv" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-CrisisMMD-orange?style=flat-square&logo=huggingface" /></a>
  <img src="https://img.shields.io/badge/ACL-2025-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch" />
</p>

<p align="center">
  <b>Shahid Shafi Dar<sup>1</sup>, Mihir Kanchan Karandikar<sup>1</sup>, Chandravardhan Singh Raghaw<sup>1</sup>, Pranjal Pandey<sup>2</sup>, Nagendra Kumar<sup>1</sup></b><br>
  <sup>1</sup>Indian Institute of Technology Indore &nbsp;·&nbsp; <sup>2</sup>Indian Institute of Information Technology Bhagalpur
</p>

---

## 📌 Overview

**CAFuNet** (**C**ontext-**A**ligned **Fu**sion **Net**work) is a multimodal classification framework for humanitarian crisis event analysis from social media. It tackles the **semantic reliability gap** — where one modality provides critical evidence while the other contains noisy or contradictory signals — through three synergistic components:

| Component | Description |
|-----------|-------------|
| **Dual-Topic Conditioning** | Corpus-induced topics (via BERTopic) are prepended to both text and image encoder inputs, grounding both modalities in a shared semantic scaffold |
| **Context-Gated Calibration (CGC)** | Learnable Gaussian + Sigmoid + Trapezoidal membership functions compute per-feature gating scores to amplify reliable signals and suppress noise |
| **Projected Bilinear Block Fusion (PBBF)** | B parallel interaction blocks capture high-order cross-modal interactions via efficient low-rank bilinear approximation |

---

## 🏆 Results

### CrisisMMD & TSEqD Benchmarks

| Method | CrisisMMD mF1 | CrisisMMD WF1 | TSEqD mF1 | TSEqD WF1 |
|--------|:---:|:---:|:---:|:---:|
| BERT | 80.22 | 82.42 | 71.18 | 73.81 |
| VisualBERT | 81.89 | 82.88 | 72.46 | 74.55 |
| DMCC | 84.93 | 86.72 | 74.90 | 77.45 |
| CLMC | 85.18 | 86.46 | 74.86 | 77.70 |
| MIDLC | 85.34 | 86.28 | 74.79 | 77.14 |
| MMA | 85.21 | 87.00 | 75.10 | 77.92 |
| **CAFuNet (Ours)** | **89.69** | **90.32** | **77.66** | **80.57** |

> ✅ **+4.35 mF1** over best baseline on CrisisMMD · **+2.56 mF1** on TSEqD  
> ✅ **+28.75 F1** over zero-shot Gemini 2.0 Flash  
> ✅ **90.32 ± 0.20** F1 across 5 random seeds (stable & reproducible)

---

## 🏗️ Architecture

<!-- Replace with your actual figure -->
> 📷 **Figure:** `assets/architecture.png` — Add your architecture figure here

The model processes multimodal inputs (text + image) through:
1. **Topic Induction** from training corpus → learnable topic embedding matrix
2. **Dual conditioning** of both BERT and ViT encoders with shared topic tokens
3. **CGC module** for feature-level reliability calibration
4. **PBBF head** for cross-modal fusion + classification

---

## 📦 Setup

```bash
git clone https://github.com/Shahid-135/Shahid-135.github.io.git
cd Shahid-135.github.io
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, Transformers, BERTopic

---

## 🗂️ Datasets

| Dataset | Events | Classes | Train | Val | Test |
|---------|--------|---------|-------|-----|------|
| [CrisisMMD](https://crisisnlp.qcri.org/crisismmd) | 7 disasters (2017) | 7 | 6,055 | 989 | 946 |
| [TSEqD](https://github.com/) | Turkey-Syria Earthquake 2023 | 7 | 7,708 | 964 | 964 |

---

## 📊 Ablation

| Configuration | mF1 |
|---------------|:---:|
| Baseline (Simple Fusion) | 78.08 |
| + Topic-Guided Prompting | 85.07 (+2.78) |
| + CGC | 86.80 (+1.33) |
| + PBBF | 88.65 (+1.23) |
| **+ Contrastive Loss (Full)** | **89.69 (+0.86)** |

---

## 📄 Citation

```bibtex
@misc{dar2025cafunet,
  title     = {Context-Aligned Topic Conditioning and Calibrated Fusion
               for Multimodal Crisis Informatics},
  author    = {Shahid Shafi Dar and Mihir Kanchan Karandikar and
               Chandravardhan Singh Raghaw and Pranjal Pandey and
               Nagendra Kumar},
  year      = {2025},
  eprint    = {2410.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2410.XXXXX},
}
```

---

## 🔗 Links

- 🌐 **Project Page:** [shahid-135.github.io](https://shahid-135.github.io)
- 📖 **Paper:** [arXiv](#)
- 🤗 **Dataset:** [CrisisMMD](#)

---

<p align="center">IIT Indore &nbsp;·&nbsp; IIIT Bhagalpur &nbsp;·&nbsp; 2025</p>
