# Relational Group Activity Recognition

<div align="center">

<!-- Replace with your actual banner image -->
![Project Banner](imgs/Screenshot%202026-05-04%20020227_4.png)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ECCV 2018](https://img.shields.io/badge/Paper-ECCV%202018-green)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mostafa_Ibrahim_Hierarchical_Relational_Networks_ECCV_2018_paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-Volleyball-orange)](https://github.com/mostafa-saad/deep-activity-rec)

> **Implementation of [*Hierarchical Relational Networks for Group Activity Recognition*](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mostafa_Ibrahim_Hierarchical_Relational_Networks_ECCV_2018_paper.pdf) (ECCV 2018) on the Volleyball dataset.**

</div>

---

## Overview

Rather than relying on simple pooling methods (max, average) that collapse person-level features into a fixed-size scene vector and discard spatial and relational information, this project introduces a **Relational Layer** — a graph-based module that enriches each person's representation by explicitly modeling pairwise interactions with every other person in the scene.

A comprehensive ablation study covers **9 non-temporal baselines** and **2 temporal (LSTM-based) models**, all using a **ResNet-50** backbone and a fully vectorized relational layer implementation.

---

## Table of Contents

- [Relational Group Activity Recognition](#relational-group-activity-recognition)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [How the Relational Layer Works](#how-the-relational-layer-works)
  - [Key Features](#key-features)
  - [Usage](#usage)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Prepare the Dataset](#3-prepare-the-dataset)
    - [4. Stage 1 — Train Person Classifier](#4-stage-1--train-person-classifier)
    - [5. Stage 2 — Train Group Models](#5-stage-2--train-group-models)
  - [Dataset Overview](#dataset-overview)
    - [Train/Val/Test Split](#trainvaltest-split)
    - [Group Activity Labels](#group-activity-labels)
    - [Player Action Labels](#player-action-labels)
  - [Ablation Study](#ablation-study)
    - [Single-Frame Baselines](#single-frame-baselines)
      - [Baseline Descriptions](#baseline-descriptions)
      - [Performance Table](#performance-table)
      - [Baseline Comparison Visualizations](#baseline-comparison-visualizations)
    - [Temporal Models](#temporal-models)
    - [Per-Class Results — Best Model (`RCRG-2R-21C-conc`)](#per-class-results--best-model-rcrg-2r-21c-conc)
  - [Citation](#citation)
  - [License](#license)

---

## Introduction

Traditional pooling methods reduce a set of person features to a single scene vector, but lose the spatial and relational structure between players. The **Hierarchical Relational Network (HRN)** addresses this with a relational layer that constructs a graph over all detected persons and propagates pairwise relationship information across stacked layers.

<div align="center">
  <img src="imgs/Screenshot%202026-05-04%20012044_1.png" alt="Architecture Diagram" width="800"/>
  <br><em>Architecture Diagram: Hierarchical Relational Layers.</em>
</div>

### How the Relational Layer Works

**1. Graph Construction**

Each person in a frame is a node. Persons are sorted left-to-right by the center x-coordinate of their bounding boxes. Edges are defined by a clique structure that can span all players or be partitioned by team.

**2. Initial Person Features**

Each person's feature is extracted by a CNN backbone (ResNet-50) from their cropped image:

$$P_i^0 = \text{ResNet50}(I_i)$$

**3. Relational Update**

At layer $\ell$, person $i$'s updated representation aggregates pairwise interactions with all neighbors $j \neq i$:

$$P_i^\ell = \sum_{j \in E_i^\ell} F^\ell\!\left(P_i^{\ell-1} \oplus P_j^{\ell-1};\, \theta^\ell\right)$$

where $\oplus$ is concatenation and $F^\ell$ is a shared 2-layer MLP.

<div align="center">
  <img src="imgs/Screenshot%202026-05-04%20015317_2.png" alt="Relational Layer" width="700"/>
  <br><em>Pairwise relational update: each node aggregates messages from all neighbors via a shared MLP.</em>
</div>

The implementation replaces the naive $O(K^2)$ Python loop with a fully vectorized tensor operation using broadcasting:

```python
xi = x.unsqueeze(2).expand(b, n, n, d)   # (b, n, n, d)
xj = x.unsqueeze(1).expand(b, n, n, d)   # (b, n, n, d)
pairs = torch.cat([xi, xj], dim=-1)       # (b, n, n, 2d)
# self-edges are masked out before summing over neighbors
```

**4. Scene Representation**

After $L$ relational layers, team features are obtained by max-pooling (or concatenation) over person representations and fed to a classification head:

$$S = P_1^L \;\triangledown\; P_2^L \;\triangledown\; \dots \;\triangledown\; P_K^L$$

**5. Hierarchical Stacking**

Multiple relational layers are stacked to progressively compress features while refining relational context. Different clique structures at each layer allow the model to capture intra-team interactions before cross-team interactions.

---

## Key Features

| Feature | Description |
|---|---|
| **ResNet-50 Backbone** | Fine-tuned person-level feature extractor trained in Stage 1 |
| **Vectorized Relational Layer** | $O(1)$ PyTorch tensor ops replacing the original $O(K^2)$ Python loops |
| **Two-Stage Training** | Person classifier trained first, frozen and reused as a feature extractor |
| **Temporal Extension** | LSTM module over 9-frame clips enabling spatio-temporal relational reasoning |
| **Comprehensive Ablation** | 9 single-frame + 2 temporal models across relational depths, clique structures, and aggregation strategies |
| **Modern Training** | Mixed-precision (AMP), gradient accumulation, cosine LR scheduler, AdamW with fused kernels, TensorBoard |

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Boules123/Relational-Networks-and-Retrieval.git
cd Relational-Networks-and-Retrieval
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Set `data_dir` and `annot_dir` in the relevant YAML config (e.g. `configs/Person_Classifier.yml`).  
The dataset is sourced from publicly available volleyball videos; see the [original authors' repository](https://github.com/mostafa-saad/deep-activity-rec) for download instructions.

<!-- Dataset sample frame -->
<div align="center">
  <img src="imgs/Screenshot%202026-05-04%20015539_3.png" alt="Dataset Sample" width="700"/>
  <br><em>Example annotated frame from the Volleyball dataset showing player bounding boxes and action labels.</em>
</div>

### 4. Stage 1 — Train Person Classifier

```bash
bash scripts/train_person.sh
```

This trains `PersonModel` (ResNet-50 + FC head, 9 person-action classes) and saves checkpoints under `experiments/Person_classifier_<timestamp>/`.

### 5. Stage 2 — Train Group Models

```bash
bash scripts/train_group.sh
```

Before running, open `src/training/training_group.py` and set the path to the person-classifier checkpoint. The `exp_name` field in the YAML config determines which model is loaded from the model registry. Checkpoints are saved under `experiments/<exp_name>_<timestamp>/`.

---

## Dataset Overview

The Volleyball dataset comprises **4,830 annotated frames** drawn from 55 YouTube volleyball videos. Each frame includes bounding boxes for up to 12 players (6 per team), individual player action labels, and a group activity label for the whole scene.

### Train/Val/Test Split

| Split | Videos | Frames |
|-------|--------|--------|
| Train | 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54 | ~3,493 |
| Val   | 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51 | — |
| Test  | 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47 | ~1,337 |

### Group Activity Labels

| Label | Description |
|-------|-------------|
| `r_set` | Right team — setting |
| `r_spike` | Right team — spiking |
| `r-pass` | Right team — passing |
| `r_winpoint` | Right team — winning point |
| `l_winpoint` | Left team — winning point |
| `l-pass` | Left team — passing |
| `l-spike` | Left team — spiking |
| `l_set` | Left team — setting |

### Player Action Labels

| Label | Description |
|-------|-------------|
| `Waiting` | Player standing still |
| `Setting` | Setting the ball |
| `Digging` | Digging / defensive dig |
| `Falling` | Falling on ground |
| `Spiking` | Attacking spike |
| `Blocking` | Blocking at net |
| `Jumping` | Jumping |
| `Moving` | Moving laterally |
| `Standing` | Passive standing |

---

## Ablation Study

All models use a ResNet-50 backbone (weights from Stage 1) with all backbone parameters **frozen** during Stage 2. Training uses AdamW (lr=1e-4, wd=0.01), cosine LR schedule, batch size 16, gradient accumulation over 2 steps, and mixed-precision AMP.

### Single-Frame Baselines

#### Baseline Descriptions

- **B1-NoRelations**: Simple CNN + pooling baseline with no relational layer. Uses ResNet-50 features directly pooled per team via max-pooling to scene vector, passed through a shared FC layer (128 dims). Establishes lower bound without relational reasoning.

- **B2-RCRG-1R-1C**: Single relational layer over a complete clique (all players). Reduces 2048 dims to 128. Shows the benefit of one layer of pairwise interactions without hierarchical depth.

- **B3-RCRG-1R-1C-notTuned**: Identical to B2 but uses vanilla ResNet-50 (not fine-tuned in Stage 1). Demonstrates how critical backbone fine-tuning is (~10% accuracy drop), validating the two-stage training strategy.

- **B4-RCRG-2R-11C**: Two relational layers, each over a single complete clique (1C means 1 clique per layer). First layer reduces 2048→256, second reduces 256→128. Tests hierarchical depth with uniform clique structure.

- **B4-RCRG-2R-11C-conc**: Same architecture as B4 but replaces max-pooling per team with concatenation of all person features as the final scene representation. Shows the effect of aggregation strategy.

- **B5-RCRG-2R-21C**: Two relational layers with hierarchical clique structure (2C then 1C). First layer processes team-level cliques (2C = left & right team separately), second layer bridges across all players (1C = full clique). Enables intra-team then cross-team reasoning.

- **B5-RCRG-2R-21C-conc**: Same as B5 but with concatenation pooling. Tests hierarchical cliques with alternative aggregation.

- **B6-RCRG-3R-421C** (Best single-frame): Three relational layers with multi-scale cliques (4C→2C→1C). First layer uses 4 cliques, second uses 2, third uses 1. Progressively refines relational context through increasing receptive fields. Achieves best single-frame performance (87.43% acc).

- **B6-RCRG-3R-421C-conc**: Same as B6 with concatenation pooling.

#### Performance Table

| Model | Architecture | Pooling | Val Acc | Val F1 |
|-------|-------------|---------|---------|--------|
| **B1-NoRelations** | ResNet-50 → shared linear 128 | max per team | 86.63% | 0.8656 |
| **B2-RCRG-1R-1C** | ResNet-50 → RL(2048→128), 1 clique | max per team | 86.06% | 0.8599 |
| **B3-RCRG-1R-1C-notTuned** | Vanilla ResNet-50 (no fine-tune) → RL(2048→128) | max per team | 76.10% | 0.7623 |
| **B4-RCRG-2R-11C** | RL(2048→256) → RL(256→128), 1 clique each | max per team | 85.44% | 0.8532 |
| **B4-RCRG-2R-11C-conc** | Same as B4 | concat all persons | 85.32% | 0.8522 |
| **B5-RCRG-2R-21C** | RL(2048→256) per team → RL(256→128) all | max per team | 86.89% | 0.8675 |
| **B5-RCRG-2R-21C-conc** | Same as B5 | concat all persons | 83.81% | 0.8363 |
| **B6-RCRG-3R-421C** | RL(2048→512, 4C) → RL(512→256, 2C) → RL(256→128, 1C) | max per team | **87.43%** | **0.8743** |
| **B6-RCRG-3R-421C-conc** | Same as B6 | concat all persons | 86.98% | 0.8695 |

> **Notes:**
> - `-conc` suffix: final scene representation uses concatenation of all person features instead of max-pooling per team.
> - B3 demonstrates the impact of fine-tuning — dropping backbone fine-tuning causes a ~10% accuracy regression.
> - B6's 3-layer hierarchical clique structure (4→2→1) achieves the best single-frame result.

#### Baseline Comparison Visualizations

<div align="center">
  <h5>B1 - NoRelations (Baseline)</h5>
  <img src="outputs/B1_NoRelations/confusion_matrix%20(1).png" alt="B1 Confusion Matrix" width="500"/>
  <p><em>Acc: 86.63% | F1: 0.8656 — Simple pooling, no relational reasoning</em></p>
</div>

<div align="center">
  <h5>B2 - RCRG-1R-1C (Single Relational Layer)</h5>
  <img src="outputs/B2_RCRG_1R_1C/confusion_matrix%20(1).png" alt="B2 Confusion Matrix" width="500"/>
  <p><em>Acc: 86.06% | F1: 0.8599 — One layer over complete clique</em></p>
</div>

<div align="center">
  <h5>B3 - RCRG-1R-1C-notTuned (No Backbone Fine-tuning)</h5>
  <img src="outputs/B3_RCRG_1R_1C_notTuned/confusion_matrix%20(1).png" alt="B3 Confusion Matrix" width="500"/>
  <p><em>Acc: 76.10% | F1: 0.7623 — Demonstrates ~10% drop without fine-tuning</em></p>
</div>

<div align="center">
  <h5>B4 - RCRG-2R-11C (Two Layers, Uniform Cliques)</h5>
  <img src="outputs/B4_RCRG_2R_11C/confusion_matrix%20(1).png" alt="B4 Confusion Matrix" width="500"/>
  <p><em>Acc: 85.44% | F1: 0.8532 — Hierarchical depth with same clique structure</em></p>
</div>

<div align="center">
  <h5>B5 - RCRG-2R-21C (Two Layers, Hierarchical Cliques)</h5>
  <img src="outputs/B5_RCRG_2R_21C/confusion_matrix%20(1).png" alt="B5 Confusion Matrix" width="500"/>
  <p><em>Acc: 86.89% | F1: 0.8675 — Team-level then cross-team reasoning</em></p>
</div>

<div align="center">
  <h5>B6 - RCRG-3R-421C ⭐ Best Single-Frame</h5>
  <img src="outputs/B6_RCRG_3R_421C/confusion_matrix%20(1).png" alt="B6 Confusion Matrix" width="500"/>
  <p><em>Acc: 87.43% | F1: 0.8743 — Three layers with multi-scale cliques (4→2→1)</em></p>
</div>

### Temporal Models

Temporal models process a 9-frame clip per person through an LSTM (hidden_size=512), producing a per-person temporal summary before the relational layers.

| Model | Architecture | Val Acc | Val F1 |
|-------|-------------|---------|--------|
| **RCRG-2R-21C** | LSTM → RL(512→256, 2C) → RL(256→128, 1C) → max pool | — (not yet run) | — |
| **RCRG-2R-21C-conc** | LSTM → RL(512→256, 2C) → RL(256→128, 1C) → concat | **89.10%** | **0.8916** |

> The temporal model with concatenation pooling (`RCRG-2R-21C-conc`) outperforms all single-frame baselines by ~2%, confirming the benefit of temporal context.

### Per-Class Results — Best Model (`RCRG-2R-21C-conc`)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| r_set | 0.81 | 0.88 | 0.84 |
| r_spike | 0.94 | 0.89 | 0.92 |
| r-pass | 0.89 | 0.85 | 0.87 |
| r_winpoint | 0.93 | 0.92 | 0.93 |
| l_winpoint | 0.98 | 0.94 | 0.95 |
| l-pass | 0.86 | 0.92 | 0.89 |
| l-spike | 0.96 | 0.88 | 0.92 |
| l_set | 0.86 | 0.87 | 0.87 |

<!-- Best model confusion matrix -->
<div align="center">
  <img src="outputs/temporal/RCRG_2R_21C_conc/confusion_matrix%20(1).png" alt="Confusion Matrix" width="600"/>
  <br><em>Confusion matrix for the best temporal model (RCRG-2R-21C-conc).</em>
</div>

---

## Citation

If you use this code or build upon the relational layer approach, please cite the original paper:

```bibtex
@inproceedings{ibrahim2018hierarchical,
  title     = {Hierarchical Relational Networks for Group Activity Recognition and Retrieval},
  author    = {Ibrahim, Mostafa S. and Mori, Greg},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2018}
}
```

**Dataset:**

```bibtex
@inproceedings{ibrahim2016hierarchical,
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition},
  author    = {Ibrahim, Mostafa S. and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
