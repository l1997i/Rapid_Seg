[![Durham](https://img.shields.io/badge/UK-Durham-blueviolet)](#)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/l1997i/rapid_seg/blob/main/LICENSE)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white&style=flat)
[![arXiv](https://img.shields.io/badge/arXiv-2407.10159-b31b1b.svg)](https://arxiv.org/abs/2407.10159)
![Stars](https://img.shields.io/github/stars/l1997i/rapid_seg?style=social)
[![paper](https://img.shields.io/badge/Paper-b31b1b.svg)](https://www.luisli.org/assets/pdf/LI_ECCV2024_PAPiDSeg_v0.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

# 🔥 RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation [ECCV 2024 Oral]

A PyTorch implementation of **RAPiD** features and the **RAPiD-Seg** network for
3D LiDAR semantic segmentation. RAPiD (Range-Aware Pointwise Distance
Distribution) combines localized geometry and surface-material reflectivity into
a rigid-transform-invariant, density-adaptive descriptor, embedded by a
double-nested autoencoder and fused for single-modal (LiDAR-only) segmentation.

## Updates
- [2024.08] RAPiD was selected as an ✨ **Oral** ✨ at ECCV 2024.
- [2024.07] Our paper is available on arXiv, click [here](https://arxiv.org/abs/2407.10159) to check it out.

## 🚀 Features

- **4D Distance Computation**: combines geometric and reflectivity components for a comprehensive feature representation.
- **Rigid-Transform Invariance**: distances within rigid bodies are preserved under rotation/translation, so RAPiD is robust to viewpoint changes.
- **Range-Aware Processing**: adaptive neighbour count (`k_near/k_mid/k_far`) matched to LiDAR density at close/mid/far range.
- **R- and C-RAPiD-Seg**: intra-ring (R) and intra-class (C) variants; C-RAPiD-Seg uses pseudo-labels at test time.
- **Double-Nested AE + Class-Aware Objective**: voxel-wise embeddings via VSA, trained with an MSE reconstruction + class-aware contrastive loss.
- **MinkowskiUNet34 Backbone**: full sparse-convolution backbone on NVIDIA GPUs; a lightweight voxel backbone is provided for CPU / quick experiments.
- **Comprehensive Testing**: `pytest` suite covering features, invariance and the network.

## 📦 Installation

### From Source

```bash
git clone https://github.com/l1997i/rapid_seg.git
cd rapid_seg

# with uv (recommended)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e .

# or with pip
pip install -e .
```

### Backbone (MinkowskiUNet34, NVIDIA GPU)

The default backbone is Minkowski-UNet34 and requires
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) with CUDA:

```bash
export CUDA_HOME=/usr/local/cuda
uv pip install ninja
uv pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-build-isolation
```

Without CUDA/MinkowskiEngine the code automatically falls back to a lightweight
voxel backbone (`backbone="auto"`), so features, examples and tests run anywhere.

### Development Installation

```bash
uv pip install -e ".[dev,all]"
```

## 🎯 Quick Start

```python
import torch
from rapid_seg import RAPiDCalculator
from rapid_seg.config import create_config

# sample point cloud
n_points = 1000
coordinates = torch.randn(n_points, 3) * 10.0     # 3D coordinates
reflectivity = torch.rand(n_points) * 0.8 + 0.1   # reflectivity values

# configure + compute RAPiD features
config = create_config("balanced", k_mid=8)
calculator = RAPiDCalculator(device=config.resolved_device())

k = config.get_k_for_standard_rapid()
rapid_features = calculator.compute_rapid_features(coordinates, reflectivity, k)

print(f"RAPiD features shape: {rapid_features.shape}")
print(f"Features range: [{rapid_features.min():.3f}, {rapid_features.max():.3f}]")
```

## 📚 Usage Examples

### Basic Feature Extraction

```python
from rapid_seg import RAPiDCalculator, PointCloudLoader

loader = PointCloudLoader()
coordinates, reflectivity = loader.load_and_preprocess("scan.bin")

calculator = RAPiDCalculator(device="cuda")
rapid_features = calculator.compute_rapid_features(coordinates, reflectivity, k=10)
```

### Range-Aware Processing

```python
range_aware_features = calculator.compute_range_aware_rapid(
    coordinates, reflectivity,
    k_close=10,   # close range: high density
    k_mid=7,      # mid range:   balanced
    k_far=5,      # far range:   low density
)
```

### Intra-Ring / Intra-Class RAPiD

```python
r_rapid = calculator.compute_r_rapid(coordinates, reflectivity)          # RoI = LiDAR ring
c_rapid = calculator.compute_c_rapid(coordinates, reflectivity, labels)  # RoI = semantic class
```

### Configuration Management

```python
from rapid_seg.config import create_config, RAPiDConfig

config = create_config("balanced")   # the paper's SemanticKITTI setting
config = create_config("fast")       # fast processing
config = create_config("accurate")   # high accuracy

custom = RAPiDConfig(k_near=10, k_mid=7, k_far=5, device="cuda")
```

## 🏋️ Training & Evaluation on SemanticKITTI

Download [SemanticKITTI](http://semantic-kitti.org/dataset.html) and arrange it
in the official layout:

```
<data_root>/sequences/{00..21}/velodyne/*.bin      # float32 [x, y, z, remission]
                                /labels/*.label     # uint32  (sem & 0xFFFF) | (inst << 16)
```

The raw ids are mapped to the 19 evaluation classes via the official
`learning_map`, and the official splits (train 00-07,09,10 · val 08 ·
test 11-21) are used automatically.

```bash
# R-RAPiD-Seg (fast variant) — MinkUNet34 backbone auto-selected on NVIDIA GPU
python scripts/train.py --dataset full --data_root /path/to/dataset --variant R

# C-RAPiD-Seg (uses R- and C-RAPiD features)
python scripts/train.py --dataset full --data_root /path/to/dataset --variant C
```

Training follows the paper: SGD (lr `1e-3`), 2-epoch warmup + cosine schedule,
an AE pretraining stage followed by full-network training with the AE objective
as a regulariser.

## 🔧 Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `k_near` / `k_mid` / `k_far` | Neighbours at close/mid/far range | 10 / 7 / 5 | 3-16 |
| `num_beams` | LiDAR rings for R-RAPiD | 64 | 32 / 64 / 128 |
| `delta` | Outlier threshold (normalised) | 0.9 | 0-1 |
| `device` | Processing device | `"auto"` | `"cpu"`, `"cuda"` |
| `batch_size` | Batch size for processing | 32 | 16-128 |
| `precision` | Floating point precision | `"float32"` | `"float16"`, `"float32"` |

## 🧪 Testing

The suite runs on the real SemanticKITTI sequence-04 subset shipped under
`data/` (`.bin`/`.label` and `.pth`); tests that need data skip gracefully if it
is absent. Coverage includes the data loaders and `learning_map`, RAPiD feature
shapes and rigid-transform invariance, the network forward/backward (both
variants), a real-frame loss-decrease check, and the mIoU meter.

```bash
pytest rapid_seg/tests/ -v
pytest rapid_seg/tests/ --cov=rapid_seg

# individual modules
pytest rapid_seg/tests/test_data_semantickitti.py -v   # loaders + learning_map
pytest rapid_seg/tests/test_features.py -v             # RAPiD features + invariance
pytest rapid_seg/tests/test_model.py -v                # network + real-frame training step
pytest rapid_seg/tests/test_metrics.py -v              # mIoU / accuracy
```

## 📖 Examples

- [`examples/01_getting_started.py`](examples/01_getting_started.py) — compute RAPiD features
- [`examples/02_train_semantickitti.py`](examples/02_train_semantickitti.py) — train on SemanticKITTI

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

## Citation

If you are making use of this work in any way, you must please reference the
following paper in any report, publication, presentation, software release or
any other associated materials:

[RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation](https://arxiv.org/abs/2407.10159) ([Li Li](https://luisli.org), [Hubert P. H. Shum](http://hubertshum.com/) and [Toby P. Breckon](https://breckon.org/toby/)), In Proc. Eur. Conf. Comput. Vis. (ECCV), 2024. [[pdf](https://www.luisli.org/assets/pdf/LI_ECCV2024_PAPiDSeg_v0.pdf)]

```bibtex
@inproceedings{li2024rapidseg,
  title = {{{RAPiD-Seg}}: {{Range-Aware}} {{Pointwise Distance Distribution}} {{Networks}} for {{3D LiDAR Segmentation}}},
  author = {Li, Li and Shum, Hubert P. H. and Breckon, Toby P.},
  keywords = {point cloud, semantic segmentation, invariance feature, pointwise distance distribution, autonomous driving},
  year = {2024},
  month = jul,
  publisher = {{Springer}},
  booktitle = {European Conference on Computer Vision (ECCV)},
}
```

## Acknowledgements

We build on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine),
[VoxSeT](https://github.com/skyhehe123/VoxSeT) and the
[SemanticKITTI](http://semantic-kitti.org/) benchmark.
