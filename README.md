# 🧠 Emotion Recognition from EEG using DEAP Dataset

This repository contains the source code for the preprint article on **Emotion Recognition using the DEAP Dataset**, focusing on signal-based features and binary classification for **Valence** and **Arousal** dimensions.

---

## 📊 Dataset Used
- **Dataset**: [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- **EEG Shape**: `(1280, 32, 7680)` — 1280 trials, 32 EEG channels, 60 seconds per trial (128Hz)
- **Labels (Valence & Arousal)**: `(1280,)`

---

## 🧾 Label Processing

To maintain **balanced class distribution**, valence and arousal scores are thresholded:

- **Valence Threshold**: `< 5.05 → Class 0`, `>= 5.05 → Class 1`
- **Arousal Threshold**: `< 5.15 → Class 0`, `>= 5.15 → Class 1`

| Dimension | Class 0 | Class 1 |
|-----------|---------|---------|
| Valence   | 642     | 638     |
| Arousal   | 635     | 645     |

The labels are multiplied to match the number of EEG time windows for model input.

---

## 🧠 Extracted Features

All features preserve the original EEG shape `(1280, 32, 7680)` for direct use in CNN models.

### ✅ 1. Wavelet Energy Bands (WEB)
```python
def compute_wavelet_energy_bands(eeg_data, wavelet='db4', level=5):
    """
    Computes Wavelet Energy Bands (WEB) for EEG data.
    Energy per wavelet band is interpolated back to original EEG length.
    """
```
### ✅ 2. Hilbert-Huang Transform Entropy (HHTE) 
```python
def compute_hilbert_entropy_full(eeg_data, eps=1e-10):
    """
    Computes point-wise entropy using normalized Hilbert amplitude envelope.
    """
```
### ✅ 3. Hilbert Spectrum Energy (HSE) 
```python
def compute_hilbert_spectrum_energy(eeg_data):
    """
    Computes the squared amplitude of the Hilbert analytic signal.
    """
```
### ✅ 4. Wavelet Differential Entropy (DE) 
```python
def compute_wavelet_de(eeg_data, wavelet='db4', level=1):
    """
    Computes per-channel wavelet-based DE and repeats across time axis.
    """
```

## 🧪 Final Shapes
After processing:

EEG Data: (1280, 32, 7680)

Labels (Valence): (1280,) → Balanced binary

Labels (Arousal): (1280,) → Balanced binary


## 🔁 Feature Representation and Mapping

This section outlines the complete process of transforming raw EEG features into a format suitable for deep learning using CNNs. The following steps are applied to each extracted feature (HSE, HHTE, DE, WEB, etc.) to preserve spatiotemporal information:


### ✅ 1. Feature Scaling 

Each extracted feature (e.g., Hilbert Spectrum Energy, Wavelet Entropy) is conceptually scaled to a common range to remove magnitude variations across trials. This ensures that the neural network doesn't get biased by signal amplitude differences between subjects or channels.


### ✅ 2. Normalization 

In practice, **min-max normalization** is applied **per sample and per channel** to bring values to the [0, 1] range and stabilize training:

```python
x_min = feature.min(axis=(1, 2), keepdims=True)
x_max = feature.max(axis=(1, 2), keepdims=True)
feature_scaled = (feature - x_min) / (x_max - x_min + 1e-8)
This helps maintain numerical stability and improves model convergence.
```
### ✅ 3. Temporal Segmentation
The time dimension (7680 samples) is segmented into smaller fixed-length windows to extract temporal dynamics.
```
python
depth = 128  # Segment length
segments = 7680 // depth  # 60 segments
reshaped = feature_scaled.reshape(samples, channels, segments, depth)
```
This converts the feature shape into (samples, channels, 60 segments, 128 time steps).

### ✅ 4. Spatial Representation
Each 32-channel EEG trial is projected into a 2D spatial layout (8×8 grid) using a predefined channel mapping.
```
python
spatial_grid = np.zeros((8, 8, time_len))
```
The function maps real EEG channels to grid positions while filling unused slots with zeros. This results in a spatiotemporal representation:
```
python
spatiotemporal = (samples, segments, 8, 8, depth)
```
Finally, the data is reshaped to match the 3D CNN input format:
```
python
final_input = spatiotemporal.reshape(-1, 8, 8, 128)
```
This preserves both spatial and temporal characteristics of the EEG signal, allowing deep models to capture rich patterns.



## 🧠 Model Architecture: Complex 3D CNN

The model is a multi-layer **3D Convolutional Neural Network (3D-CNN)** designed to learn spatiotemporal features from EEG data. It processes input volumes of shape `(1, 8, 8, 128)` representing EEG signals across spatial grid and temporal segments.

### ✅ Architecture Overview

- **3 Convolution Blocks**:
  - Each block contains: `Conv3D → BatchNorm3D → ReLU → MaxPool3D`
  - Kernel size: `(3, 3, 7)`
  - Pool sizes: `(1,2,2) → (2,2,4) → (2,2,4)`
- **Dropout Layer**: Applied before fully connected layers with `dropout_rate=0.6`
- **Fully Connected Layers**:
  - FC1: Linear → ReLU
  - FC2: Output logits (binary classification: Valence or Arousal)

### ✅ Code Snippet (Model Initialization)

```python
model = Complex3DCNN(num_classes=2).to('cuda' if torch.cuda.is_available() else 'cpu')
```

### ⚙️ Training Setup
Optimizer: Adam

Learning Rate: 1e-4

Weight Decay: 1e-5

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
Learning Rate Scheduler: ReduceLROnPlateau
```

Mode: 'min'

Patience: 3

Factor: 0.5

Adjusts learning rate when validation loss plateaus.

```
python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True
)
```

The model is trained separately for valence and arousal classification with balanced labels.

