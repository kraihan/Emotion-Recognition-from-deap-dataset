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
