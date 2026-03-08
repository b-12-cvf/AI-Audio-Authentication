# AI-Audio-Authentication | AI Audio Deepfake Detection Core

## The Architecture
This repository contains the foundational machine learning architecture and signal processing blueprints for detecting synthetic audio and voice cloning. 

Rather than relying on human auditory perception, this multi-modal solution treats audio as a multidimensional data structure. By synthesizing deep spectral analysis (via a custom ResNet-18 topology) with biological silence heuristics, this framework establishes a robust defense against both "Fully Synthetic" and "Partial" audio attacks.

## Performance & Visual Proof
The neural engine extracts Mel-Frequency Cepstral Coefficients (MFCCs) to isolate the deep acoustic features that differentiate organic vocal cord resonance from digital synthesis.

| Organic Audio Signature | Synthetic Audio Clone |
| :---: | :---: |
| <img src="assets/organic_spectrogram.PNG" width="400"> | <img src="/assets/synthetic_spectrogram.PNG" width="400"> |
| **Classification: Authentic** | **Classification: Deepfake** |
| *Natural biometric silence gaps detected.* | *High-frequency artifacting and phase anomalies present.* |

## Core Capabilities
* **Spectral Mapping:** Standardized audio-to-image tensor extraction with Min-Max normalization to prevent gradient explosion.
* **Neural Classification:** A layered Deep Neural Network (ResNet) built on TensorFlow/Keras, optimized for high-dimensional auditory pattern recognition.
* **Biometric Heuristics:** Advanced silence-duration calculations to verify physiological respiratory continuity (humans breathe; AI does not).
* **Gaussian Augmentation:** Prevents Batch Normalization collapse on localized datasets.

---

## How to Use This Blueprint

This repository is designed for two distinct execution modes:

### Mode 1: Local Demonstration (Plug & Play)
By default, the `neural_audio_engine.ipynb` is configured for immediate local execution. 
1. Install dependencies via `pip install -r requirements.txt`.
2. Run the notebook. It will automatically utilize the audio files provided in the `/samples` directory.
3. The engine will perform a localized micro-batch training sequence using Gaussian data augmentation to prevent mode collapse, allowing you to test the inference pipeline on specific target files instantly.

### Mode 2: Enterprise Production Scale (General Use)
To train this architecture for wide, general-use deployment, the neural network requires exposure to a massive corpus of diverse audio signatures. 
1. Download the **ASVspoof 2019 Logical Access** dataset (or your preferred enterprise dataset) to your local machine.
2. Navigate to **Cell 6 (Enterprise Training Pipeline)** in the Jupyter Notebook.
3. Update the `ASVSPOOF_PATH` to your local directory.
4. **Uncomment** the execution lines in Cell 6. 
5. The pipeline will transition to utilizing a production-grade data generator, processing the 100,000+ files batch-by-batch to optimize the CNN weights for global accuracy.

## Dependencies
* `tensorflow`
* `librosa`
* `opencv-python`
* `numpy`
* `matplotlib`
