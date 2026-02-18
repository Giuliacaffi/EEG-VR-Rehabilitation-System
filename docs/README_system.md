# System Architecture

## Signal Processing Pipeline

1. EEG Acquisition
2. Band-pass filtering
3. Artifact rejection
4. Epoch extraction
5. Feature extraction (PSD / CSP / time-domain features)
6. Classification (LDA / SVM / Ensemble)
7. VR feedback rendering

## Training Logic

- Each level consists of 30 repetitions
- Performance threshold for unlocking next level
- Automatic session summary generation

## User Types

Complete Paralysis:
EEG-only interaction for assistive control.

Incomplete Paralysis:
EEG + EMG integration for intention-execution training.

## Safety and Adaptivity

- No physical risk
- Controlled VR environment
- Performance-based adaptation
