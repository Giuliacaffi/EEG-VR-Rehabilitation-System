# EEG-VR Rehabilitation System

A human-centered, multi-level neurorehabilitation system integrating EEG-based motor intention decoding with immersive Virtual Reality environments.

## Research Context

Motor rehabilitation for individuals with severe motor impairment requires adaptive, motivating, and neurophysiologically grounded training paradigms. Traditional rehabilitation approaches often suffer from low engagement and limited scalability.

This project proposes a structured rehabilitation framework combining:

- EEG-based motor intention decoding
- Multi-level adaptive training logic
- Immersive Virtual Reality interaction
- Human-centered design principles

The system is designed for both assistive (complete paralysis) and rehabilitative (partial motor function) use cases.

---

## System Overview

The architecture integrates neural signal processing with immersive VR feedback.

EEG signals → Preprocessing → Feature Extraction → Classification → Feedback in VR

The system adapts training difficulty based on performance stability and classification accuracy.

---

## Core Components

### 1. EEG-Based Motor Intention Decoding
- Motor imagery paradigm
- Rest vs Movement classification
- Progressive multi-class decoding
- Performance threshold unlocking logic

### 2. Adaptive Multi-Level Training
- Level 1: Binary (Rest vs Movement)
- Level 2: Left vs Right motor imagery
- Higher levels unlocked based on stability criteria

### 3. Multimodal Extension
For users with residual motor function:
- EEG + EMG + IMU integration
- Intention–execution consistency analysis

### 4. Human-Centered Design
- Minimal cognitive load interface
- Real-time feedback without overstimulation
- Performance tracking through companion monitoring application

---

## Validation Strategy

- Within-session classification accuracy
- Cross-session stability
- Intention–execution coherence (EEG-EMG coupling)
- User adherence and engagement metrics

---

## Research Contribution

This project contributes to the development of adaptive neurotechnology systems by integrating:

- Signal processing
- Machine learning
- Rehabilitation science
- Human–machine interaction principles

The framework is designed to support long-term engagement and scalable neurorehabilitation deployment.

---

## Repository Structure

- `docs/` → System documentation and conceptual framework
- `src/` → Signal processing and classification modules
- `figures/` → System architecture diagrams and workflow illustrations

---

## Future Work

- Online real-time classification pipeline
- Transfer learning for subject adaptation
- Closed-loop BCI integration with assistive devices
