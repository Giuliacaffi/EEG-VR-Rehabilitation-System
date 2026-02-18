# EEG-VR Rehabilitation System

A human-centered, multi-level neurorehabilitation system integrating EEG-based motor intention decoding with immersive Virtual Reality environments.

---

## Research Context

Motor rehabilitation for individuals with severe motor impairment requires adaptive, motivating, and neurophysiologically grounded training paradigms. Traditional rehabilitation approaches often suffer from limited engagement, reduced adherence, and insufficient personalization.

This project proposes a structured neurotechnology framework combining:

- EEG-based motor intention decoding  
- Progressive multi-level training logic  
- Immersive Virtual Reality interaction  
- Human-centered design principles  
- Real-time and offline signal processing workflows  

The system is designed for both:

- **Assistive use** (complete paralysis, EEG-only control)  
- **Rehabilitative use** (partial motor function, EEG + EMG/IMU integration)  

---

## System Overview

The architecture integrates neural signal processing with immersive VR feedback in a structured training loop.

### High-Level Pipeline

EEG acquisition → Preprocessing → Epoching → Feature Extraction → Classification → VR Feedback

The system adapts training difficulty based on decoding stability and classification accuracy across repetitions and sessions.

---

## Experimental Structure

Each training level follows a structured trial paradigm:

Rest → Preparation → Motor Imagery / Movement

Each level includes:

- **60 repetitions total**
- **Phase 1 (Offline – 30 repetitions):** data collection and subject-specific classifier training
- **Phase 2 (Online – 30 repetitions):** real-time decoding with contingent feedback

### Progression Rule

If real-time classification accuracy ≥ 85%, the next level is unlocked.  
Otherwise, the level is repeated with supportive feedback.

---

## Core Components

### 1. EEG-Based Motor Intention Decoding

- Motor imagery paradigm  
- Binary classification (Rest vs Movement)  
- Progressive multi-class decoding (e.g., Left vs Right)  
- Subject-specific calibration  
- CSP-based feature extraction (conceptual framework)  
- LDA-based classification (prototype stage)  

The decoding framework is designed to support extension toward more complex BCI paradigms.

---

### 2. Adaptive Multi-Level Training Logic

**Level 1**  
Binary decoding (Rest vs Movement)

**Level 2**  
Left vs Right motor imagery

**Higher Levels**  
Increased task complexity and reduced feedback support

The unlocking mechanism ensures gradual cognitive load increase and prevents frustration.

---

### 3. Multimodal Extension (Rehabilitative Pathway)

For users with residual motor function:

- EEG + EMG + IMU integration  
- Intention–execution consistency analysis  
- Motor output monitoring  
- Adaptive feedback based on coherence metrics  

This pathway shifts the focus from pure intention decoding to neuro-motor alignment.

---

### 4. Human-Centered Design

The system design was guided by:

- Semi-structured interviews  
- Persona development  
- User journey mapping  
- Cognitive load minimization principles  

Design principles include:

- Minimal interface complexity  
- Supportive, non-judgmental feedback  
- Clear session structure  
- Fatigue-aware stopping logic  
- Longitudinal progress tracking via monitoring application  

---

## Human-Centered Design & Figma Prototyping

The system architecture, user journeys, and interface wireframes were developed iteratively using Figma.

Interactive prototype: https://www.figma.com/design/4JW7qm4cgDYbretVs8oTp7/HMI-prototipo

The prototype includes:

- System architecture diagrams  
- Multi-level task flow  
- Offline/Online training logic  
- Monitoring application interface concept  
- User journey visualization  
- Feedback design iterations  

Selected screenshots are available in the `figures/` directory.

---

## Technical Architecture

### Two-Machine Setup

**Acquisition PC**
- EEG recording  
- LSL streaming  

**Processing / Interface PC**
- Task control (GUI)  
- Marker transmission via LSL  
- Offline model training  
- Real-time classification  
- Feedback rendering  

Synchronization is managed using Lab Streaming Layer (LSL).

---

## Validation Strategy

The framework includes multiple evaluation dimensions.

### Signal-Level Validation
- Within-session classification accuracy  
- Cross-session stability  
- Feature separability  

### System-Level Validation
- Real-time decoding reliability  
- Intention–execution coherence (EEG–EMG coupling)  
- Level progression stability  

### User-Level Validation
- Adherence  
- Fatigue monitoring  
- Engagement metrics  

---

## Repository Structure
<img width="310" height="488" alt="Screenshot 2026-02-18 alle 14 31 54" src="https://github.com/user-attachments/assets/c8154431-7456-4283-bbd4-6bad7a974701" />

---

## File Description

### `src/Level1HMI.py`

Prototype implementation of Level 1 training interface.

Responsibilities:

- Trial timing management  
- Rest / Preparation / Movement phase transitions  
- Visual cue presentation  
- LSL marker transmission  
- Repetition counting  
- Basic performance logic  

This script represents the task-control layer of the system and is designed to interface with an external EEG processing pipeline.

---

### `docs/Report_HMI.pdf`

Full project report including:

- Theoretical background  
- System rationale  
- User analysis  
- Personas  
- Task analysis  
- Ethical considerations  
- Evaluation framework  

---

### `docs/Slides_HMI.pdf`

Presentation slides containing:

- System architecture diagrams  
- Multi-level design visualization  
- Offline vs Online structure  
- EEG analysis overview  
- Evaluation schema  

---

### `docs/Personas_interview.md`

Interview guide used for:

- Persona creation  
- Identifying user needs  
- Defining cognitive constraints  
- Informing interface design  

---

### `figures/`

Contains visual material extracted from Figma and presentation slides:

- Architecture diagrams  
- Task flow diagrams  
- EEG processing pipeline  
- Interface screenshots  

These images support the conceptual understanding of the system.

---

## How to Run the Prototype

Requirements:

- Python 3.10+
- Required libraries listed in the imports of `Level1HMI.py`

To run:

python src/Level1HMI.py

Note: Real-time EEG classification requires an external LSL-compatible acquisition pipeline.

---

## Research Contribution

This project contributes to neurotechnology research by:

- Integrating EEG decoding within a structured rehabilitation framework  
- Embedding adaptive logic in BCI-based training  
- Combining signal processing and human-centered design  
- Addressing engagement and long-term adherence in neurorehabilitation  

The framework is designed to scale toward real-time BCI control and assistive device integration.

---

## Future Work

- Real-time adaptive classifier updating  
- Transfer learning for inter-subject generalization  
- Closed-loop assistive device control  
- Longitudinal clinical validation  
- Fatigue-aware adaptive session duration  
- Reinforcement learning for dynamic level adjustment  
