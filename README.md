# Intelligent Driver Drowsiness & Safety Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Overview
The Intelligent Driver Drowsiness & Safety Detection System is an end-to-end Machine Learning and Computer Vision pipeline designed to prevent fatigue-related accidents. The system monitors a driver's facial landmarks in real-time, extracts temporal features (Eye Aspect Ratio and Mouth Aspect Ratio), and uses a trained Scikit-Learn classifier with temporal smoothing logic to detect sustained drowsiness.

## 2. Problem Statement & Motivation
Driver fatigue is a leading cause of severe traffic accidents. Microsleeps and delayed reaction times associated with drowsiness can be fatal. This project was developed as a professional engineering demonstration for Toyota Connected India, showcasing a complete ML engineering lifecycle: from rigorous data preprocessing (avoiding temporal data leakage) to feature engineering, model evaluation, and edge-deployable real-time inference.

## 3. Key Technical Features
- **Real-Time Landmark Detection:** Uses MediaPipe Face Mesh for CPU-efficient, 468-point facial landmark extraction without the heavy C++ build requirements of `dlib`.
- **Robust Feature Engineering:** Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) mathematically using Euclidean distances.
- **Temporal Sequence Modeling:** Applies rolling windows to compute rolling means and standard deviations, transforming static frame data into time-series features.
- **Data Leakage Prevention:** Uses `GroupShuffleSplit` on `session_id` to ensure rolling windows from the training set do not leak into the test set.
- **Temporal Alert Logic:** Prevents false positives (e.g., from normal blinking) by requiring 20 consecutive frames of positive drowsiness predictions before triggering an alarm.

## 4. System Architecture
```mermaid
flowchart TD
    A[Camera / Video Stream] --> B[Frame Processing (OpenCV)]
    B --> C[Face Mesh Detection (MediaPipe)]
    C --> D[Feature Extraction: EAR & MAR]
    D --> E[Temporal Features: Rolling Mean/Std]
    E --> F[ML Classifier (Random Forest / LR)]
    F --> G{Consecutive Drowsy Frames >= 20?}
    G -- Yes --> H[Trigger ALERT]
    G -- No --> I[NORMAL State]
```

## 5. Dataset & Methodology
Due to the constraints of hosting massive video datasets (like UTA-RLDD or YawDD) in a standard GitHub repository, this project utilizes a **synthetic tabular dataset generator** (`src/generate_sample_data.py`). 

- The script generates 10,000 frames across 50 distinct sessions, simulating realistic EAR and MAR behaviors for both "Alert" (normal blinks) and "Drowsy" (long closures, drooping eyelids, yawning) states.
- During training, the data is split using `GroupShuffleSplit` grouped by `session_id`. This strictly isolates video sessions, guaranteeing that the model evaluates on strictly unseen behavioral sequences.
- For a production deployment, developers can run the provided `vision_utils.py` against standard video datasets to generate a real-world `features.csv`.

## 6. Actual Evaluation Results
The pipeline evaluates baseline models to map the rolling features to the target label. Because the synthetic dataset generates distinct linear distributions for behavioral states, the models achieve mathematically perfect separation on the test set.

- **Logistic Regression:** Accuracy: 100.00%, F1-Score: 1.0000
- **Random Forest:** Accuracy: 99.90%, F1-Score: 0.9992
- **Support Vector Machine (SVM):** Accuracy: 100.00%, F1-Score: 1.0000

*Note: In real-world human data, accuracy will naturally be lower due to genetic variance in eye shape and lighting conditions. Recall is prioritized in this pipeline to ensure minimal false negatives.*

## 7. Project Structure
```text
driver-drowsiness-detection/
├── data/
│   ├── processed/
│   │   └── features.csv (Generated)
├── models/
│   ├── drowsiness_model.pkl
│   └── scaler.pkl
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── alert_manager.py
│   ├── config.py
│   ├── evaluate_model.py
│   ├── feature_extraction.py
│   ├── generate_sample_data.py
│   ├── model_utils.py
│   ├── realtime_detection.py
│   └── vision_utils.py
├── tests/
├── requirements.txt
└── README.md
```

## 8. Installation & Usage

### Prerequisites
- Python 3.10+
- Webcam

### Setup
```bash
git clone <your-repo-url>
cd driver-drowsiness-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Execution Pipeline
```bash
# 1. Generate the session-based dataset
python src/generate_sample_data.py

# 2. Train models and save the best one
python src/train_model.py

# 3. Generate evaluation visualizations (Confusion Matrix, etc.)
python src/evaluate_model.py

# 4. Run the real-time detection application
python src/realtime_detection.py
```

### Running Tests
The project includes automated Pytest coverage for feature calculations, temporal logic, and mathematics.
```bash
pytest tests/
```

## 9. Known Limitations
- **Hardware/Lighting:** MediaPipe may struggle in extreme low light, heavy backlight, or when the driver wears thick occlusions (like sunglasses).
- **Synthetic Baseline:** The current pre-trained weights (`models/`) are based on synthetic distributions. The pipeline MUST be retrained on real-world datasets (e.g., UTA-RLDD) before physical deployment.

## 10. Future Improvements
- **Head Pose Estimation:** Incorporate Pitch and Yaw calculations to detect distracted driving (looking away from the road).
- **Sequential Neural Networks:** Transition from manual rolling statistics to an LSTM or GRU to natively capture sequence dependencies.
- **Audio Integration:** Add a hardware or software buzzer (`pygame`) for the alert state.

## 11. Safety Disclaimer
**This project is a portfolio prototype and research demonstration.** It is NOT certified for automotive safety or medical diagnosis. Do not rely on this software as a primary safety mechanism while operating a vehicle.
