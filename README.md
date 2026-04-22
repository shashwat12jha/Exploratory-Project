# 🚦 Automatic Video-Based Road Safety Analysis for Urban Streets using Conflict Technique

### Surrogate Safety-Based Risk Detection using MTTC & PET

---

## 📌 Overview

This project presents a **computer vision-based traffic safety analysis system** that detects and quantifies *near-miss events* (traffic conflicts) from video footage.

Instead of relying on crash data (which is rare and delayed), the system uses **Surrogate Safety Measures (SSMs)** to identify risky interactions in real time.

### 🎯 Core Objective

To detect, classify, and spatially analyze **traffic conflicts** using:

* **MTTC (Modified Time to Collision)** → Rear-end conflicts
* **PET (Post Encroachment Time)** → Crossing conflicts

---

## 🧠 Why This Matters

Traditional accident analysis:

* Reactive ❌
* Sparse data ❌

This system:

* Proactive ✅
* High-frequency safety insights ✅
* Works on normal traffic video ✅

👉 Enables **early intervention in dangerous zones**

---

## 🏗️ System Architecture

```
Input Video
     ↓
YOLO Object Detection
     ↓
SORT Multi-Object Tracking
     ↓
Homography (Pixel → World Mapping)
     ↓
Trajectory Estimation (EMA + Regression)
     ↓
Pairwise Vehicle Interaction Analysis
     ↓
Closest Approach Computation
     ↓
Scenario Classification
     ↓
Metric Evaluation (MTTC / PET)
     ↓
Risk Classification
     ↓
Persistence Filtering
     ↓
Heatmap Generation + Logging
     ↓
Final Outputs (Video + Excel + Heatmaps)
```

---

## 🔄 Detailed Workflow

```
               ┌──────────────────────┐
               │   Input Video Feed   │
               └─────────┬────────────┘
                         ↓
            ┌──────────────────────────┐
            │ YOLO Detection (Vehicles)│
            └─────────┬────────────────┘
                      ↓
            ┌──────────────────────────┐
            │ SORT Tracking (ID assign)│
            └─────────┬────────────────┘
                      ↓
        ┌──────────────────────────────┐
        │ Homography Transformation    │
        │ Pixel → Real World (meters)  │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Trajectory Estimation        │
        │ EMA Velocity + Regression    │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Pairwise Vehicle Analysis    │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Closest Approach Calculation │
        │ (t_ca, d_ca, conflict point) │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Scenario Classification      │
        │ Rear-End vs Crossing         │
        └───────┬───────────┬──────────┘
                ↓           ↓
        ┌──────────────┐ ┌──────────────┐
        │ Rear-End     │ │ Crossing     │
        │              │ │              │
        │ MTTC + DRAC  │ │ PET          │
        └──────┬───────┘ └──────┬───────┘
               ↓                ↓
        ┌──────────────────────────────┐
        │ Risk Classification          │
        │ CRITICAL / WARNING / MONITOR │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Persistence Filtering        │
        │ (remove flickering alerts)   │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Heatmap Accumulation         │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Logging (Excel + Summary)    │
        └─────────┬────────────────────┘
                  ↓
        ┌──────────────────────────────┐
        │ Final Outputs                │
        │ Video + Heatmaps + Logs      │
        └──────────────────────────────┘
```

---

## ⚙️ Core Components

### 🔍 1. Object Detection

* Model: YOLO (Ultralytics)
* Detects:

  * Cars
  * Two-wheelers
  * Pedestrians

---

### 🧾 2. Multi-Object Tracking

* Algorithm: SORT
* Assigns persistent IDs across frames

---

### 🌍 3. World Coordinate Mapping

* Homography transforms pixel → meters
* Enables **real-world physics-based analysis**

---

### 📈 4. Motion Estimation

* **Velocity** → Exponential Moving Average (EMA)
* **Direction** → Regression (stable trajectory)
* **Acceleration** → Derived from velocity

---

## ⚠️ Conflict Detection Logic

### Step 1: Distance Filtering

Ignore far-away vehicles

### Step 2: Closest Approach

Compute:

* Time to closest approach (TCA)
* Minimum distance
* Conflict point

### Step 3: Scenario Classification

| Scenario | Condition          |
| -------- | ------------------ |
| Rear-End | Same direction     |
| Crossing | Intersecting paths |

---

## 📊 Safety Metrics

### 🔴 Rear-End Conflicts

* **TTC** → Time to Collision
* **DRAC** → Required deceleration
* **MTTC** → Accounts for acceleration

---

### 🔵 Crossing Conflicts

* **PET (Post Encroachment Time)**

[
PET = |t_1 - t_2|
]

Where:

* ( t_1, t_2 ) = arrival times at conflict point

---

## 🚨 Risk Classification

### Rear-End

| Level    | Condition                                |
| -------- | ---------------------------------------- |
| CRITICAL | TTC < 1.5s AND DRAC ≥ 3.4 AND MTTC < 1.5 |
| WARNING  | Moderate risk                            |
| MONITOR  | Safe                                     |

---

### Crossing

| Level    | Condition   |
| -------- | ----------- |
| CRITICAL | PET < 1.0 s |
| WARNING  | PET < 2.0 s |
| MONITOR  | PET ≥ 2.0 s |

---

## 🔥 Heatmap Generation

### 🔴 MTTC Heatmap

* Only counts:

```
MTTC < 1.5 seconds
```

✔ Represents **critical rear-end conflicts**

---

### 🔵 PET Heatmap

* Only counts:

```
PET < 2.0 seconds
```

📚 Based on:

* Allen et al. (1978)
* FHWA Traffic Conflict Technique

✔ Represents **unsafe crossing interactions**

---

## 📁 Outputs

Generated after execution:

```
/kaggle/working/
│
├── final.mp4                         # Annotated video
├── collision_log.xlsx               # Event log
├── heatmap_mttc_rearend.png         # Rear-end conflicts
├── heatmap_pet_crossing.png         # Crossing conflicts
├── heatmap_combined.png             # Combined view
```

---

## 📊 Excel Log Structure

Includes:

* Vehicle IDs
* Scenario type
* TTC, DRAC, MTTC, PET
* Risk level
* Timestamp

Also provides:

* Pair-wise summary
* Peak risk detection

---

## 🧪 Key Strengths

* Physics-based modeling (not just ML)
* Scenario-aware risk computation
* Noise reduction via persistence filtering
* Spatial risk visualization
* Research-backed thresholds

---

## 📚 References

* Allen, B. L., Shin, B. T., & Cooper, P. J. (1978)
  *Analysis of Traffic Conflicts and Collisions*

* Gettman, D., & Head, L. (2003)
  *Surrogate Safety Measures from Traffic Simulation Models*

* AASHTO
  *Design Guidelines for DRAC*

---

## 🚀 Future Work

* Real-time deployment (CCTV integration)
* Lane-wise conflict detection
* AI-based trajectory prediction (LSTM / Transformer)
* Risk normalization by traffic density
* Dashboard visualization

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLO (Ultralytics)
* SORT Tracker
* NumPy
* Matplotlib
* OpenPyXL

---

## 👨‍💻 Author

**Shashwat Kumar Jha**
IIT (BHU), Varanasi

---

## ⭐ Support

If you found this useful:

* Star ⭐ the repo
* Fork 🍴 it
* Build on it 🚀
