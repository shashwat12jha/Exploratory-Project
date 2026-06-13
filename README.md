<div align="center">

# 🚦 Automatic Video-Based Road Safety Analysis for Urban Streets
### Surrogate Safety-Based Risk Detection using MTTC & PET

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)](#)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

**A Proactive, Computer Vision-Based Traffic Safety Analysis System.**

[Overview](#overview) •
[Architecture](#architecture) •
[Metrics](#metrics) •
[Outputs](#outputs) •
[Tech Stack](#tech-stack)

</div>

---

<a id="overview"></a>
## 📌 Overview

This project presents a **computer vision-based traffic safety analysis system** that detects and quantifies *near-miss events* (traffic conflicts) from video footage.

Instead of relying on historical crash data—which is rare, delayed, and reactive—this system uses **Surrogate Safety Measures (SSMs)** to identify risky interactions in real time directly from normal traffic video.

### 🎯 Core Objective

To detect, classify, and spatially analyze **traffic conflicts** using research-backed metrics:
- 🔴 **MTTC (Modified Time to Collision)** → For Rear-end conflicts
- 🔵 **PET (Post Encroachment Time)** → For Crossing conflicts

---

## 🧠 Why This Matters

| Traditional Accident Analysis ❌ | This System ✅ |
| :--- | :--- |
| **Reactive:** Relies on crashes that have already happened. | **Proactive:** Identifies dangerous behavior *before* crashes occur. |
| **Sparse Data:** Crashes are statistically rare events. | **High-Frequency:** Generates rich safety insights from daily traffic. |
| **Manual:** Requires police reports and manual surveys. | **Automated:** Works on standard traffic video feeds. |

👉 **Ultimate Goal:** Enable early intervention in dangerous zones by identifying high-risk hotspots.

---

<a id="architecture"></a>
## 🏗️ System Architecture

```mermaid
flowchart TD
    A([Input Video]) --> B[YOLO Object Detection]
    B --> C[SORT Multi-Object Tracking]
    C --> D[Homography Pixel → World Mapping]
    D --> E[Trajectory Estimation EMA + Regression]
    E --> F[Pairwise Vehicle Interaction Analysis]
    F --> G[Closest Approach Computation]
    G --> H{Scenario Classification}
    H -->|Rear-End| I[Metric Evaluation: MTTC]
    H -->|Crossing| J[Metric Evaluation: PET]
    I --> K[Risk Classification]
    J --> K
    K --> L[Persistence Filtering]
    L --> M[Heatmap Generation + Logging]
    M --> N([Final Outputs Video + Excel + Heatmaps])
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef highlight fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    class H,I,J,K highlight;
```

---

## 🔄 Detailed Workflow

```mermaid
graph TD
    subgraph 1. Vision & Tracking
        V[🎥 Input Video Feed] --> Y[YOLO Detection <br/> 🚗 Vehicles / 🚶 Pedestrians]
        Y --> S[SORT Tracking <br/> 🆔 ID assignment]
        S --> H[Homography Transformation <br/> 🌍 Pixel → Real World meters]
        H --> TE[Trajectory Estimation <br/> 📈 EMA Velocity + Regression]
    end

    subgraph 2. Conflict Analysis
        TE --> PA[Pairwise Vehicle Analysis]
        PA --> CA[Closest Approach Calculation <br/> ⏱️ t_ca, d_ca, conflict point]
        CA --> SC{Scenario Classification}
        
        SC -- Rear-End <br/> Same Direction --> RE[MTTC + DRAC]
        SC -- Crossing <br/> Intersecting Paths --> CR[PET]
        
        RE --> RC[Risk Classification]
        CR --> RC
        
        RC --> PF[Persistence Filtering <br/> 🛡️ remove flickering alerts]
    end

    subgraph 3. Output Generation
        PF --> HM[Heatmap Accumulation]
        PF --> LG[Logging <br/> 📊 Excel + Summary]
        HM --> FO[🏁 Final Outputs <br/> Video + Heatmaps + Logs]
        LG --> FO
    end
```

---

## ⚙️ Core Components

### 🔍 1. Object Detection
* **Model:** YOLO (Ultralytics)
* **Detects:** Cars, Two-wheelers, Pedestrians

### 🧾 2. Multi-Object Tracking
* **Algorithm:** SORT (Simple Online and Realtime Tracking)
* **Function:** Assigns and maintains persistent IDs across video frames.

### 🌍 3. World Coordinate Mapping
* **Method:** Homography matrix transformation.
* **Function:** Transforms pixel coordinates into real-world meters, enabling **physics-based analysis**.

### 📈 4. Motion Estimation
* **Velocity:** Exponential Moving Average (EMA) for smooth speed estimation.
* **Direction:** Linear regression for stable trajectory prediction.
* **Acceleration:** Derived mathematically from velocity changes over time.

---

## ⚠️ Conflict Detection Logic

**Step 1: Distance Filtering**  
Ignore vehicles that are far away from each other to save compute.

**Step 2: Closest Approach**  
Compute Time to closest approach (TCA), minimum distance, and the exact spatial conflict point.

**Step 3: Scenario Classification**  

| Scenario | Condition | Safety Metric Used |
| :--- | :--- | :--- |
| **Rear-End** | Same direction of travel | TTC, DRAC, MTTC |
| **Crossing** | Intersecting paths | PET |

---

<a id="metrics"></a>
## 📊 Safety Metrics & Risk Classification

### 🔴 Rear-End Conflicts
* **TTC** → Time to Collision
* **DRAC** → Deceleration Rate to Avoid Crash
* **MTTC** → Modified Time to Collision (Accounts for relative acceleration)

| Risk Level | Condition |
| :--- | :--- |
| 🚨 **CRITICAL** | `TTC < 1.5s` **AND** `DRAC ≥ 3.4` **AND** `MTTC < 1.5s` |
| ⚠️ **WARNING** | Moderate risk parameters |
| ✅ **MONITOR** | Safe interactions |

### 🔵 Crossing Conflicts
* **PET (Post Encroachment Time)** = `|t1 - t2|` *(Where t1, t2 are arrival times at the conflict point)*

| Risk Level | Condition |
| :--- | :--- |
| 🚨 **CRITICAL** | `PET < 1.0s` |
| ⚠️ **WARNING** | `PET < 2.0s` |
| ✅ **MONITOR** | `PET ≥ 2.0s` |

---

## 🔥 Heatmap Generation

Visualizing spatial risk is crucial for urban planning.

* **🔴 MTTC Heatmap:** Only accumulates critical rear-end conflicts (`MTTC < 1.5s`).
* **🔵 PET Heatmap:** Only accumulates unsafe crossing interactions (`PET < 2.0s`). Based on thresholds defined by Allen et al. (1978) and FHWA.

---

<a id="outputs"></a>
## 📁 Outputs Structure

After execution, the system generates the following assets:

```text
/working_directory/
│
├── final.mp4                 # 🎥 Annotated video with bounding boxes & risk levels
├── collision_log.xlsx        # 📊 Detailed event log (IDs, TTC, PET, Risk)
├── heatmap_mttc_rearend.png  # 🔴 Spatial heatmap of rear-end conflicts
├── heatmap_pet_crossing.png  # 🔵 Spatial heatmap of crossing conflicts
└── heatmap_combined.png      # 🟣 Combined risk visualization
```

---

## 🧪 Key Strengths

- 🛡️ **Physics-Based Modeling:** Relies on real-world kinematics, not just opaque ML black-boxes.
- 🚦 **Scenario-Aware:** Dynamically applies the right metric (MTTC vs PET) based on the interaction type.
- 📉 **Noise Reduction:** Implements persistence filtering to prevent alert flickering.
- 🗺️ **Spatial Visualization:** Transforms abstract numbers into actionable geographic heatmaps.
- 📚 **Research-Backed:** Utilizes established safety thresholds from FHWA and AASHTO.

---

<a id="tech-stack"></a>
## 🛠️ Tech Stack

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black" alt="YOLO" />
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" />
  <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" alt="Matplotlib" />
</div>

* **Tracker:** SORT (Simple Online and Realtime Tracking)
* **Data Processing:** OpenPyXL (for Excel logging)

---

## 🚀 Future Work

- [ ] **Real-time deployment:** Direct CCTV stream integration.
- [ ] **Lane-wise conflict detection:** Granular analysis by lane topology.
- [ ] **AI Trajectory Prediction:** Implement LSTM / Transformer networks for better path forecasting.
- [ ] **Risk Normalization:** Adjust risk scores based on real-time traffic density.
- [ ] **Dashboard Visualization:** Interactive web-based analytics dashboard.

---

## 📚 References

1. **Allen, B. L., Shin, B. T., & Cooper, P. J. (1978)** - *Analysis of Traffic Conflicts and Collisions*
2. **Gettman, D., & Head, L. (2003)** - *Surrogate Safety Measures from Traffic Simulation Models*
3. **AASHTO** - *Design Guidelines for DRAC*

---

## 👨‍💻 Author

**Shashwat Kumar Jha**  
IIT (BHU), Varanasi  

---

## ⭐ Support

If you found this project useful or interesting:
- **Star** ⭐ the repository
- **Fork** 🍴 it to experiment
- **Build** 🚀 on top of it!
