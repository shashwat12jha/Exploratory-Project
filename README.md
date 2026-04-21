# 🚀 Predictive Traffic Collision Detection System

A computer vision + physics-based system for **multi-class traffic monitoring and collision prediction** using real-world video data.

---

## 📌 Overview

This project detects, tracks, and analyzes traffic participants in a video feed and predicts **potential collisions before they occur** using **time-to-collision (TTC)** and trajectory modeling.

Unlike traditional systems that rely on proximity, this system is **predictive**, leveraging velocity and motion patterns to estimate future interactions.

---

## 🎯 Key Features

* 🚗 **Multi-class Detection & Tracking**

  * Car (0)
  * Person (1)
  * 2-Wheeler (2)

* 📍 **Homography-based World Mapping**

  * Converts pixel coordinates → real-world coordinates (meters)

* 🧠 **Velocity Estimation with EMA Smoothing**

  * Stable and realistic motion tracking

* 🔮 **Future Trajectory Prediction**

  * Visualizes where each object will move

* ⚠️ **Collision Prediction using Time-to-Collision (TTC)**

  * Predicts future collisions based on relative motion

* 🚦 **Advanced Collision Filtering**

  * Lateral + longitudinal thresholds
  * Avoids false positives (parallel vehicles)

* 📊 **Live Analytics**

  * Vehicle count
  * Speed estimation (km/h)

---
## 🔄 Pipeline Flowchart

```mermaid
flowchart TD

A[Input Video] --> B[ROI Masking]
B --> C[YOLO Detection]

C --> D[Class Filtering<br>(Car / Person / 2W)]
D --> E[SORT Tracking]

E --> F[Pixel → World Mapping<br>(Homography)]
F --> G[Trajectory History]

G --> H[Velocity Estimation<br>(EMA Smoothing)]
H --> I[Speed Calculation]

I --> J[Future Trajectory Prediction]

J --> K[Collision Detection<br>(TTC + Relative Motion)]
K --> L[Filtering<br>(Lateral + Longitudinal)]

L --> M[Visualization]
M --> N[Output Video]
---

## 🧠 Core Idea

We model each vehicle using:

* Position: ( p )
* Velocity: ( v )

Then compute:

```
t = - (rel_p ⋅ rel_v) / (rel_v ⋅ rel_v)
```

Where:

* `rel_p = p2 - p1`
* `rel_v = v2 - v1`

This gives the **time at which two objects are closest** → used for collision prediction.

---

## 🏗️ Tech Stack

* **Python**
* **OpenCV**
* **NumPy**
* **Ultralytics YOLO (custom trained model)**
* **SORT Tracker**
* **Homography (projective geometry)**

---

## 📂 Project Structure

```
.
├── main.py                # Main pipeline
├── sort/                 # SORT tracking implementation
├── model/
│   └── best.pt          # Trained YOLO model
├── input/
│   └── video.mp4        # Input traffic video
├── output/
│   └── final_output.mp4 # Output processed video
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

2. Update paths in the script:

* YOLO model path
* Input video path

3. Run:

```bash
python main.py
```

4. Output video will be saved in:

```
/kaggle/working/final_output.mp4
```

---

## 📸 Output Highlights

* Bounding boxes with:

  * Class labels
  * Speed (km/h)
  * Track ID

* Trajectories:

  * Past (smoothed)
  * Future (predicted)

* Collision visualization:

  * Predicted intersection point
  * Highlighted interaction paths

---

## ⚙️ Important Parameters

| Parameter              | Description                     |
| ---------------------- | ------------------------------- |
| `ALPHA`                | Velocity smoothing factor (EMA) |
| `dt`                   | Frame time interval             |
| `TTC threshold`        | Max prediction window (5 sec)   |
| Lateral threshold      | X-axis distance filter          |
| Longitudinal threshold | Y-axis distance filter          |

---

## 🚀 Future Improvements

* 🔥 Collision severity classification (vehicle vs pedestrian)
* 📊 Traffic heatmaps
* 🧠 DeepSORT / ReID for better tracking
* 🌐 Multi-camera integration
* 📈 Real-time dashboard

---

## 🏆 Applications

* Smart traffic monitoring systems
* Autonomous driving safety modules
* Accident prevention systems
* Urban traffic analytics

---

## 💡 Key Insight

> This system predicts **where vehicles WILL be**, not just where they ARE.

---

## 🙌 Acknowledgements

* Ultralytics YOLO
* SORT Tracking Algorithm
* OpenCV community

---

## 📬 Author

**Shashwat Kumar Jha**
IIT BHU

---

## ⭐ If you like this project

Give it a star ⭐ — it helps a lot!
