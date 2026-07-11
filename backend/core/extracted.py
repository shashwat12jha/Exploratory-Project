# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install ultralytics opencv-python-headless

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model (you can later replace with your trained weights)
model = YOLO("/kaggle/input/datasets/shashwatkumarjha/weightstrainedfinal/best.pt", task='detect')

# Video path
video_path = "/kaggle/input/datasets/shashwatkumarjha/newtrimvideo/ChotiClip.mp4"

cap = cv2.VideoCapture(video_path)

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("Actual FPS:", fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Resolution:", width, "x", height)

print(model.names)

import math

def is_valid(prev, curr, max_dist=60):
    if prev is None:
        return True
    dist = math.hypot(curr[0]-prev[0], curr[1]-prev[1])
    return dist < max_dist


def smooth(points, k=5):
    if len(points) < k:
        return points
    
    smoothed = []
    for i in range(len(points)):
        pts = points[max(0, i-k):i+1]
        x_avg = sum(p[0] for p in pts) / len(pts)
        y_avg = sum(p[1] for p in pts) / len(pts)
        smoothed.append((int(x_avg), int(y_avg)))
    
    return smoothed

roi_points = np.array([
    [540,1066],
    [972,216],
    [1080,216],
    [1650,1066]
], dtype=np.int32)

def apply_roi_mask(frame):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_points], (255,255,255))
    masked = cv2.bitwise_and(frame, mask)
    return masked

import numpy as np
import cv2

# Your ROI points (same as masking)
src = np.array([
    [540,1066],
    [1650,1066],
    [1080,216],
    [972,216]
], dtype=np.float32)

# Real world coordinates (meters)
# width = 3.5m, length ≈ 20m
dst = np.array([
    [0, 20],
    [3.5, 20],
    [3.5, 0],
    [0, 0]
], dtype=np.float32)

# Compute homography
H, _ = cv2.findHomography(src, dst)

print("Homography matrix:\n", H)

def pixel_to_world(x, y):
    pt = np.array([x, y, 1.0])  # 1D array (important)
    world = H @ pt              # matrix multiply

    world = world / world[2]    # normalize

    return float(world[0]), float(world[1])

world_history = defaultdict(list)
speed_dict = {}

import cv2
import numpy as np
from collections import defaultdict
import math
from IPython.display import FileLink, display

# =========================
# CONFIG
# =========================
VEHICLE_CLASSES = [0, 2]
MAX_HISTORY = 30

# =========================
# COLORS
# =========================
COLOR_CAR = (0, 255, 255)
COLOR_2W = (255, 255, 0)
COLOR_TRAJ = (255, 0, 0)
COLOR_LINE = (0, 0, 255)

# =========================
# ROI
# =========================
roi_points = np.array([
    [540,1066],
    [972,216],
    [1080,216],
    [1650,1066]
], dtype=np.int32)

def apply_roi_mask(frame):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_points], (255,255,255))
    return cv2.bitwise_and(frame, mask)

# =========================
# TRAJECTORY
# =========================
def is_valid(prev, curr, max_dist=60):
    if prev is None:
        return True
    return math.hypot(curr[0]-prev[0], curr[1]-prev[1]) < max_dist

def smooth(points, k=5):
    if len(points) < k:
        return points
    smoothed = []
    for i in range(len(points)):
        pts = points[max(0,i-k):i+1]
        smoothed.append((
            int(sum(p[0] for p in pts)/len(pts)),
            int(sum(p[1] for p in pts)/len(pts))
        ))
    return smoothed

# =========================
# WORLD CONVERSION
# =========================
def pixel_to_world(x, y):
    pt = np.array([x, y, 1.0])
    world = H @ pt
    world = world / world[2]
    return float(world[0]), float(world[1])

# =========================
# STORAGE
# =========================
track_history = defaultdict(list)
world_history = defaultdict(list)
speed_display = {}

# =========================
# SINGLE SPEED LINE
# =========================
speed_line_y = 700   # 🔥 adjust if needed

# =========================
# VIDEO IO
# =========================
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps < 60:
    fps = 100

print("Using FPS:", fps)

output_path = "/kaggle/working/output_final.mp4"

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    masked = apply_roi_mask(frame)

    # Draw speed line
    cv2.line(frame, (0, speed_line_y), (width, speed_line_y), COLOR_LINE, 3)

    results = model.track(
        source=masked,
        conf=0.2,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True
    )[0]

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls in zip(boxes, ids, classes):

            if cls not in VEHICLE_CLASSES:
                continue

            x1,y1,x2,y2 = map(int, box)
            cx = int((x1+x2)/2)
            cy = int(y2)

            color = COLOR_CAR if cls==0 else COLOR_2W

            # ================= WORLD TRACK =================
            wx, wy = pixel_to_world(cx, cy)
            world_history[tid].append((wx, wy))

            # ================= SPEED LOGIC =================
            if len(world_history[tid]) >= 6:

                # only compute when near line (best detection zone)
                if abs(cy - speed_line_y) < 50:

                    x_prev, y_prev = world_history[tid][-6]
                    x_curr, y_curr = world_history[tid][-1]

                    dist = ((x_curr-x_prev)**2 + (y_curr-y_prev)**2)**0.5

                    dt = 5 / fps
                    speed = (dist / dt) * 3.6

                    # smooth update
                    prev_spd = speed_display.get(tid, speed)

                    if abs(speed - prev_spd) > 3:
                        speed_display[tid] = speed
                    else:
                        speed_display[tid] = prev_spd

            # ================= TRAJECTORY =================
            prev = track_history[tid][-1] if track_history[tid] else None
            if is_valid(prev, (cx, cy)):
                track_history[tid].append((cx, cy))

            if len(track_history[tid]) > MAX_HISTORY:
                track_history[tid].pop(0)

            pts = smooth(track_history[tid])
            for i in range(1,len(pts)):
                cv2.line(frame, pts[i-1], pts[i], COLOR_TRAJ, 2)

            # ================= DRAW =================
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID {tid}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            spd = speed_display.get(tid,0)
            cv2.putText(frame,f"{spd:.1f} km/h",(x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    out.write(frame)

cap.release()
out.release()

print("✅ FINAL SPEED WORKING (ROBUST METHOD)")
display(FileLink(output_path))

import numpy as np
import cv2

# ---- ROI points (image) ----
img_pts = np.array([
    [540,1066],   # A (bottom-left)
    [1650,1066],  # D (bottom-right)
    [972,216],    # B (top-left)
    [1080,216]    # C (top-right)
], dtype=np.float32)

# ---- Real world points (meters) ----
depth = 74.65
width = 3.0

world_pts = np.array([
    [0,0],
    [width,0],
    [0,depth],
    [width,depth]
], dtype=np.float32)

# ---- Homography ----
H, _ = cv2.findHomography(img_pts, world_pts)

print("Homography Matrix:\n", H)

def pixel_to_world(u, v, H):
    pt = np.array([u, v, 1]).reshape(3,1)
    world = H @ pt
    world /= world[2]
    return world[0][0], world[1][0]

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    u = (x1 + x2) / 2
    v = y2   # bottom
    return u, v

FPS = 100
dt = 1 / FPS

def compute_speed(Y1, Y2):
    speed_mps = abs(Y2 - Y1) / dt
    speed_kmph = speed_mps * 3.6
    return speed_mps, speed_kmph

pip install ultralytics

!git clone https://github.com/abewley/sort.git

import sys
sys.path.append("./sort")

import fileinput

file_path = "/kaggle/working/sort/sort.py"

for line in fileinput.input(file_path, inplace=True):
    if "TkAgg" in line:
        print("matplotlib.use('Agg')")
    else:
        print(line, end="")

!pip install filterpy

!rm -rf sort
!git clone https://github.com/abewley/sort.git

import sys
sys.path.insert(0, "/kaggle/working/sort")

import fileinput

file_path = "/kaggle/working/sort/sort.py"

for line in fileinput.input(file_path, inplace=True):
    if "TkAgg" in line:
        print("matplotlib.use('Agg')")
    else:
        print(line, end="")

!rm -rf sort

!git clone https://github.com/abewley/sort.git

import sys
import os

sys.path.insert(0, "/kaggle/working/sort")

# DEBUG: check file exists
print(os.listdir("/kaggle/working/sort"))

import fileinput

file_path = "/kaggle/working/sort/sort.py"

for line in fileinput.input(file_path, inplace=True):
    if "TkAgg" in line:
        print("matplotlib.use('Agg')")
    else:
        print(line, end="")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "sort_module", "/kaggle/working/sort/sort.py"
)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)

Sort = sort_module.Sort

print("SORT working ✅")

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ---- FIXED SORT IMPORT ----
import importlib.util

spec = importlib.util.spec_from_file_location(
    "sort_module", "/kaggle/working/sort/sort.py"
)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)

Sort = sort_module.Sort

# ---- LOAD MODEL (LIGHTWEIGHT) ----
model = YOLO("/kaggle/input/datasets/shashwatkumarjha/weightstrainedfinal/best.pt")   # use nano model for stability

# ---- VIDEO ----
cap = cv2.VideoCapture("/kaggle/input/datasets/shashwatkumarjha/newtrimvideo/ChotiClip.mp4")

# ---- OUTPUT VIDEO ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
input_fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("outputnew.mp4", fourcc, input_fps, (960,540))
# ---- PARAMETERS ----
FPS = 100
dt = 1 / FPS

# ---- INIT TRACKER ----
tracker = Sort()

# ---- STORAGE ----
track_history = defaultdict(lambda: deque(maxlen=5))
track_speed = defaultdict(float)

# ---- SCALE ROI (since we resize frame) ----
scale_x = 960 / 1920
scale_y = 540 / 1080

img_pts = np.array([
    [540*scale_x,1066*scale_y],
    [1650*scale_x,1066*scale_y],
    [972*scale_x,216*scale_y],
    [1080*scale_x,216*scale_y]
], dtype=np.float32)

depth = 74.65
width = 3.0

world_pts = np.array([
    [0,0],
    [width,0],
    [0,depth],
    [width,depth]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

# ---- FUNCTIONS ----

def pixel_to_world(u, v):
    pt = np.array([u, v, 1]).reshape(3,1)
    world = H @ pt
    world /= world[2]
    return world[0][0], world[1][0]

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2

def compute_smooth_speed(track_id, Y):
    history = track_history[track_id]
    history.append(Y)

    if len(history) < 5:
        return 0

    dy = history[-1] - history[0]
    dt_total = (len(history) - 1) * dt

    speed_mps = abs(dy) / dt_total
    speed_kmph = speed_mps * 3.6

    # smoothing
    speed_kmph = 0.7 * track_speed[track_id] + 0.3 * speed_kmph
    track_speed[track_id] = speed_kmph

    return speed_kmph

# ---- PROCESS VIDEO ----
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # resize (CRITICAL for stability)
    frame = cv2.resize(frame, (960, 540))

    
    mttc_list = compute_mttc_advanced(track_positions)
    # ---- DETECTION ----
    results = model(frame, imgsz=640, conf=0.4, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # only vehicles
        if cls not in [0, 2]:
            continue

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, conf])

    # handle empty
    if len(detections) == 0:
        dets = np.empty((0,5))
    else:
        dets = np.array(detections)

    # ---- TRACKING ----
    tracks = tracker.update(dets)

    # ---- PROCESS TRACKS ----
    for trk in tracks:
        x1, y1, x2, y2, track_id = trk
        track_id = int(track_id)

        # bottom center
        u, v = get_bottom_center((x1, y1, x2, y2))

        # ignore far detections
        if v < 120:
            continue

        # pixel → world
        X, Y = pixel_to_world(u, v)

        # speed
        speed = compute_smooth_speed(track_id, Y)

        # draw box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        # draw speed
        cv2.putText(frame,
                    f"ID:{track_id} {speed:.1f} km/h",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2)

    # ---- SAVE FRAME ----
    out.write(frame)

# ---- CLEANUP ----
cap.release()
out.release()

print("✅ Done! Video saved as output.mp4")

#adding storage
track_history = defaultdict(lambda: deque(maxlen=5))
track_speed = defaultdict(float)

track_prev_speed = defaultdict(float)
track_acc = defaultdict(float)


track_positions = {}   # ✅ ADD THIS

#computing acceleration and speed
prev_speed = track_prev_speed[track_id]

# convert km/h → m/s
speed_mps = speed / 3.6
prev_speed_mps = prev_speed / 3.6

acc = (speed_mps - prev_speed_mps) / dt

track_acc[track_id] = acc
track_prev_speed[track_id] = speed

#Update position storage
track_positions[track_id] = {
    "X": X,
    "Y": Y,
    "speed": speed,      # km/h
    "acc": acc           # m/s²
}

import math

def compute_mttc_advanced(track_positions):
    results = []
    ids = list(track_positions.keys())

    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue

            id1 = ids[i]
            id2 = ids[j]

            p1 = track_positions[id1]
            p2 = track_positions[id2]

            # lane filter
            if abs(p1["X"] - p2["X"]) > 1.5:
                continue

            # identify rear/front
            if p1["Y"] < p2["Y"]:
                rear, front = p1, p2
                rear_id, front_id = id1, id2
            else:
                rear, front = p2, p1
                rear_id, front_id = id2, id1

            V_r = rear["speed"] / 3.6
            V_f = front["speed"] / 3.6

            a_r = rear["acc"]
            a_f = front["acc"]

            delta_v = V_r - V_f
            delta_a = a_r - a_f
            S = abs(front["Y"] - rear["Y"])

            # avoid division by zero
            if abs(delta_a) < 1e-3:
                if delta_v > 0:
                    mttc = S / delta_v
                    results.append((rear_id, front_id, mttc))
                continue

            discriminant = delta_v**2 + 2 * delta_a * S

            if discriminant < 0:
                continue

            sqrt_term = math.sqrt(discriminant)

            t1 = (delta_v + sqrt_term) / delta_a
            t2 = (delta_v - sqrt_term) / delta_a

            # apply paper rule
            valid_times = [t for t in [t1, t2] if t > 0]

            if len(valid_times) == 0:
                continue

            mttc = min(valid_times)

            results.append((rear_id, front_id, mttc))

    return results

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import importlib.util
import math

# ---- LOAD SORT (SAFE IMPORT) ----
spec = importlib.util.spec_from_file_location(
    "sort_module", "/kaggle/working/sort/sort.py"
)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)
Sort = sort_module.Sort

# ---- LOAD MODEL ----
model = YOLO("/kaggle/input/datasets/shashwatkumarjha/weightstrainedfinal/best.pt")

# ---- VIDEO ----
cap = cv2.VideoCapture("/kaggle/input/datasets/shashwatkumarjha/newtrimvideo/ChotiClip.mp4")
input_fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputmttc.mp4", fourcc, input_fps, (960,540))

# ---- PARAMETERS ----
FPS = input_fps
dt = 1 / FPS

tracker = Sort()

# ---- STORAGE ----
track_history = defaultdict(lambda: deque(maxlen=5))
track_speed = defaultdict(float)
track_prev_speed = defaultdict(float)
track_acc = defaultdict(float)

# ---- CSV STORAGE ----
csv_data = []

# ---- HOMOGRAPHY ----
scale_x = 960 / 1920
scale_y = 540 / 1080

img_pts = np.array([
    [540*scale_x,1066*scale_y],
    [1650*scale_x,1066*scale_y],
    [972*scale_x,216*scale_y],
    [1080*scale_x,216*scale_y]
], dtype=np.float32)

depth = 74.65
width = 3.0

world_pts = np.array([
    [0,0],
    [width,0],
    [0,depth],
    [width,depth]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

# ---- FUNCTIONS ----

def pixel_to_world(u, v):
    pt = np.array([u, v, 1]).reshape(3,1)
    world = H @ pt
    world /= world[2]
    return world[0][0], world[1][0]

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2

def compute_smooth_speed(track_id, Y):
    history = track_history[track_id]
    history.append(Y)

    if len(history) < 5:
        return 0

    dy = history[-1] - history[0]
    dt_total = (len(history) - 1) * dt

    speed_mps = abs(dy) / dt_total
    speed_kmph = speed_mps * 3.6

    speed_kmph = 0.7 * track_speed[track_id] + 0.3 * speed_kmph
    track_speed[track_id] = speed_kmph

    return speed_kmph

def compute_mttc(track_positions):
    results = []
    ids = list(track_positions.keys())

    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue

            id1, id2 = ids[i], ids[j]
            p1, p2 = track_positions[id1], track_positions[id2]

            if abs(p1["X"] - p2["X"]) > 1.5:
                continue

            if p1["Y"] < p2["Y"]:
                rear, front = p1, p2
                rear_id, front_id = id1, id2
            else:
                rear, front = p2, p1
                rear_id, front_id = id2, id1

            Vr = rear["speed"] / 3.6
            Vf = front["speed"] / 3.6
            ar = rear["acc"]
            af = front["acc"]

            dv = Vr - Vf
            da = ar - af
            S = abs(front["Y"] - rear["Y"])

            if abs(da) < 1e-3:
                if dv > 0:
                    results.append((rear_id, front_id, S / dv))
                continue

            disc = dv**2 + 2 * da * S
            if disc < 0:
                continue

            sqrt_term = math.sqrt(disc)
            t1 = (dv + sqrt_term) / da
            t2 = (dv - sqrt_term) / da

            valid = [t for t in [t1, t2] if t > 0]
            if valid:
                results.append((rear_id, front_id, min(valid)))

    return results

# ---- MAIN LOOP ----
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960,540))
    track_positions = {}   # reset each frame

    results = model(frame, imgsz=640, conf=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls not in [0,2] or conf < 0.5:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, conf])

    dets = np.array(detections) if detections else np.empty((0,5))
    tracks = tracker.update(dets)

    for trk in tracks:
        x1, y1, x2, y2, track_id = trk
        track_id = int(track_id)

        u, v = get_bottom_center((x1, y1, x2, y2))
        if v < 120:
            continue

        X, Y = pixel_to_world(u, v)
        speed = compute_smooth_speed(track_id, Y)

        # ---- ACCELERATION ----
        prev_speed = track_prev_speed[track_id]
        speed_mps = speed / 3.6
        prev_speed_mps = prev_speed / 3.6

        acc = (speed_mps - prev_speed_mps) / dt
        acc = 0.7 * track_acc[track_id] + 0.3 * acc  # smoothing

        track_acc[track_id] = acc
        track_prev_speed[track_id] = speed

        track_positions[track_id] = {
            "X": X, "Y": Y, "speed": speed, "acc": acc
        }

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID:{track_id} {speed:.1f} km/h",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    # ---- MTTC ----
    mttc_list = compute_mttc(track_positions)

    for rear_id, front_id, mttc in mttc_list:
        if mttc < 5:
            color = (0,0,255) if mttc < 1.5 else (0,165,255)
            text = f"MTTC:{mttc:.2f}s ({rear_id}->{front_id})"

            cv2.putText(frame, text,
                        (50, 50 + 30*rear_id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- SAVE CSV ----
        csv_data.append({
            "frame": frame_count,
            "rear_id": rear_id,
            "front_id": front_id,
            "MTTC": mttc
        })

    out.write(frame)
    frame_count += 1

# ---- SAVE CSV ----
df = pd.DataFrame(csv_data)
df.to_csv("mttc_results.csv", index=False)

cap.release()
out.release()

print("✅ DONE: Video + CSV saved")

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import importlib.util
import math
import time

# ---- LOAD SORT ----
spec = importlib.util.spec_from_file_location(
    "sort_module", "/kaggle/working/sort/sort.py"
)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)
Sort = sort_module.Sort

# ---- MODEL ----
model = YOLO("/kaggle/input/datasets/shashwatkumarjha/weightstrainedfinal/best.pt")

# ---- VIDEO ----
cap = cv2.VideoCapture("/kaggle/input/datasets/shashwatkumarjha/newtrimvideo/ChotiClip.mp4")

# ---- FPS (your video is 100 FPS) ----
FPS = 100
dt = 1 / FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputnew3mttc.mp4", fourcc, FPS, (960,540))

tracker = Sort()

# ---- STORAGE ----
track_history = defaultdict(lambda: deque(maxlen=10))  # 🔥 smoother speed
track_speed = defaultdict(float)
track_prev_speed = defaultdict(float)
track_acc = defaultdict(float)

csv_data = []

# ---- HOMOGRAPHY ----
scale_x = 960 / 1920
scale_y = 540 / 1080

img_pts = np.array([
    [540*scale_x,1066*scale_y],
    [1650*scale_x,1066*scale_y],
    [972*scale_x,216*scale_y],
    [1080*scale_x,216*scale_y]
], dtype=np.float32)

depth = 74.65
width = 3.0

world_pts = np.array([
    [0,0],[width,0],[0,depth],[width,depth]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

# ---- FUNCTIONS ----

def pixel_to_world(u, v):
    pt = np.array([u, v, 1]).reshape(3,1)
    world = H @ pt
    world /= world[2]
    return world[0][0], world[1][0]

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2

def compute_speed(track_id, Y):
    hist = track_history[track_id]
    hist.append(Y)

    if len(hist) < 5:
        return 0

    dy = hist[-1] - hist[0]
    dt_total = (len(hist)-1)*dt

    speed = abs(dy)/dt_total * 3.6
    speed = 0.7*track_speed[track_id] + 0.3*speed
    track_speed[track_id] = speed

    return speed

# ---- TIMING CONTROL ----
start_time = time.time()
frame_index = 0

# ---- MAIN ----
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame.copy()  # 🔥 prevents decoding jitter
    frame = cv2.resize(frame, (960,540))

    track_positions = {}

    # ---- DETECTION ----
    results = model(frame, imgsz=640, conf=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls not in [0,2] or conf < 0.5:
            continue

        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        detections.append([x1,y1,x2,y2,conf])

    dets = np.array(detections) if detections else np.empty((0,5))
    tracks = tracker.update(dets)

    # ---- PROCESS TRACKS ----
    for trk in tracks:
        x1,y1,x2,y2,track_id = trk
        track_id = int(track_id)

        u,v = get_bottom_center((x1,y1,x2,y2))
        if v < 120:
            continue

        X,Y = pixel_to_world(u,v)

        speed = compute_speed(track_id, Y)

        prev_speed = track_prev_speed[track_id]
        acc = ((speed - prev_speed)/3.6)/dt
        acc = 0.7*track_acc[track_id] + 0.3*acc

        track_prev_speed[track_id] = speed
        track_acc[track_id] = acc

        track_positions[track_id] = {
            "X":X,"Y":Y,"speed":speed,"acc":acc
        }

        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,f"{speed:.1f} km/h",(int(x1),int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    # ---- MTTC ----
    vehicles = sorted(track_positions.items(), key=lambda x: x[1]["Y"])

    for i in range(len(vehicles)-1):
        rear_id, rear = vehicles[i]
        front_id, front = vehicles[i+1]

        if abs(rear["X"] - front["X"]) > 1.5:
            continue

        Vr = rear["speed"]/3.6
        Vf = front["speed"]/3.6
        ar = rear["acc"]
        af = front["acc"]

        dv = Vr - Vf
        da = ar - af
        S = abs(front["Y"] - rear["Y"])

        if dv <= 0:
            continue

        if abs(da) < 1e-3:
            mttc = S/dv
        else:
            disc = dv**2 + 2*da*S
            if disc < 0:
                continue
            t1 = (dv + math.sqrt(disc))/da
            t2 = (dv - math.sqrt(disc))/da
            valid = [t for t in [t1,t2] if t>0]
            if not valid:
                continue
            mttc = min(valid)

        if mttc < 3:
            cv2.putText(frame,f"MTTC:{mttc:.1f}s",
                        (50,50+30*i),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        csv_data.append({
            "frame":frame_count,
            "rear":rear_id,
            "front":front_id,
            "mttc":mttc
        })

    # ---- 🔥 TIMING-CORRECT WRITE ----
    expected_time = frame_index / FPS
    actual_time = time.time() - start_time

    repeat = int(max(1, round((actual_time - expected_time) * FPS)))

    for _ in range(repeat):
        out.write(frame)

    frame_index += 1
    frame_count += 1

# ---- SAVE CSV ----
pd.DataFrame(csv_data).to_csv("mttc2.csv", index=False)

cap.release()
out.release()

print("✅ Smooth video + MTTC CSV done")

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import importlib.util
import math
import time

# ---- LOAD SORT ----
spec = importlib.util.spec_from_file_location(
    "sort_module", "/kaggle/working/sort/sort.py"
)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)
Sort = sort_module.Sort

# ---- MODEL ----
model = YOLO("/kaggle/input/datasets/shashwatkumarjha/weightstrainedfinal/best.pt")

# ---- VIDEO ----
cap = cv2.VideoCapture("/kaggle/input/datasets/shashwatkumarjha/newtrimvideo/ChotiClip.mp4")

# ---- FPS ----
FPS = 100
dt = 1 / FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_30sec_mttc.mp4", fourcc, FPS, (960,540))

tracker = Sort()

# ---- STORAGE ----
track_history = defaultdict(lambda: deque(maxlen=10))
track_speed = defaultdict(float)
track_prev_speed = defaultdict(float)
track_acc = defaultdict(float)

csv_data = []

# ---- HOMOGRAPHY ----
scale_x = 960 / 1920
scale_y = 540 / 1080

img_pts = np.array([
    [540*scale_x,1066*scale_y],
    [1650*scale_x,1066*scale_y],
    [972*scale_x,216*scale_y],
    [1080*scale_x,216*scale_y]
], dtype=np.float32)

depth = 74.65
width = 3.0

world_pts = np.array([
    [0,0],[width,0],[0,depth],[width,depth]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

# ---- FUNCTIONS ----

def pixel_to_world(u, v):
    pt = np.array([u, v, 1]).reshape(3,1)
    world = H @ pt
    world /= world[2]
    return world[0][0], world[1][0]

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2

def compute_speed(track_id, Y):
    hist = track_history[track_id]
    hist.append(Y)

    if len(hist) < 5:
        return 0

    dy = hist[-1] - hist[0]
    dt_total = (len(hist)-1)*dt

    speed = abs(dy)/dt_total * 3.6
    speed = 0.7*track_speed[track_id] + 0.3*speed
    track_speed[track_id] = speed

    return speed

# ---- TIMING CONTROL ----
start_time = time.time()
frame_index = 0

# ---- MAIN ----
frame_count = 0
MAX_SECONDS = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 LIMIT TO 30 SECONDS
    if frame_count >= FPS * MAX_SECONDS:
        break

    frame = frame.copy()
    frame = cv2.resize(frame, (960,540))

    track_positions = {}

    # ---- DETECTION ----
    results = model(frame, imgsz=640, conf=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls not in [0,2] or conf < 0.5:
            continue

        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        detections.append([x1,y1,x2,y2,conf])

    dets = np.array(detections) if detections else np.empty((0,5))
    tracks = tracker.update(dets)

    # ---- PROCESS TRACKS ----
    for trk in tracks:
        x1,y1,x2,y2,track_id = trk
        track_id = int(track_id)

        u,v = get_bottom_center((x1,y1,x2,y2))
        if v < 120:
            continue

        X,Y = pixel_to_world(u,v)

        speed = compute_speed(track_id, Y)

        prev_speed = track_prev_speed[track_id]
        acc = ((speed - prev_speed)/3.6)/dt
        acc = 0.7*track_acc[track_id] + 0.3*acc

        track_prev_speed[track_id] = speed
        track_acc[track_id] = acc

        track_positions[track_id] = {
            "X":X,"Y":Y,"speed":speed,"acc":acc
        }

        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,f"{speed:.1f} km/h",(int(x1),int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    # ---- MTTC ----
    vehicles = sorted(track_positions.items(), key=lambda x: x[1]["Y"])

    for i in range(len(vehicles)-1):
        rear_id, rear = vehicles[i]
        front_id, front = vehicles[i+1]

        if abs(rear["X"] - front["X"]) > 1.5:
            continue

        Vr = rear["speed"]/3.6
        Vf = front["speed"]/3.6
        ar = rear["acc"]
        af = front["acc"]

        dv = Vr - Vf
        da = ar - af
        S = abs(front["Y"] - rear["Y"])

        if dv <= 0:
            continue

        if abs(da) < 1e-3:
            mttc = S/dv
        else:
            disc = dv**2 + 2*da*S
            if disc < 0:
                continue
            t1 = (dv + math.sqrt(disc))/da
            t2 = (dv - math.sqrt(disc))/da
            valid = [t for t in [t1,t2] if t>0]
            if not valid:
                continue
            mttc = min(valid)

        if mttc < 3:
            cv2.putText(frame,f"MTTC:{mttc:.1f}s",
                        (50,50+30*i),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        csv_data.append({
            "frame":frame_count,
            "rear":rear_id,
            "front":front_id,
            "mttc":mttc
        })

    # ---- CONTROLLED WRITE (NO HUGE FILE) ----
    expected_time = frame_index / FPS
    actual_time = time.time() - start_time

    lag = (actual_time - expected_time) * FPS
    repeat = int(max(1, min(3, round(lag))))  # 🔥 capped duplication

    for _ in range(repeat):
        out.write(frame)

    frame_index += 1
    frame_count += 1

# ---- SAVE CSV ----
pd.DataFrame(csv_data).to_csv("mttc2.csv", index=False)

cap.release()
out.release()

print("✅ 30 sec smooth video + MTTC CSV done")

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import importlib.util
import math

# ---- LOAD SORT ----
spec = importlib.util.spec_from_file_location(
    "sort_module", "/kaggle/working/sort/sort.py"
)
sort_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort_module)
Sort = sort_module.Sort

# ---- MODEL ----
model = YOLO("/kaggle/input/datasets/shashwatkumarjha/weightstrainedfinal/best.pt")

# ---- VIDEO ----
cap = cv2.VideoCapture("/kaggle/input/datasets/shashwatkumarjha/newtrimvideo/ChotiClip.mp4")

# ---- FPS ----
FPS = 100
dt = 1 / FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_final_fixed.mp4", fourcc, FPS, (960,540))

tracker = Sort()

# ---- STORAGE ----
track_history = defaultdict(lambda: deque(maxlen=10))
track_speed = defaultdict(float)
track_prev_speed = defaultdict(float)
track_acc = defaultdict(float)

csv_data = []

# ---- HOMOGRAPHY ----
scale_x = 960 / 1920
scale_y = 540 / 1080

img_pts = np.array([
    [540*scale_x,1066*scale_y],
    [1650*scale_x,1066*scale_y],
    [972*scale_x,216*scale_y],
    [1080*scale_x,216*scale_y]
], dtype=np.float32)

depth = 74.65
width = 3.0

world_pts = np.array([
    [0,0],[width,0],[0,depth],[width,depth]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

# ---- FUNCTIONS ----

def pixel_to_world(u, v):
    pt = np.array([u, v, 1]).reshape(3,1)
    world = H @ pt
    world /= world[2]
    return world[0][0], world[1][0]

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2

# 🔥 CORRECT SPEED (original logic restored)
def compute_speed(track_id, Y):
    hist = track_history[track_id]
    hist.append(Y)

    if len(hist) < 5:
        return 0

    dy = hist[-1] - hist[0]
    dt_total = (len(hist)-1)*dt

    speed = abs(dy)/dt_total * 3.6
    speed = 0.7*track_speed[track_id] + 0.3*speed
    track_speed[track_id] = speed

    return speed

# ---- MAIN ----
frame_count = 0
MAX_SECONDS = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ---- LIMIT 30 SEC ----
    if frame_count >= FPS * MAX_SECONDS:
        break

    frame = cv2.resize(frame, (960,540))

    # ---- ROI MASK ----
    roi_pts = np.array([
        [540*scale_x,1066*scale_y],
        [1650*scale_x,1066*scale_y],
        [1080*scale_x,216*scale_y],
        [972*scale_x,216*scale_y]
    ], dtype=np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts], 255)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    # draw ROI
    cv2.polylines(frame, [roi_pts], True, (255,0,0), 2)

    track_positions = {}

    # ---- DETECTION ----
    results = model(frame_masked, imgsz=640, conf=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls not in [0,2] or conf < 0.5:
            continue

        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        detections.append([x1,y1,x2,y2,conf])

    dets = np.array(detections) if detections else np.empty((0,5))
    tracks = tracker.update(dets)

    # ---- PROCESS TRACKS ----
    for trk in tracks:
        x1,y1,x2,y2,track_id = trk
        track_id = int(track_id)

        u,v = get_bottom_center((x1,y1,x2,y2))
        if v < 120:
            continue

        X,Y = pixel_to_world(u,v)

        speed = compute_speed(track_id, Y)

        prev_speed = track_prev_speed[track_id]

        # 🔥 CORRECT ACC (consistent with dt)
        acc = ((speed - prev_speed)/3.6)/dt
        acc = 0.7*track_acc[track_id] + 0.3*acc

        track_prev_speed[track_id] = speed
        track_acc[track_id] = acc

        track_positions[track_id] = {
            "X":X,"Y":Y,"speed":speed,"acc":acc
        }

        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,f"{speed:.1f} km/h",(int(x1),int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    # ---- MTTC ----
    vehicles = sorted(track_positions.items(), key=lambda x: x[1]["Y"])

    for i in range(len(vehicles)-1):
        rear_id, rear = vehicles[i]
        front_id, front = vehicles[i+1]

        if abs(rear["X"] - front["X"]) > 1.5:
            continue

        Vr = rear["speed"]/3.6
        Vf = front["speed"]/3.6
        ar = rear["acc"]
        af = front["acc"]

        dv = Vr - Vf
        da = ar - af
        S = abs(front["Y"] - rear["Y"])

        if dv <= 0:
            continue

        if abs(da) < 1e-3:
            mttc = S/dv
        else:
            disc = dv**2 + 2*da*S
            if disc < 0:
                continue
            t1 = (dv + math.sqrt(disc))/da
            t2 = (dv - math.sqrt(disc))/da
            valid = [t for t in [t1,t2] if t>0]
            if not valid:
                continue
            mttc = min(valid)

        if mttc < 3:
            cv2.putText(frame,f"MTTC:{mttc:.1f}s",
                        (50,50+30*i),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        csv_data.append({
            "frame":frame_count,
            "rear":rear_id,
            "front":front_id,
            "mttc":mttc
        })

    # 🔥 NORMAL WRITE (NO DUPLICATION → FIXES SLOW VIDEO)
    out.write(frame)

    frame_count += 1

# ---- SAVE CSV ----
pd.DataFrame(csv_data).to_csv("mttc2.csv", index=False)

cap.release()
out.release()

print("✅ FINAL: correct speed + normal video playback + ROI + MTTC")

