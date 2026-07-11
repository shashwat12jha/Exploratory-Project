import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import math
import time
import json

# Import SORT
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort"))
from sort import Sort

def process_video_pipeline(video_path: str, output_dir: str, job_id: str, best_pt_path: str = "best.pt") -> dict:
    print(f"Starting pipeline for {job_id} on {video_path}")
    
    # ---- MODEL ----
    model = YOLO(best_pt_path)

    # ---- VIDEO ----
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Failed to open video: {video_path}")

    # ---- FPS ----
    FPS = cap.get(cv2.CAP_PROP_FPS)
    if FPS < 1 or math.isnan(FPS):
        FPS = 100  # Fallback if unreadable
    dt = 1 / FPS

    # ---- OUTPUT VIDEO ----
    output_video_path = os.path.join(output_dir, f"{job_id}_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (960, 540))

    tracker = Sort()

    # ---- STORAGE ----
    track_history = defaultdict(lambda: deque(maxlen=10))
    track_speed = defaultdict(float)
    track_prev_speed = defaultdict(float)
    track_acc = defaultdict(float)

    csv_data = []

    # ---- HOMOGRAPHY ----
    # Based on notebook scale_x = 960 / 1920
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
        [0,0], [width,0], [0,depth], [width,depth]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts)

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

    # ---- MAIN LOOP ----
    frame_count = 0
    total_vehicles = set()
    total_mttc_conflicts = 0
    total_pet_conflicts = 0  # Assuming similar logic for crossing/PET if added later

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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

            # Class 0 and 2 typical for vehicle/person tracking in this model
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
            total_vehicles.add(track_id)

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

            mttc = float('inf')
            if abs(da) < 1e-3:
                mttc = S/dv
            else:
                disc = dv**2 + 2*da*S
                if disc >= 0:
                    t1 = (dv + math.sqrt(disc))/da
                    t2 = (dv - math.sqrt(disc))/da
                    valid = [t for t in [t1,t2] if t>0]
                    if valid:
                        mttc = min(valid)

            if mttc < 3:
                total_mttc_conflicts += 1
                cv2.putText(frame,f"MTTC:{mttc:.1f}s",
                            (50,50+30*i),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            csv_data.append({
                "frame":frame_count,
                "rear":rear_id,
                "front":front_id,
                "mttc":mttc
            })

        out.write(frame)
        frame_count += 1

    # ---- SAVE CSV ----
    stats_path = os.path.join(output_dir, f"{job_id}_stats.json")
    csv_path = os.path.join(output_dir, f"{job_id}_mttc.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    cap.release()
    out.release()
    
    # Generate Heatmap (Dummy logic for now, or you can add your KDE/Heatmap logic here)
    heatmap_path = os.path.join(output_dir, f"{job_id}_heatmap.jpg")
    cv2.imwrite(heatmap_path, np.zeros((540, 960, 3), dtype=np.uint8))

    stats = {
        "total_vehicles": len(total_vehicles),
        "rear_end_conflicts": total_mttc_conflicts,
        "crossing_conflicts": total_pet_conflicts, # Placeholder
        "mttc_avg": sum(d['mttc'] for d in csv_data if d['mttc'] < 10) / max(1, len([d for d in csv_data if d['mttc'] < 10])),
        "pet_avg": 0.0 # Placeholder
    }
    
    with open(stats_path, "w") as f:
        json.dump(stats, f)
        
    print(f"Finished pipeline for {job_id}")
    return stats
