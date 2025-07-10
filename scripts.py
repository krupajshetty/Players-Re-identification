import cv2
import numpy as np
import csv
from ultralytics import YOLO
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

# Output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# CSV log file
csv_path = output_dir / "tracking_log.csv"
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Track_ID",
                     "Orig_X1", "Orig_Y1", "Orig_X2", "Orig_Y2",
                     "Adj_X1", "Adj_Y1", "Adj_X2", "Adj_Y2"])

# Load YOLO model
print("[INFO] Loading YOLO model...")
model = YOLO('models/best.pt')

# Class names
class_names = model.names
print(f"[INFO] Class names: {class_names}")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.4)

# Input & output videos
video_path = 'videos/15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = str(output_dir / 'tracked_output_deepsort.mp4')

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"[INFO] Saving output video to: {output_path}")

frame_num = 0

# Parameters
target_height = 150   # pixels (can adjust depending on video resolution)
target_width = 50     # pixels
scale_factor = 1.2    # to slightly scale the boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Run YOLO detection
    results = model(frame)

    # Extract player detections for DeepSORT
    detections = []  # format: [x1, y1, x2, y2, confidence, class_name]
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = class_names.get(cls, None)
            if label == 'player':  # adjust if your model uses a different name
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                detections.append(([float(x1), float(y1), float(x2), float(y2)], conf, label))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results & log
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

        # Original box
        l, t, r, b = track.to_ltrb()
        orig_box = [int(l), int(t), int(r), int(b)]

        # Compute bottom-center point
        cx = int((l + r) / 2)
        cy = int(b)  # bottom of detected box

        # Adjust box based on bottom-center point
        adj_w = int((r - l) * scale_factor)
        adj_h = int((b - t) * scale_factor)

        # Optionally override with fixed size:
        # adj_w, adj_h = target_width, target_height

        # Compute adjusted box
        l_adj = max(cx - adj_w // 2, 0)
        r_adj = min(cx + adj_w // 2, width - 1)
        t_adj = max(cy - adj_h, 0)
        b_adj = cy

        adj_box = [l_adj, t_adj, r_adj, b_adj]

        # Log to CSV
        csv_writer.writerow([
            frame_num, track_id,
            *orig_box, *adj_box
        ])

        # Draw adjusted box
        cv2.rectangle(frame, (l_adj, t_adj), (r_adj, b_adj), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (l_adj, t_adj - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Player Tracking (Bottom-Center Fit)", frame)

    # Write frame
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"[INFO] Video saved successfully at: {output_path}")
print(f"[INFO] Tracking log saved at: {csv_path}")
