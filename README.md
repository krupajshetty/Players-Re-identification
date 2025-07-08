Here’s the complete **README.md** file content you can copy and save as `README.md` in your project folder:

---

# 📹 Player Tracking with YOLO & DeepSORT

This project performs **player detection and tracking** in a video using a custom-trained YOLOv8 model and the DeepSORT tracker.
It generates an output video with adjusted bounding boxes and consistent track IDs for players, and also logs tracking data to a CSV file.

---

## 📦 Setup

### 1️⃣ Clone or download the repository

```bash
git clone <your-repo-url>
cd Stealth_Mode
```

### 2️⃣ Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies are:

* `ultralytics` (YOLOv8)
* `deep_sort_realtime`
* `opencv-python`
* `torch`
* `numpy`

---

## 🎬 Run the script

1️⃣ Place your input video in the `videos/` folder.
2️⃣ Make sure your YOLOv8 weights file (`best.pt`) is in the `models/` folder.

Run the tracking script:

```bash
python scripts.py
```

You can stop the script early by pressing **q** while the video preview window is open.

---

## 📂 Outputs

After running the script, you’ll find:

* 🎥 Annotated video:
  `outputs/tracked_output_deepsort.mp4`

* 📄 Tracking log CSV:
  `outputs/tracking_log.csv`

The CSV contains:

| Frame | Track\_ID | Orig\_X1 | Orig\_Y1 | Orig\_X2 | Orig\_Y2 | Adj\_X1 | Adj\_Y1 | Adj\_X2 | Adj\_Y2 |
| ----- | --------- | -------- | -------- | -------- | -------- | ------- | ------- | ------- | ------- |

Where `Orig_*` are the original bounding box coordinates and `Adj_*` are the adjusted box coordinates centered on the player’s bottom.

---

## 📝 Notes & Tips

* The YOLO model must be trained to detect players and have a class named exactly `player`. Check `model.names` if necessary.
* You can adjust parameters like `scale_factor`, `target_height`, `target_width` in `scripts.py` to customize box size.
* If tracking is unstable (ID switches), experiment with DeepSORT parameters like `max_cosine_distance`, `max_age`.

---

## 📋 Project Structure

```
Stealth_Mode/
├── models/
│   └── best.pt                # YOLO weights
├── outputs/
│   ├── tracked_output_deepsort.mp4
│   └── tracking_log.csv
├── videos/
│   └── 15sec_input_720p.mp4   # your input video
├── scripts.py                 # main tracking script
├── requirements.txt           # Python dependencies
├── README.md                  # this file
```

