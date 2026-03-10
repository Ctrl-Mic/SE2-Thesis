from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = FastAPI()

# Load YOLO model once
model = YOLO("./model/yolov8n.pt")

# Exit zones (define once)
EXIT_ZONES = [
    (0, 300, 100, 480),   # door 1
    (540, 300, 640, 480)  # door 2
]

# ---------------- HELPERS ----------------

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def intersects(box, zone):
    bx1, by1, bx2, by2 = box
    zx1, zy1, zx2, zy2 = zone

    return not (
        bx2 < zx1 or bx1 > zx2 or
        by2 < zy1 or by1 > zy2
    )

def compute_motion(current_centers, prev_centers):
    if not prev_centers or not current_centers:
        return 0.0

    distances = []

    for c in current_centers:
        d = min(
            np.linalg.norm(np.array(c) - np.array(p))
            for p in prev_centers
        )
        distances.append(d)

    motion = float(np.mean(distances))
    return min(motion / 50.0, 1.0)

# -----------------------------------------

previous_centers = []

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    global previous_centers

    try:
        contents = await file.read()

        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image file"}

        # Run YOLO
        results = model(image)

        # Extract detection data
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        people_boxes = []
        confidences = []

        for box, conf, cls in zip(boxes, confs, classes):
            class_name = model.names[int(cls)]

            if class_name == "person":
                people_boxes.append(box)
                confidences.append(float(conf))

        people_count = len(people_boxes)

        avg_confidence = (
            sum(confidences) / people_count
            if people_count > 0 else 0
        )

        # Motion calculation
        centers = [get_center(b) for b in people_boxes]
        motion_level = compute_motion(centers, previous_centers)

        # Exit activity detection
        exit_activity = any(
            intersects(box, zone)
            for box in people_boxes
            for zone in EXIT_ZONES
        )

        previous_centers = centers

        # Create YOLO annotated frame
        annotated_frame = results[0].plot()

        # Draw exit zones
        for zone in EXIT_ZONES:
            x1, y1, x2, y2 = zone

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                5
            )

            cv2.putText(
                annotated_frame,
                "EXIT",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Save annotated image
        os.makedirs("./results", exist_ok=True)
        output_path = "./results/annotated_image.jpg"

        cv2.imwrite(output_path, annotated_frame)

        features = {
            "people_count": people_count,
            "motion_level": motion_level,
            "exit_activity": exit_activity,
            "avg_confidence": avg_confidence
        }
        
        print(f"Features: {features} \nprevious_centers: {previous_centers} \noutput_path: {output_path}")

        return {
            "message": "Detection complete",
            "features": features,
            "output_path": output_path
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health_check():
    return {"status": "healthy"}