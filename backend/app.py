from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from collections import deque
import numpy as np
import cv2
import os
import uuid

from helpers.feature import extract_features, EXIT_ZONES
from helpers.inference import update_belief, infer_state

MAX_HISTORY = 10

image_history = deque(maxlen=MAX_HISTORY)
feature_history = deque(maxlen=MAX_HISTORY)

belief = {
    "Occupied": 0.1,
    "Leaving": 0.0,
    "Empty": 0.9
}

DT = 3.0

# -------------------------------
# NEW: Estimated occupancy (core memory)
# -------------------------------
estimated_occupancy = 0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory="results"), name="results")

model = YOLO("./model/yolov8n.pt")

previous_centers = []


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    global previous_centers
    global belief
    global estimated_occupancy

    try:
        contents = await file.read()

        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image file"}

        results = model(image)

        # -------------------------------
        # Feature extraction
        # -------------------------------
        features, centers = extract_features(results, previous_centers)
        previous_centers = centers

        # -------------------------------
        # NEW: Update estimated occupancy
        # -------------------------------

        # ENTRY → increase count
        if features["entry_count"] > 0:
            estimated_occupancy += features["entry_count"]

        # EXIT → decrease count
        if features["exit_count"] > 0:
            estimated_occupancy -= features["exit_count"]

        # Clamp to 0 (no negative people)
        estimated_occupancy = max(0, estimated_occupancy)

        # Fallback: if we detect more people than estimated
        if features["people_count"] > estimated_occupancy:
            estimated_occupancy = features["people_count"]

        # Inject into features for inference
        features["estimated_occupancy"] = estimated_occupancy

        feature_history.append(features)

        # -------------------------------
        # Inference
        # -------------------------------
        belief = update_belief(
            belief,
            features,
            feature_history,
            dt=DT
        )

        state = infer_state(belief)

        # -------------------------------
        # Visualization
        # -------------------------------
        annotated_frame = results[0].plot()

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

        os.makedirs("./results", exist_ok=True)

        filename = f"{uuid.uuid4()}.jpg"
        output_path = f"./results/{filename}"

        cv2.imwrite(output_path, annotated_frame)

        image_history.append(filename)

        if len(image_history) == MAX_HISTORY:
            existing_files = set(os.listdir("./results"))
            valid_files = set(image_history)

            for file in existing_files:
                if file not in valid_files:
                    try:
                        os.remove(os.path.join("./results", file))
                    except:
                        pass

        image_url = f"http://localhost:8000/results/{filename}"

        print(f"Features: {features}")
        print(f"Estimated Occupancy: {estimated_occupancy}")

        return {
            "message": "Detection complete",
            "features": features,
            "state": state,
            "belief": belief,
            "history": list(feature_history),
            "image_url": image_url
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health_check():
    return {"status": "healthy"}