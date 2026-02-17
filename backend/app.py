from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = FastAPI()
model = YOLO("./model/yolov8n.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_array = np.frombuffer(contents, np.uint8)

        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image file"}

        results = model(image)

        count = 0
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name == "person":
                count += 1

        # 6️⃣ Optional: save annotated image
        annotated_frame = results[0].plot()
        os.makedirs("./results", exist_ok=True)
        output_path = "./results/annotated_image.jpg"
        cv2.imwrite(output_path, annotated_frame)

        return {"message": "Detection complete", "count": count, "output_path": output_path}

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
