from fastapi import APIRouter, UploadFile, File, Form
import numpy as np
import cv2

from services.detection_service import process_frame
from services.state_service import get_room_state
from services.visualization_service import draw_annotations
from services.storage_service import save_image

router = APIRouter()

@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    room_id: str = Form(...)
):
    contents = await file.read()

    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    room_state = get_room_state(room_id)

    results, features, state, belief = process_frame(image, room_state)

    frame = draw_annotations(results)
    filename, image_url = save_image(frame)

    return {
        "features": features,
        "state": state,
        "belief": belief,
        "image_url": image_url
    }