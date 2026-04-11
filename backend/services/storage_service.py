import os
import uuid
import cv2

RESULT_DIR = "./results"
MAX_FILES = 5

def save_image(frame):
    os.makedirs(RESULT_DIR, exist_ok=True)

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(RESULT_DIR, filename)

    cv2.imwrite(path, frame)

    # cleanup old files
    files = sorted(
        [os.path.join(RESULT_DIR, f) for f in os.listdir(RESULT_DIR)],
        key=os.path.getctime
    )

    if len(files) > MAX_FILES:
        for old_file in files[:len(files) - MAX_FILES]:
            os.remove(old_file)

    return filename, f"http://localhost:8000/results/{filename}"