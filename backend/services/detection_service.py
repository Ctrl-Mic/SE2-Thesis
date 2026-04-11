import numpy as np
import cv2

from core.model import model
from helpers.feature import extract_features
from helpers.inference import update_belief, infer_state

DT = 3.0

def process_frame(image, room_state):
    results = model(image)

    # Feature extraction
    features, centers = extract_features(
        results,
        room_state.previous_centers
    )
    room_state.previous_centers = centers

    # Occupancy logic
    if features["entry_count"] > 0:
        room_state.estimated_occupancy += features["entry_count"]

    # STRONG EXIT HANDLING
    if features["exit_count"] > 0:
        room_state.estimated_occupancy -= features["exit_count"]
        # HARD CLAMP (important)
        if room_state.estimated_occupancy < 0.5:
            room_state.estimated_occupancy = 0

    # ONLY decay if ZERO exit activity for a while
    recent = list(room_state.feature_history)[-3:]
    recent_exit = any(f["exit_activity"] for f in recent)

    if not recent_exit:
        room_state.estimated_occupancy = max(
            0,
            room_state.estimated_occupancy - 0.1
        )

    if features["people_count"] == 0 and not features["exit_activity"]:
        # VERY slow decay to handle missed exits
        room_state.estimated_occupancy = max(
            0,
            room_state.estimated_occupancy - 0.2
        )
    if features["people_count"] > room_state.estimated_occupancy:
        room_state.estimated_occupancy = features["people_count"]

    features["estimated_occupancy"] = room_state.estimated_occupancy
        
    room_state.feature_history.append(features)

    # Inference
    room_state.belief = update_belief(
        room_state.belief,
        features,
        room_state.feature_history,
        dt=DT,
        room_state=room_state
    )

    state = infer_state(
        room_state.belief,
        room_state,
        DT
    )
    

    return results, features, state, room_state.belief