from collections import deque

MAX_HISTORY = 10

class RoomState:
    def __init__(self):
        self.previous_centers = []
        self.feature_history = deque(maxlen=MAX_HISTORY)
        self.belief = {
            "Occupied": 0.1,
            "Leaving": 0.0,
            "Empty": 0.9
        }
        self.estimated_occupancy = 0
        self.current_state = "Empty"
        self.state_duration = 0.0

room_states = {}

def get_room_state(room_id: str) -> RoomState:
    if room_id not in room_states:
        room_states[room_id] = RoomState()
    return room_states[room_id]