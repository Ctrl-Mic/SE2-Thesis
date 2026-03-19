# helpers/inference.py

# -------------------------------
# Utility: Normalize probabilities
# Ensures belief sums to 1
# -------------------------------
def normalize(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


# -------------------------------
# TIME-AWARE TRANSITION MODEL
# Models how states evolve over time
# dt = time between frames (3 seconds in your case)
# -------------------------------
def state_transition(prev, dt=1.0):
    alpha = min(dt, 3.0)  # cap to prevent instability

    return {
        # Mostly stays Occupied unless strong evidence to leave
        "Occupied": (1 - 0.15*alpha) * prev["Occupied"]
                    + 0.10*alpha * prev["Leaving"]
                    + 0.05*alpha * prev["Empty"],

        # Leaving is a temporary state
        "Leaving":  0.10*alpha * prev["Occupied"]
                    + (1 - 0.20*alpha) * prev["Leaving"]
                    + 0.10*alpha * prev["Empty"],

        # Empty is stable but only reached through transitions
        "Empty":    0.05*alpha * prev["Occupied"]
                    + 0.10*alpha * prev["Leaving"]
                    + (1 - 0.15*alpha) * prev["Empty"]
    }


# -------------------------------
# HISTORY-AWARE FEATURE ENHANCEMENT
# Uses last N frames to stabilize noisy detections
# -------------------------------
def enhance_with_history(obs, history):
    if len(history) < 3:
        return obs  # not enough data yet

    recent = list(history)[-3:]  # last ~9 seconds

    # -------------------------------
    # 1. Exit persistence
    # If exit detected recently, keep it active
    # -------------------------------
    exit_recent = any(f["exit_activity"] for f in recent)

    # -------------------------------
    # 2. People trend
    # Detect decreasing count → possible leaving
    # -------------------------------
    people_trend = recent[0]["people_count"] - recent[-1]["people_count"]

    # -------------------------------
    # 3. Motion confirmation
    # Use max instead of average to avoid smoothing away events
    # -------------------------------
    motion_recent = max(f["motion_level"] for f in recent)

    # -------------------------------
    # 4. Occupancy persistence
    # If someone was seen recently, assume still inside
    # -------------------------------
    recent_people = any(f["people_count"] > 0 for f in recent)

    enhanced = obs.copy()

    # Preserve occupancy even if current frame misses detection
    if recent_people:
        enhanced["force_occupied"] = True

    # Strengthen exit signal
    if exit_recent:
        enhanced["exit_activity"] = True

    # Add trend info
    enhanced["trend_leaving"] = people_trend > 0

    # Use strongest motion
    enhanced["motion_level"] = max(obs["motion_level"], motion_recent)

    return enhanced


# -------------------------------
# OBSERVATION LIKELIHOOD
# Converts features → probability of each state
# IMPORTANT:
# - No direct "Empty" inference from observation
# - Camera sees only exit area → absence ≠ empty
# -------------------------------
def observation_likelihood(obs):
    people = obs["people_count"]
    motion = obs["motion_level"]
    exit_act = obs["exit_activity"]
    trend_leaving = obs.get("trend_leaving", False)
    force_occupied = obs.get("force_occupied", False)

    # -------------------------------
    # OCCUPIED likelihood
    # Strong default assumption due to partial observability
    # -------------------------------
    occ = 0.6 + 0.4 * (people > 0)

    # -------------------------------
    # LEAVING likelihood
    # Dominated by exit detection
    # -------------------------------
    lea = 0.9 * exit_act + 0.1 * trend_leaving

    # -------------------------------
    # EMPTY likelihood
    # NEVER inferred directly from observation
    # -------------------------------
    emp = 0.05

    # -------------------------------
    # If recent occupancy detected → strongly bias against empty
    # -------------------------------
    if force_occupied:
        occ = max(occ, 0.8)
        emp = 0.01

    return {
        "Occupied": max(0.001, min(occ, 0.999)),
        "Leaving":  max(0.001, min(lea, 0.999)),
        "Empty":    max(0.001, min(emp, 0.999))
    }


# -------------------------------
# BELIEF UPDATE (Bayesian Step)
# Combines:
# - Previous belief (memory)
# - Current observation (evidence)
# -------------------------------
def update_belief(prev_belief, obs, history, dt=1.0):
    # Enhance observation using history
    obs = enhance_with_history(obs, history)

    # Predict next state (transition)
    predicted = state_transition(prev_belief, dt)

    # Compute likelihood of observation
    likelihood = observation_likelihood(obs)

    # Combine (Bayesian update)
    updated = {
        k: predicted[k] * likelihood[k]
        for k in prev_belief
    }

    return normalize(updated)


# -------------------------------
# FINAL DECISION
# Select state with highest probability
# -------------------------------
def infer_state(belief):
    return max(belief, key=belief.get)