import cv2
from helpers.feature import EXIT_ZONES

def draw_annotations(results):
    frame = results[0].plot()

    for zone in EXIT_ZONES:
        x1, y1, x2, y2 = zone

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
        cv2.putText(frame, "EXIT", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame