import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
drawing = False  # Initially, not drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # 480 x 640
    
    # Instructions
    cv2.putText(frame, 'Press Space to start/stop drawing, c to clear', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Convert the image color to RGB for mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Marking index finger tip
            
            if drawing:  # Draw only if drawing mode is enabled
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 3)  # Draw on the canvas
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0

    # Combine the original frame with the canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow('Quick Draw', combined)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Toggle drawing mode with spacebar
    if key == 32:  # Spacebar
        drawing = not drawing  # Toggle between drawing and not drawing
        prev_x, prev_y = 0, 0  # Reset the previous points when toggling

    # Clear the canvas if 'c' is pressed
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Exit if 'q' is pressed or the window is closed
    if key == ord('q') or cv2.getWindowProperty('Quick Draw', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
