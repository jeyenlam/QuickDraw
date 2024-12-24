import cv2
import mediapipe as mp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
drawing = False
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

webcam = cv2.VideoCapture(0)

model = pickle.load(open('./quickdraw_mlp_model.pkl', 'rb'))
label_dict = {0: 'Angel', 1: 'Banana'}

while True:
    sucess, frame = webcam.read()
    if not sucess:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # 480 x 640
    cv2.putText(frame, 'Press Space to start', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

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
            
            if drawing:  # Draw only if drawing model is enabled
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 3)  # Draw on the canvas
                    
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0

    # Combine the original frame with the canvas
    combined_frame_canvas = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow('Quick Draw', combined_frame_canvas)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # Spacebar
        drawing = not drawing  # Toggle between drawing and not drawing
        prev_x, prev_y = 0, 0  # Reset the previous points when toggling

    # Clear the canvas if 'c' is pressed
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
    if key == ord('p'):
        canvas_resized = cv2.resize(canvas, (28, 28))
        # canvas_gray = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
        canvas_gray = cv2.normalize(canvas_resized, None, 0, 255, cv2.NORM_MINMAX)
        canvas_normalized = canvas_gray / 255.0
        canvas_reshaped = canvas_normalized.reshape(-1, 784)
        prediction = model.predict(canvas_reshaped)
        # print(prediction)
        result = label_dict[prediction[0]]
        print(result)
        # canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for correct plotting
        plt.imshow(canvas_gray, cmap='gray')
        plt.show()

    # Exit if 'q' is pressed or the window is closed
    if key == ord('q') or cv2.getWindowProperty('Quick Draw', cv2.WND_PROP_VISIBLE) < 1:
        break


# Release resources
webcam.release()
cv2.destroyAllWindows()
