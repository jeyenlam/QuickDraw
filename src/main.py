import cv2
import mediapipe as mp
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained model and define labels
model = pickle.load(open('C:/Users/yenla/AppsByYen/QuickDraw/src/quickdraw_mlp_model.pkl', 'rb'))
label_dict = {0: 'Airplane', 1: 'Angel', 2: 'Banana', 3: 'Bowtie', 4: 'Butterfly', 5: 'Cactus'}

# Initialize global variables
prev_x, prev_y = 0, 0
drawing = False
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def init_webcam():
    return cv2.VideoCapture(0)

def process_frame(frame, hands):
    """Process each frame and return the processed result from mediapipe"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    return result

def draw_landmarks(frame, result):
    """Draw hand landmarks on the frame"""
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def handle_drawing(result, frame, canvas, w, h):
    """Handle drawing logic"""
    global prev_x, prev_y, drawing
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            if drawing:
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 5)
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0
    return canvas

def predict_and_show(canvas, model, label_dict):
    """Resize, normalize, and predict the drawing on canvas"""
    canvas_resized = cv2.resize(canvas, (28, 28))
    canvas_gray = cv2.normalize(canvas_resized, None, 0, 255, cv2.NORM_MINMAX)
    canvas_normalized = canvas_gray / 255.0
    canvas_reshaped = canvas_normalized.reshape(-1, 784)
    
    prediction = model.predict(canvas_reshaped)
    result = label_dict[prediction[0]]
    print(result)
    
    plt.imshow(canvas_gray, cmap='gray')
    plt.show()

def main():
    global drawing, canvas
    webcam = init_webcam()
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
    
    while True:
        success, frame = webcam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process frame using mediapipe hands
        result = process_frame(frame, hands)
        frame = draw_landmarks(frame, result)
        canvas = handle_drawing(result, frame, canvas, w, h)
        
        # Combine frame with canvas
        combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow('Quick Draw', combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar
            drawing = not drawing
            prev_x, prev_y = 0, 0
        
        if key == ord('c'):  # Clear canvas
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if key == ord('p'):  # Predict
            predict_and_show(canvas, model, label_dict)
        
        if key == ord('q') or cv2.getWindowProperty('Quick Draw', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()