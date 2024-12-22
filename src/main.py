import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

while True:
  ret, frame = webcam.read()
  
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = hands.process(frame_rgb)
  
  if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
      mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      
      index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
      h, w, _ = frame.shape
      cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
      cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
      
      
  cv2.imshow('Quick Draw', frame)

  if (cv2.waitKey(25) & 0xFF == ord('q')) or cv2.getWindowProperty('Quick Draw', cv2.WND_PROP_VISIBLE) < 1:
    break
  

webcam.release()
cv2.destroyAllWindows
