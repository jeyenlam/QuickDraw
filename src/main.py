import cv2

webcam = cv2.VideoCapture(0)

while True:
  ret, frame = webcam.read()
  cv2.imshow('Quick Draw', frame)
  
  if (cv2.waitKey(25) & 0xFF == ord('q')) or cv2.getWindowProperty('Quick Draw', cv2.WND_PROP_VISIBLE) < 1:
    break
  

webcam.release()
cv2.destroyAllWindows
