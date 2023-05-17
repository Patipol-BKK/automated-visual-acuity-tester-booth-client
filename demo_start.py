import cv2
import numpy as np
import sys
from ffpyplayer.player import MediaPlayer

file_name = "intro.mp4"
window_name = "window"
interframe_wait_ms = 24

cap = cv2.VideoCapture(file_name)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

player = MediaPlayer(file_name)

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while (True):
    ret, frame = cap.read()
    audio_frame, val = player.get_frame()
    if not ret:
        print("Reached end of video, exiting.")
        break

    cv2.imshow(window_name, frame)
    if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
        print("Exit requested.")
        break

cap.release()
cv2.destroyAllWindows()

os.system("python display.py")
