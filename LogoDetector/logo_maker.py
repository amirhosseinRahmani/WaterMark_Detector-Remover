import cv2
import numpy as np
import random
import os

VIDEO_PATH = 'somefile.mp4'             # Input video path
OUTPUT_VIDEO = 'video_with_logo.mp4'  # Output video
LOGO_IMAGE = 'logo.png'               # Logo image

# Create dummy logo if it doesn't exist
if not os.path.exists(LOGO_IMAGE):
    logo = np.zeros((50, 100, 3), dtype=np.uint8)
    logo[:] = (0, 255, 255)
    cv2.putText(logo, 'LOGO', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(LOGO_IMAGE, logo)

# Load and resize logo
logo = cv2.imread(LOGO_IMAGE)
logo = cv2.resize(logo, (30, 15))  # Make it smaller
logo_h, logo_w = logo.shape[:2]

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_idx = 0
x, y = 0, 0  # Initial logo position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only move logo every 10 frames
    if frame_idx % 10 == 0:
        x = random.randint(0, frame_width - logo_w)
        y = random.randint(0, frame_height - logo_h)

    # Overlay logo
    frame[y:y+logo_h, x:x+logo_w] = logo

    out.write(frame)

    #Show progress
    percent = (frame_idx + 1) / total_frames * 100
    print(f'\rProcessing: {percent:.2f}%', end='')

    frame_idx += 1

cap.release()
out.release()
print("\n✅ Done! Output saved as:", OUTPUT_VIDEO)
