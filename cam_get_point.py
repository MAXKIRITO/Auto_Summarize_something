import cv2
import numpy as np
import os
from PIL import Image
import imagehash
from collections import deque

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_different(image1, image2, hash_size=8, similarity_threshold=30):
    hash1 = imagehash.average_hash(Image.fromarray(image1), hash_size=hash_size)
    hash2 = imagehash.average_hash(Image.fromarray(image2), hash_size=hash_size)
    return hash1 - hash2 > (hash_size * hash_size * similarity_threshold / 100)

# Create the directory if it doesn't exist
save_dir = 'img_a'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(1)  # 0 indicates the first camera
ret, prev_frame = cap.read()

if not ret:
    print("Error: Could not read from camera.")
    cap.release()
    exit()

save_count = 0
recent_frames = deque(maxlen=10)  # Store hashes of the most recent frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not is_blurry(frame):
        different_from_all = all(is_different(frame, prev_frame) for prev_frame in recent_frames)
        if different_from_all:
            save_path = os.path.join(save_dir, f'image_{save_count}.jpg')
            cv2.imwrite(save_path, frame)
            save_count += 1
            recent_frames.append(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
