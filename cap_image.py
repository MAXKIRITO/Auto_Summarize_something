import os
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
import torch
import torchvision.transforms as transforms
from PIL import Image

# Initialize the image model
image_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
image_model.eval()
if torch.cuda.is_available():
    image_model = image_model.to('cuda')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def extract_features(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    with torch.no_grad():
        features = image_model(input_tensor)
    return features

def is_different(features1, features2, similarity_threshold=75):
    similarity = torch.nn.functional.cosine_similarity(features1, features2).item()                                         #here
    return similarity < (similarity_threshold / 100)

def capture_images(save_dir, capture_interval=1, similarity_threshold=75):
    recent_features = deque(maxlen=10) # Store features of last 10 images
    cap = cv2.VideoCapture(2)  # 攝像頭索引，根據需要更改
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 嘗試設置攝像頭最高解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 獲取並打印當前解析度設置
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Current camera resolution: {width}x{height}")

    fgbg = cv2.createBackgroundSubtractorMOG2()
    last_saved_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        non_zero_count = np.count_nonzero(fgmask)

        if non_zero_count > (frame.shape[0] * frame.shape[1] * 0.05):
            current_time = time.time()
            if current_time - last_saved_time >= capture_interval:
                blurry = is_blurry(frame)
                if not blurry:
                    fg_frame = cv2.bitwise_and(frame, frame, mask=fgmask)
                    frame_features = extract_features(fg_frame)
                    different_from_all = all(is_different(frame_features, prev_features, similarity_threshold=similarity_threshold) for prev_features in recent_features)

                    if different_from_all:
                        timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S.%fZ')
                        save_path = os.path.join(save_dir, f'image_{timestamp_str}.jpg')
                        cv2.imwrite(save_path, frame)
                        recent_features.append(frame_features)
                        last_saved_time = current_time

        cv2.imshow('Video', frame)
        cv2.imshow('Foreground Mask', fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    save_dir = 'img_a'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    capture_images(save_dir)

if __name__ == "__main__":
    main()
