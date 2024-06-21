import cv2
import numpy as np
import pyautogui
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from langdetect import detect
import nltk
import os

# 設定 tesseract_cmd 為 Tesseract 的安裝路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 檢查並下載必要的 NLTK 資料
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

# 檢查是否有攝像頭
def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    cap.release()
    return True

# 擷取螢幕畫面
def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# 擷取鏡頭畫面
def capture_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# 影像預處理
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary

# 文字識別（OCR）
def extract_text(frame):
    preprocessed_frame = preprocess_image(frame)
    text = pytesseract.image_to_string(preprocessed_frame)
    return text

# 物件偵測
def detect_objects(frame):
    # 使用 OpenCV 的簡單物件偵測方法 (例如邊緣檢測)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 重點整理
def summarize_text(text, num_sentences=3):
    if not text.strip():
        return "無法提取重點內容，文本為空。"
    
    try:
        lang = detect(text)
    except:
        lang = 'en'
    
    sentences = sent_tokenize(text)
    
    if lang == 'en':
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(stopwords.words('english'))  # 這裡你可以添加更多語言的停用詞

    filtered_sentences = [sentence for sentence in sentences if any(word.lower() not in stop_words for word in sentence.split())]
    if not filtered_sentences:
        return "無法提取重點內容，所有句子均為停用詞。"
    
    if len(filtered_sentences) < 2:
        return ' '.join(filtered_sentences)
    
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(filtered_sentences)
    
    if X.shape[1] < 2:
        return ' '.join(filtered_sentences)
    
    svd = TruncatedSVD(n_components=1, n_iter=100)
    svd.fit(X.T)
    scores = np.argsort(svd.components_[0])
    
    summary = [filtered_sentences[i] for i in scores[:num_sentences]]
    return ' '.join(summary)

# 主程式
def main():
    use_camera = check_camera()
    
    while True:
        if use_camera:
            frame = capture_camera()
            if frame is None:
                continue
        else:
            frame = capture_screen()
        
        # 顯示擷取的影像
        cv2.imshow('Captured Image', frame)
        
        # 文字識別
        text = extract_text(frame)
        
        print("識別出的文字：")
        print(text)
        
        # 重點整理
        summary = summarize_text(text)
        
        print("\n重點整理：")
        print(summary)
        
        # 物件偵測
        objects = detect_objects(frame)
        cv2.imshow('Detected Objects', objects)
        
        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
