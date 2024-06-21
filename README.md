# 期末報告書

## Install 
```
pip install sounddevice torch torchaudio opencv-python numpy pillow
pip install python-docx torchvision pyaudio transformers vosk jieba imagehash
```

## 初衷

在上課或聽演講時，難免會遇到需要專心聆聽並理解內容的情況。然而，經常會因為各種原因而分心。為了解決這個問題，我設計了一個工具，可以幫助自動截圖、錄音、語音轉文字和做摘要，這樣在需要回顧時就能快速抓住重點。

## 目標

    截圖/拍照
    錄音
    語音轉文字
    存檔
    做摘要
    紀錄已經獲取的資訊

## 主要功能

* 語音轉文字：使用 VOSK 和 Wav2Vec2 模型進行語音識別，實現高效的語音轉文字。
* 文字摘要：使用 T5 模型進行文字摘要，幫助快速抓住演講或課程中的重點。
* 影像處理：包括模糊檢測、特徵提取和圖像儲存，確保每個關鍵畫面都能被捕捉。
* 多線程處理：通過多線程技術，避免某個動作卡住其他功能，確保所有數據都能完整儲存。

## 各部分程式碼說明
sum_all_rr.py (主程式)

此程式結合語音轉文字、影像處理及文字摘要功能，並將結果儲存到 Word 文檔中。
它統籌了整個項目的執行流程，確保各部分功能協同工作。

speech2text_vosk.py

用於實時語音轉文字，使用 VOSK 語音識別模型來實現，並將識別出的文字進行自動分段。VOSK 在離線語音辨識方面表現出色，是一個可靠的選擇。

video.py

影像處理程式，包括螢幕擷取、攝像頭擷取、影像預處理、文字識別（OCR）、物件偵測和文字摘要。它能夠捕捉和處理多種形式的視覺數據，並從中提取有用的信息。

cap_image.py

捕捉圖像並檢測是否模糊，若不模糊則進行特徵提取並儲存不同的圖像。這樣可以確保儲存的圖像都是清晰且有用的。

cleancache.py

清理 Hugging Face 模型緩存，並重新載入預訓練的 Wav2Vec2 模型和處理器。如果模型無法正確載入，可以通過執行此程式來解決問題。

speech2text.py

用於實時語音轉文字，使用 Wav2Vec2 模型進行語音識別，並使用 T5 模型進行文字摘要。這個程式能夠處理實時語音輸入，並生成相應的文字摘要。

cam_get_point.py

從攝像頭捕捉圖像，進行模糊檢測和特徵提取，並保存不同的圖像。確保每一個重要瞬間都能被記錄下來，而不會因為模糊或重複的圖像而浪費資源。

## 實際畫面

![圖片](https://github.com/MAXKIRITO/Auto_Summarize_something/assets/67036239/72efb6f1-4edb-41bd-9a91-d1c29ce35273)

![圖片](https://github.com/MAXKIRITO/Auto_Summarize_something/assets/67036239/b2a61c56-9aa9-429b-ae1c-1b1aba5ccb28)


如果要看單獨圖片的話會在image_a/ 底下會出現有擷取到的圖片原始檔案
