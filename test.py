import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

# 下載已經訓練好的BERT模型和標籤
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)

# 定義pipeline
punctuation_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# 需要處理的文字
text = "what are other ways we can process talk about think about citizen in yes as a verb"

# 預測標點符號
punctuated_text = punctuation_pipeline(text)

# 處理標點符號的預測結果並生成帶有標點符號的文本
def add_punctuation(text, predictions):
    words = text.split()
    punctuated_text = ""
    for word, prediction in zip(words, predictions):
        punctuated_text += word
        if prediction['entity'] == 'O':
            punctuated_text += ' '
        else:
            punctuated_text += prediction['word'][-1] + ' '
    return punctuated_text.strip()

# 獲取預測結果
predictions = punctuation_pipeline(text)
# 加上標點符號
output_text = add_punctuation(text, predictions)
print(output_text)
