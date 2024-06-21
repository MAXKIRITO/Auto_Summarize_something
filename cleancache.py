import shutil
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 清理本地緩存
cache_dir = './cache/huggingface'
shutil.rmtree(cache_dir, ignore_errors=True)

# 加載預訓練的多語言 Wav2Vec2 模型和 processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir)
