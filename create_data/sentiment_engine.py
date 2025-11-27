from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class VietnameseSentimentAnalyzer:
    def __init__(self):
        logger.info("⏳ Loading Sentiment Model (PhoBERT)...")
        # Sử dụng CPU (device=-1) hoặc GPU (device=0)
        self.classifier = pipeline(
            "sentiment-analysis", 
            model="wonrax/phobert-base-vietnamese-sentiment",
            tokenizer="wonrax/phobert-base-vietnamese-sentiment",
            device=-1, 
            truncation=True, 
            max_length=256
        )
        logger.info(" Sentiment Model Loaded!")

    def analyze_batch_hybrid(self, contents, ratings):
        """
        Kết hợp AI (nếu có text) và Rating (nếu không có text)
        """
        results = []
        indices_to_analyze = []
        texts_to_analyze = []
        
        # 1. Lọc các review có nội dung để chạy AI
        for i, text in enumerate(contents):
            if text and len(str(text).strip()) > 3:
                indices_to_analyze.append(i)
                texts_to_analyze.append(str(text))
        
        # 2. Chạy AI Batch
        ai_outputs = []
        if texts_to_analyze:
            try:
                ai_outputs = self.classifier(texts_to_analyze, batch_size=32)
            except Exception as e:
                logger.error(f"AI Batch Error: {e}")
                ai_outputs = [{'label': 'NEU', 'score': 0.5}] * len(texts_to_analyze)

        # 3. Merge kết quả
        ai_cursor = 0
        for i in range(len(contents)):
            rating = ratings[i]
            if i in indices_to_analyze:
                hf_res = ai_outputs[ai_cursor]
                label, score = self._convert_hf(hf_res)
                # Fallback nếu AI không chắc chắn
                if abs(score) < 0.6:
                    label, score = self._convert_rating(rating)
                ai_cursor += 1
            else:
                label, score = self._convert_rating(rating)
            
            results.append((label, score))
        return results

    def _convert_hf(self, res):
        label = res['label']
        score = res['score']
        if label == 'POS': return 'positive', score
        elif label == 'NEG': return 'negative', -score
        return 'neutral', 0.0

    def _convert_rating(self, rating):
        if rating >= 5: return 'positive', 1.0
        elif rating >= 4: return 'positive', 0.8
        elif rating == 3: return 'neutral', 0.0
        elif rating == 2: return 'negative', -0.5
        return 'negative', -1.0