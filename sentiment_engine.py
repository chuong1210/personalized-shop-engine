"""
sentiment_engine.py - Module phân tích cảm xúc tiếng Việt (Hybrid)
Sử dụng Hugging Face Transformers + Logic Rating fallback
"""
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class VietnameseSentimentAnalyzer:
    def __init__(self):
        logger.info("⏳ Loading Sentiment Analysis Model (PhoBERT)...")
        try:
            # Sử dụng model chuyên cho sentiment tiếng Việt
            # device=-1 là CPU. Nếu có GPU NVIDIA, đổi thành device=0
            self.classifier = pipeline(
                "sentiment-analysis", 
                model="wonrax/phobert-base-vietnamese-sentiment",
                tokenizer="wonrax/phobert-base-vietnamese-sentiment",
                device=-1, 
                truncation=True, 
                max_length=256
            )
            logger.info("✅ Sentiment Model Loaded Successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def analyze_batch_hybrid(self, contents, ratings):
        """
        Xử lý thông minh kết hợp cả Text và Rating.
        Input: 
           - contents: List[str] (Nội dung review)
           - ratings: List[int] (Số sao 1-5)
        Output: 
           - List[tuple]: [(label, score), ...]
        """
        results = []
        
        # 1. Lọc ra các review có nội dung đủ dài để chạy AI
        indices_to_analyze = []
        texts_to_analyze = []
        
        for i, text in enumerate(contents):
            # Text phải tồn tại và dài hơn 3 ký tự (tránh "ok", "k", "...")
            if text and len(str(text).strip()) > 3:
                indices_to_analyze.append(i)
                texts_to_analyze.append(str(text))
            
        # 2. Chạy AI Batch cho các text hợp lệ
        ai_outputs = []
        if texts_to_analyze:
            try:
                # batch_size=16 giúp tối ưu tốc độ xử lý
                ai_outputs = self.classifier(texts_to_analyze, batch_size=16)
            except Exception as e:
                logger.error(f"⚠️ AI Inference Error: {e}")
                # Fallback về neutral nếu AI lỗi
                ai_outputs = [{'label': 'NEU', 'score': 0.5}] * len(texts_to_analyze)

        # 3. Ghép kết quả lại (AI hoặc Rating)
        ai_cursor = 0
        
        for i in range(len(contents)):
            rating = ratings[i]
            
            if i in indices_to_analyze:
                # TRƯỜNG HỢP A: Có comment -> Dùng kết quả AI
                hf_res = ai_outputs[ai_cursor]
                label, score = self._convert_hf_to_db_format(hf_res)
                
                # Logic an toàn: Nếu AI không chắc chắn (< 60%), tham khảo thêm rating
                # (Optional: Có thể bỏ qua đoạn này nếu tin tưởng model hoàn toàn)
                if abs(score) < 0.6:
                     # Nếu AI mơ hồ mà Rating cực đoan (1 hoặc 5 sao), ưu tiên Rating
                     if rating == 1 or rating == 5:
                         label, score = self._convert_rating_to_score(rating)
                
                results.append((label, score))
                ai_cursor += 1
            else:
                # TRƯỜNG HỢP B: Không comment hoặc quá ngắn -> Dùng Rating
                label, score = self._convert_rating_to_score(rating)
                results.append((label, score))
                
        return results

    def _convert_hf_to_db_format(self, hf_result):
        """Helper: Convert kết quả PhoBERT sang điểm số DB"""
        label = hf_result['label']
        conf = hf_result['score'] # Độ tin cậy (0.5 - 1.0)
        
        # Model wonrax trả về: POS, NEG, NEU
        if label == 'POS':
            return 'positive', conf
        elif label == 'NEG':
            return 'negative', -conf
        else:
            return 'neutral', 0.0

    def _convert_rating_to_score(self, rating):
        """Helper: Convert Rating 1-5 sang điểm số giả lập"""
        if rating == 5:
            return 'positive', 1.0     # Tuyệt vời
        elif rating == 4:
            return 'positive', 0.7     # Tốt
        elif rating == 3:
            return 'neutral', 0.0      # Bình thường
        elif rating == 2:
            return 'negative', -0.5    # Tệ
        else: # rating 1
            return 'negative', -1.0    # Rất tệ