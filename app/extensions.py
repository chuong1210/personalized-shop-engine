# app/extensions.py
import logging
from kafka import KafkaProducer
from sentence_transformers import SentenceTransformer
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ShopServiceAI")

# Khai báo các biến global (sẽ được init trong __init__.py)
service = None
db = None
redis_client = None
clip_model = None
producer = None

def init_kafka_producer(bootstrap_servers):
    global producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks=1,
            linger_ms=10
        )
        logger.info("Kafka Producer initialized")
    except Exception as e:
        logger.error(f"Failed to init Kafka: {e}")

def init_clip_model():
    global clip_model
    logger.info("Loading CLIP model...")
    clip_model = SentenceTransformer('clip-ViT-B-32')
    logger.info("CLIP model loaded")