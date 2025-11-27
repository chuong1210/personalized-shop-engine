# app/__init__.py
from flask import Flask, jsonify
from flask_cors import CORS
import yaml
import redis
import app.extensions as ext
from database import Database
from recommend_service import RecommendationService

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # 1. Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Init Shared Resources
    ext.db = Database(config['database'])
    ext.redis_client = redis.Redis(**config['redis'], decode_responses=True)
    
    # Init Service Core
    ext.service = RecommendationService(ext.db, ext.redis_client, config)
    
    # Init Kafka
    kafka_conf = config.get('kafka', {})
    ext.init_kafka_producer(kafka_conf.get('bootstrap_servers', ['localhost:9092']))
    
    # Init AI Model (CLIP)
    ext.init_clip_model()
    
    # 3. Register Blueprints (Đăng ký routes)
    from app.routes.tracking import tracking_bp
    from app.routes.recommend import recommend_bp
    from app.routes.search import search_bp
    from app.routes.analytics import analytics_bp
    
    app.register_blueprint(tracking_bp)
    app.register_blueprint(recommend_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(analytics_bp)
    
    # 4. Error Handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
    return app