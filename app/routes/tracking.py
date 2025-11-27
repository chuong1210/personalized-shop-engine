# app/routes/tracking.py
from flask import Blueprint, request, jsonify
from app.extensions import producer, service, logger
import time

tracking_bp = Blueprint('tracking', __name__)

@tracking_bp.route('/api/events', methods=['POST'])
def track_event():
    try:
        data = request.json
        if not data.get('user_id') or not data.get('event_type'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        data['server_timestamp'] = time.time()
        
        if producer:
            producer.send('user-events', value=data)
            return jsonify({'status': 'queued'}), 200
        else:
            return jsonify({'error': 'Kafka unavailable'}), 503

    except Exception as e:
        logger.error(f"Tracking error: {e}")
        return jsonify({'status': 'error', 'msg': str(e)}), 500

@tracking_bp.route('/api/track/click', methods=['POST'])
def track_click():
    try:
        data = request.json
        service.track_click(data['user_id'], data['product_id'], data['rec_type'])
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Track click failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@tracking_bp.route('/api/track/purchase', methods=['POST'])
def track_purchase():
    try:
        data = request.json
        service.track_purchase(data['user_id'], data['product_id'], data['amount'])
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Track purchase failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500