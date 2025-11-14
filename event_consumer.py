
import json
import logging
from typing import Dict, Any
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from database import Database

logger = logging.getLogger(__name__)


class EventConsumer:
    """
    Consumer để xử lý user events từ Kafka
    """
    
    def __init__(self, kafka_config: dict, db: Database):
        """
        Initialize event consumer
        
        Args:
            kafka_config: Kafka configuration
            db: Database instance
        """
        self.db = db
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            'user-events',  # Topic name
            bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
            group_id=kafka_config.get('consumer_group', 'recommendation-engine'),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
        
        logger.info("EventConsumer initialized")
    
    def start(self):
        """
        Start consuming events
        """
        logger.info("=" * 60)
        logger.info("EVENT CONSUMER STARTED")
        logger.info("Listening for events on topic: user-events")
        logger.info("=" * 60)
        
        try:
            for message in self.consumer:
                self._process_message(message)
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            self.consumer.close()
            logger.info("Consumer closed")
    
    def _process_message(self, message):
        """
        Process a single Kafka message
        
        Args:
            message: Kafka message
        """
        try:
            event = message.value
            event_type = event.get('event_type')
            
            logger.info(f"Processing event: {event_type} from user {event.get('user_id')}")
            
            # Route to appropriate handler
            if event_type == 'product_view':
                self._handle_product_view(event)
            elif event_type == 'product_search':
                self._handle_search(event)
            elif event_type == 'cart_add':
                self._handle_cart_add(event)
            elif event_type == 'cart_remove':
                self._handle_cart_remove(event)
            elif event_type == 'purchase':
                self._handle_purchase(event)
            elif event_type == 'recommendation_click':
                self._handle_recommendation_click(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Don't crash the consumer, continue processing
    
    def _handle_product_view(self, event: Dict[str, Any]):
        """
        Handle product view event
        
        Insert into user_interactions table with action_type='view'
        """
        try:
            self.db.execute("""
                INSERT INTO user_interactions 
                (user_id, product_id, shop_id, action_type, score, metadata, created_at)
                VALUES (%s, %s, %s, 'view', 1.0, %s, %s)
            """, (
                event['user_id'],
                event['product_id'],
                event['shop_id'],
                json.dumps({
                    'duration': event.get('duration', 0),
                    'source': event.get('source'),
                    'device_type': event.get('device_type'),
                    'session_id': event.get('session_id')
                }),
                self._parse_timestamp(event.get('timestamp'))
            ))
            
            logger.debug(f"Product view saved: {event['user_id']} -> {event['product_id']}")
            
        except Exception as e:
            logger.error(f"Failed to save product view: {e}")
    
    def _handle_search(self, event: Dict[str, Any]):
        """
        Handle search event
        
        Note: Bạn có thể lưu search history nếu cần,
        nhưng để đơn giản, ta chỉ log hoặc bỏ qua
        """
        try:
            # Option 1: Just log (đơn giản nhất)
            logger.info(f"Search: {event['user_id']} searched for '{event['query']}' "
                       f"({event.get('result_count', 0)} results)")
            
            # Option 2: Save to database (nếu cần analyze search behavior)
            # Bạn có thể tạo bảng user_searches riêng hoặc lưu vào metadata
            
        except Exception as e:
            logger.error(f"Failed to process search event: {e}")
    
    def _handle_cart_add(self, event: Dict[str, Any]):
        """
        Handle cart add event
        
        Insert into user_interactions with action_type='cart_add'
        """
        try:
            self.db.execute("""
                INSERT INTO user_interactions 
                (user_id, product_id, shop_id, action_type, score, price, quantity, created_at)
                VALUES (%s, %s, %s, 'cart_add', 3.0, %s, %s, %s)
            """, (
                event['user_id'],
                event['product_id'],
                event['shop_id'],
                event.get('price', 0),
                event.get('quantity', 1),
                self._parse_timestamp(event.get('timestamp'))
            ))
            
            logger.debug(f"Cart add saved: {event['user_id']} -> {event['product_id']}")
            
        except Exception as e:
            logger.error(f"Failed to save cart add: {e}")
    
    def _handle_cart_remove(self, event: Dict[str, Any]):
        """
        Handle cart remove event
        
        Insert with action_type='cart_remove' and negative score
        """
        try:
            self.db.execute("""
                INSERT INTO user_interactions 
                (user_id, product_id, shop_id, action_type, score, created_at)
                VALUES (%s, %s, %s, 'cart_remove', -2.0, %s)
            """, (
                event['user_id'],
                event['product_id'],
                event.get('shop_id', ''),  # Shop ID might not be in remove event
                self._parse_timestamp(event.get('timestamp'))
            ))
            
            logger.debug(f"Cart remove saved: {event['user_id']} -> {event['product_id']}")
            
        except Exception as e:
            logger.error(f"Failed to save cart remove: {e}")
    
    def _handle_purchase(self, event: Dict[str, Any]):
        """
        Handle purchase event
        
        Insert into user_interactions with action_type='purchase' and high score
        """
        try:
            self.db.execute("""
                INSERT INTO user_interactions 
                (user_id, product_id, shop_id, action_type, score, price, quantity, 
                 metadata, created_at)
                VALUES (%s, %s, %s, 'purchase', 10.0, %s, %s, %s, %s)
            """, (
                event['user_id'],
                event['product_id'],
                event['shop_id'],
                event.get('price', 0),
                event.get('quantity', 1),
                json.dumps({
                    'order_id': event.get('order_id'),
                    'sku_id': event.get('sku_id')
                }),
                self._parse_timestamp(event.get('timestamp'))
            ))
            
            logger.debug(f"Purchase saved: {event['user_id']} -> {event['product_id']}")
            
            # Update user profile asynchronously
            self._update_user_profile(event['user_id'])
            
        except Exception as e:
            logger.error(f"Failed to save purchase: {e}")
    
    def _handle_recommendation_click(self, event: Dict[str, Any]):
        """
        Handle recommendation click event
        
        Update recommendation_logs table
        """
        try:
            # Update the most recent impression for this user-product combo
            self.db.execute("""
                UPDATE recommendation_logs
                SET clicked_at = %s
                WHERE user_id = %s 
                AND product_id = %s 
                AND rec_type = %s
                AND shown_at >= NOW() - INTERVAL '1 hour'
                AND clicked_at IS NULL
                ORDER BY shown_at DESC
                LIMIT 1
            """, (
                self._parse_timestamp(event.get('timestamp')),
                event['user_id'],
                event['product_id'],
                event.get('rec_type', 'unknown')
            ))
            
            logger.debug(f"Recommendation click tracked: {event['user_id']} -> {event['product_id']}")
            
        except Exception as e:
            logger.error(f"Failed to track recommendation click: {e}")
    
    def _update_user_profile(self, user_id: str):
        """
        Update user profile after purchase
        
        Args:
            user_id: User ID
        """
        try:
            # Recalculate user profile metrics
            self.db.execute("""
                INSERT INTO user_profiles (
                    user_id, 
                    total_orders, 
                    total_spent, 
                    avg_order_value,
                    last_purchase_at,
                    profile_updated_at
                )
                SELECT 
                    user_id,
                    COUNT(DISTINCT DATE(created_at)) as total_orders,
                    SUM(price * quantity) as total_spent,
                    AVG(price * quantity) as avg_order_value,
                    MAX(created_at) as last_purchase_at,
                    NOW() as profile_updated_at
                FROM user_interactions
                WHERE user_id = %s
                AND action_type = 'purchase'
                GROUP BY user_id
                ON CONFLICT (user_id) DO UPDATE SET
                    total_orders = EXCLUDED.total_orders,
                    total_spent = EXCLUDED.total_spent,
                    avg_order_value = EXCLUDED.avg_order_value,
                    last_purchase_at = EXCLUDED.last_purchase_at,
                    profile_updated_at = NOW()
            """, (user_id,))
            
            logger.debug(f"User profile updated: {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    def _parse_timestamp(self, timestamp_str: Any) -> datetime:
        """
        Parse timestamp from event
        
        Args:
            timestamp_str: Timestamp string or datetime
            
        Returns:
            datetime object
        """
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        
        if isinstance(timestamp_str, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                pass
        
        # Default to now
        return datetime.now()


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """
    Main function to run event consumer
    """
    import yaml
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/event_consumer.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = Database(config['database'])
    
    # Initialize consumer
    consumer = EventConsumer(config['kafka'], db)
    
    # Start consuming
    try:
        consumer.start()
    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    finally:
        db.close()


if __name__ == '__main__':
    main()