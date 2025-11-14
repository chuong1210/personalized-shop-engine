"""
scheduler.py - Schedule and run batch jobs
"""

import schedule
import time
import yaml
import logging
import redis
import os

from database import Database, MySQLDatabase
from recommend_service import RecommendationService
from batch_jobs import BatchJobs

# Create logs directory if not exists
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main scheduler function"""
    logger.info("=" * 60)
    logger.info("RECOMMENDATION ENGINE SCHEDULER STARTING")
    logger.info("=" * 60)
    
    # Load config
    config = load_config()
    
    # Initialize database
    db = Database(config['database'])
    mysql_db = MySQLDatabase(config['mysql_database'])
    
    # Initialize Redis
    redis_client = redis.Redis(**config['redis'], decode_responses=True)
    
    # Initialize service
    service = RecommendationService(db, redis_client, config)
    
    # Initialize batch jobs
    batch_jobs = BatchJobs(db, service)
    
    # Get schedule config
    schedule_config = config.get('schedule', {})
    
    # ========================================================================
    # DAILY JOBS
    # ========================================================================
    
    # Update product metrics (daily at 2:00 AM)
    update_metrics_time = schedule_config.get('update_metrics', '02:00')
    schedule.every().day.at(update_metrics_time).do(
        batch_jobs.update_product_metrics
    )
    logger.info(f"Scheduled: Update product metrics daily at {update_metrics_time}")
    
    # Update user profiles (daily at 3:00 AM)
    update_profiles_time = schedule_config.get('update_profiles', '03:00')
    schedule.every().day.at(update_profiles_time).do(
        batch_jobs.update_user_profiles
    )
    logger.info(f"Scheduled: Update user profiles daily at {update_profiles_time}")
    
    # Refresh materialized views (daily at 3:30 AM)
    schedule.every().day.at("03:30").do(
        batch_jobs.refresh_materialized_views
    )
    logger.info("Scheduled: Refresh materialized views daily at 03:30")
    
    # Sync products from main database (daily at 1:00 AM)
    schedule.every().day.at("01:00").do(
        batch_jobs.sync_products_from_main_db,
        mysql_db
    )
    logger.info("Scheduled: Sync products daily at 01:00")
    
    # ========================================================================
    # WEEKLY JOBS
    # ========================================================================
    
    # Train models (weekly on Sunday at 2:00 AM)
    train_models_time = schedule_config.get('train_models', 'Sunday 02:00')
    day, time_str = train_models_time.split()
    
    schedule_obj = getattr(schedule.every(), day.lower())
    schedule_obj.at(time_str).do(batch_jobs.train_models)
    logger.info(f"Scheduled: Train models weekly on {train_models_time}")
    
    # Cleanup old data (weekly on Sunday at 4:00 AM)
    schedule.every().sunday.at("04:00").do(
        batch_jobs.cleanup_old_data,
        days_to_keep=180
    )
    logger.info("Scheduled: Cleanup old data weekly on Sunday at 04:00")
    
    # ========================================================================
    # RUN SCHEDULER LOOP
    # ========================================================================
    
    logger.info("=" * 60)
    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    # Run pending jobs on startup (optional)
    # batch_jobs.run_all_daily_jobs()
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise
    finally:
        db.close()
        mysql_db.close()
        logger.info("Scheduler shutdown complete")


if __name__ == '__main__':
    main()