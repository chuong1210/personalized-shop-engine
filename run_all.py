"""
run_all.py - Complete setup and training pipeline
Chạy file này để setup toàn bộ hệ thống từ đầu
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/run_all.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(description, command):
    """Run a shell command and log result"""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f" {description} - SUCCESS")
        if result.stdout:
            logger.info(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f" {description} - FAILED")
        logger.error(f"Error: {e.stderr}")
        return False


def run_python_script(description, script_name):
    """Run a Python script"""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}")
    
    try:
        # Import and run
        if script_name == 'migrate_reviews':
            from migrate_reviews import sync_reviews
            sync_reviews()
        elif script_name == 'train':
            from train import main as train_main
            train_main()
        else:
            logger.error(f"Unknown script: {script_name}")
            return False
        
        logger.info(f" {description} - SUCCESS")
        return True
        
    except Exception as e:
        logger.error(f" {description} - FAILED")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main setup and training pipeline"""
    logger.info("="*70)
    logger.info("AI RECOMMENDATION SYSTEM - COMPLETE SETUP")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    steps_status = []
    
    # Step 1: Create directories
    logger.info("Step 1: Creating directories...")
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logger.info(" Directories created")
    steps_status.append(("Create directories", True))
    
    # Step 2: Check database connection
    logger.info("\nStep 2: Checking database connections...")
    try:
        from database import Database, MySQLDatabase
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test PostgreSQL
        pg_db = Database(config['database'])
        pg_db.connect()
        pg_db.fetchone("SELECT 1")
        logger.info(" PostgreSQL connected")
        
        # Test MySQL
        mysql_db = MySQLDatabase(config['mysql_database'])
        mysql_db.connect()
        logger.info(" MySQL connected")
        
        pg_db.close()
        mysql_db.close()
        
        steps_status.append(("Database connection", True))
        
    except Exception as e:
        logger.error(f" Database connection failed: {e}")
        steps_status.append(("Database connection", False))
        return
    
    # Step 3: Add review tables (if not exist)
    logger.info("\nStep 3: Setting up review tables...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        pg_db = Database(config['database'])
        pg_db.connect()
        
        # Check if review table exists
        result = pg_db.fetchone("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'product_reviews'
            )
        """)
        
        if not result[0]:
            logger.info("Creating product_reviews table...")
            with open('add_review_tables.sql', 'r') as f:
                sql = f.read()
            pg_db.execute(sql, commit=True)
            logger.info(" Review tables created")
        else:
            logger.info(" Review tables already exist")
        
        pg_db.close()
        steps_status.append(("Setup review tables", True))
        
    except Exception as e:
        logger.error(f" Setup review tables failed: {e}")
        steps_status.append(("Setup review tables", False))
    
    # Step 4: Migrate review data
    success = run_python_script("Migrate review data from MySQL", "migrate_reviews")
    steps_status.append(("Migrate reviews", success))
    
    if not success:
        logger.warning("Review migration failed, but continuing...")
    
    # Step 5: Train models
    success = run_python_script("Train AI models (CF + CB)", "train")
    steps_status.append(("Train models", success))
    
    if not success:
        logger.error("Model training failed!")
        return
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SETUP SUMMARY")
    logger.info("="*70)
    
    for step_name, status in steps_status:
        status_icon = "" if status else ""
        logger.info(f"{status_icon} {step_name}: {'SUCCESS' if status else 'FAILED'}")
    
    all_success = all(status for _, status in steps_status)
    
    if all_success:
        logger.info("\nALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info("\nNext steps:")
        logger.info("1. Start API server: python api.py")
        logger.info("2. Start scheduler: python scheduler.py")
        logger.info("3. Test API: curl http://localhost:5000/health")
    else:
        logger.warning("\n Some steps failed. Check logs above.")
    
    logger.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nSetup interrupted by user")
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        sys.exit(1)