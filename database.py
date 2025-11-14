"""
database.py - Database connection manager (Fixed for type errors and encoding)
"""

import psycopg2
import psycopg2.extras
import pandas as pd
from typing import Optional, Any, List, Tuple, Dict
import logging
import os  # Để detect Windows và set encoding

logger = logging.getLogger(__name__)

# Fix encoding cho Windows (nếu chạy console)
if os.name == 'nt':  # Windows
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')


class Database:
    """PostgreSQL database connection manager (Fixed)"""
    
    REQUIRED_KEYS = {'host', 'database', 'user', 'password', 'port'}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connection with config validation
        
        Args:
            config: Database configuration dict
                    e.g., {'host': 'localhost', 'database': 'ecommerce_ai_db', 'user': 'postgres', 'password': '101204', 'port': 5432}
        """
        self.config = self._validate_config(config)
        self.conn = None
        logger.info("Database config validated successfully")
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize config types"""
        validated = config.copy()
        
        # Check required keys
        missing = self.REQUIRED_KEYS - set(validated.keys())
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        
        # Normalize types
        validated['host'] = str(validated['host']) if validated['host'] is not None else 'localhost'
        validated['database'] = str(validated['database'])
        validated['user'] = str(validated['user'])
        validated['password'] = str(validated['password']) if validated['password'] is not None else ''
        validated['port'] = int(validated['port'])  # Ensure int for port
        
        # Log sanitized config (ẩn password)
        log_config = validated.copy()
        log_config['password'] = '***HIDDEN***'
        logger.debug(f"Validated config: {log_config}")
        
        return validated
        
    def connect(self) -> psycopg2.extensions.connection:
        """
        Get or create database connection
        
        Returns:
            PostgreSQL connection object
        """
        if not self.conn or self.conn.closed:
            try:
                # Use explicit params để tránh **kwargs issues
                self.conn = psycopg2.connect(
                    host=self.config['host'],
                    database=self.config['database'],
                    user=self.config['user'],
                    password=self.config['password'],
                    port=self.config['port']
                )
                # Set autocommit=False mặc định
                self.conn.autocommit = False
                logger.info(f"Database connected to {self.config['database']}@{self.config['host']}:{self.config['port']}")
            except Exception as e:
                logger.error(f"[ERROR] Database connection failed: {e}")
                # Raise để caller handle
                raise psycopg2.Error(f"Connection failed - Check config: {e}")
        return self.conn
    
    def query(self, sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Execute SELECT query and return DataFrame
        
        Args:
            sql: SQL query string
            params: Query parameters tuple
            
        Returns:
            Pandas DataFrame with query results
        """
        conn = self.connect()
        try:
            df = pd.read_sql(sql, conn, params=params)
            logger.debug(f"Query executed: {sql[:100]}... (returned {len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"[ERROR] Query failed: {e}")
            conn.rollback()
            raise
    
    def execute(self, sql: str, params: Optional[Tuple] = None, commit: bool = True):
        """
        Execute INSERT/UPDATE/DELETE query
        
        Args:
            sql: SQL query string
            params: Query parameters tuple
            commit: Whether to commit transaction
        """
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            if commit:
                conn.commit()
            logger.debug(f"Execute successful: {sql[:100]}...")
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Execute failed: {e}")
            raise
    
    def execute_many(self, sql: str, data: List[Tuple], commit: bool = True):
        """
        Execute batch INSERT/UPDATE
        
        Args:
            sql: SQL query string
            data: List of parameter tuples
            commit: Whether to commit transaction
        """
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, data)
            if commit:
                conn.commit()
            logger.info(f"Batch execute successful: {len(data)} rows")
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Batch execute failed: {e}")
            raise
    
    def fetchone(self, sql: str, params: Optional[Tuple] = None) -> Optional[Tuple]:
        """
        Fetch single row
        
        Args:
            sql: SQL query string
            params: Query parameters tuple
            
        Returns:
            Single row tuple or None
        """
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = cur.fetchone()
            return result
        except Exception as e:
            logger.error(f"[ERROR] Fetchone failed: {e}")
            conn.rollback()
            raise
    
    def fetchall(self, sql: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        Fetch all rows
        
        Args:
            sql: SQL query string
            params: Query parameters tuple
            
        Returns:
            List of row tuples
        """
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results = cur.fetchall()
            return results
        except Exception as e:
            logger.error(f"[ERROR] Fetchall failed: {e}")
            conn.rollback()
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class MySQLDatabase:
    """MySQL database connection (for main ecommerce database)"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MySQL connection with validation"""
        try:
            import mysql.connector as mysql
            self.mysql = mysql
            self.config = self._validate_mysql_config(config)
            self.conn = None
            logger.info("MySQL config validated successfully")
        except ImportError:
            logger.error("[ERROR] mysql-connector-python not installed. Run: pip install mysql-connector-python")
            raise
    
    def _validate_mysql_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MySQL config"""
        required = {'host', 'database', 'user', 'password', 'port'}
        missing = required - set(config.keys())
        if missing:
            raise ValueError(f"Missing MySQL config keys: {missing}")
        
        validated = config.copy()
        validated['host'] = str(validated['host']) if validated['host'] is not None else 'localhost'
        validated['database'] = str(validated['database'])
        validated['user'] = str(validated['user'])
        validated['password'] = str(validated['password']) if validated['password'] is not None else ''
        validated['port'] = int(validated['port'])
        
        log_config = validated.copy()
        log_config['password'] = '***HIDDEN***'
        logger.debug(f"MySQL validated config: {log_config}")
        
        return validated
    
    def connect(self):
            """Connect to MySQL with timeout"""

            if not self.conn or not self.conn.is_connected():
                logger.info(f"MySQL 2connected to {self.config['database']}@{self.config['host']}:{self.config['port']}")

                try:
                    self.conn = self.mysql.connect(
                    host='127.0.0.1',  # Force IP
                    port=3306,
                    database=self.config['database'],
                    user=self.config['user'],
                    password=self.config['password'],
                    charset='utf8mb4',  # Fix charset
                    use_unicode=True,
                    connect_timeout=5,  # 5s timeout
                    autocommit=True
                )
                    logger.info(f"MySQL connected to {self.config['database']}@{self.config['host']}:{self.config['port']}")
                except self.mysql.Error as e:  # Specific mysql error
                    error_msg = f"[ERROR] MySQL connection failed: errno={e.errno}, msg={e}"
                    logger.error(error_msg)  # Không có end=''
                    raise  # Re-raise để caller handle
                except Exception as e:
                    logger.error(f"[ERROR] Unexpected MySQL connect error: {type(e).__name__}: {str(e)}")
                    raise
            return self.conn
    def query(self, sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        conn = self.connect()
        try:
            df = pd.read_sql(sql, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"[ERROR] MySQL query failed: {e}")
            raise
        finally:
            if conn and not conn.is_connected():
                conn.close()
    
    def close(self):
        """Close connection"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            logger.info("MySQL connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()