"""Persistent memory service for tracking last update time."""

import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

from backend.config import config


class MemoryService:
    """Service for managing persistent memory of last update time."""
    
    def __init__(self):
        self.memory_file_path = config.MEMORY_FILE_PATH
        
    
    def get_last_update_time(self) -> datetime:
        """
        Load last update time from file with database fallback.
        
        Returns:
            datetime: Last update time, using fallback mechanisms if needed
        """
        # Try to load from file first
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, "r") as f:
                    last_update_str = f.read().strip()
                    last_update = datetime.fromisoformat(last_update_str)
                    print(f"Loaded last update time from file: {last_update}")
                    return last_update
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading last update time from file: {e}")
        
        # Fallback: Try to get from database (latest digest_date)
        try:
            last_update = self._get_last_update_from_database()
            if last_update:
                print(f"Using database fallback for last update time: {last_update}")
                # Save to file for future use
                self.save_last_update_time(last_update)
                return last_update
        except Exception as e:
            print(f"Error getting last update time from database: {e}")
        
        # Final fallback - default to January 1st, 2025 (start of this year)
        default_time = datetime(2025, 1, 1, 0, 0, 0)
        print(f"Using default fallback time: {default_time}")
        return default_time
    
    def save_last_update_time(self, update_time: datetime) -> None:
        """
        Save last update time to file.
        
        Args:
            update_time: The timestamp to save
        """
        try:
            with open(self.memory_file_path, "w") as f:
                f.write(update_time.isoformat())
            print(f"Saved last update time: {update_time}")
        except Exception as e:
            print(f"Error saving last update time: {e}")
    
    def update_last_update_time(self) -> None:
        """Update last update time to current time."""
        self.save_last_update_time(datetime.now())
    
    def _get_last_update_from_database(self) -> Optional[datetime]:
        """
        Get last update time from database by finding the latest digest_date.
        
        Returns:
            datetime: Latest digest date from database, or None if no data
        """
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                cursor_factory=RealDictCursor
            )
            
            cursor = conn.cursor()
            
            # Get the latest digest_date from the articles table
            cursor.execute(f"""
                SELECT MAX(digest_date) as latest_digest_date
                FROM {config.VECTOR_TABLE_NAME}
                WHERE digest_date IS NOT NULL
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result['latest_digest_date']:
                # Convert date to datetime (start of next day for inclusive search)
                latest_date = result['latest_digest_date']
                # Return the day after the latest digest date so we fetch from the next day forward
                return datetime.combine(latest_date, datetime.min.time()) + timedelta(days=1)
            
            return None
            
        except Exception as e:
            print(f"Error querying database for last update time: {e}")
            return None


# Global memory service instance
memory_service = MemoryService()
