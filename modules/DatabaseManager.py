"""
Database Manager for storing face entries and logs using PostgreSQL
"""

import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json
import pickle

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database interface for storing face events and metadata in PostgreSQL
    """
    
    def __init__(self, config: Dict):
        self.full_config = config
        self.config = config['database']
        self.connection = None
        self.cursor = None
        
        if self.config['type'] != 'postgresql':
            logger.warning(f"Database type '{self.config['type']}' not supported. Defaulting to PostgreSQL.")
            
        self._init_postgresql()
        self._create_tables()
        logger.info("Database initialized (PostgreSQL)")
    
    def _init_postgresql(self):
        try:
            pg_config = self.config['postgresql']
            self.connection = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['database'],
                user=pg_config['user'],
                password=pg_config['password']
            )
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise ConnectionError(f"Could not connect to PostgreSQL: {e}")
            
    def _execute(self, query: str, params: tuple = ()):
        # Ensure query uses %s for PostgreSQL
        query = query.replace('?', '%s')
        self.cursor.execute(query, params)
        return self.cursor
        
    def _create_tables(self):
        # Using PostgreSQL specific data types
        self._execute("""
            CREATE TABLE IF NOT EXISTS face_entries (
                id SERIAL PRIMARY KEY,
                face_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                image_path TEXT,
                confidence FLOAT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._execute("""
            CREATE TABLE IF NOT EXISTS unique_visitors (
                face_id TEXT PRIMARY KEY,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                total_entries INTEGER DEFAULT 1,
                total_exits INTEGER DEFAULT 1,
                metadata JSONB
            )
        """)
        
        self._execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                face_id TEXT PRIMARY KEY,
                embedding BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.info("Database tables created/verified")
    
    def log_event(self, face_id: str, event_type: str, image_path: str = None, 
                  confidence: float = None, metadata: Dict = None):
        timestamp = datetime.now()
        
        self._execute("""
            INSERT INTO face_entries (face_id, event_type, timestamp, image_path, confidence, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (face_id, event_type, timestamp, image_path, confidence, json.dumps(metadata) if metadata else None))
        
        logger.info(f"Logged {event_type} event for face {face_id}")
        
        if event_type in ('entry', 'exit'):
            self._update_unique_visitor(face_id, timestamp, event_type, confidence, metadata)
    
    def _update_unique_visitor(self, face_id: str, timestamp: datetime, event_type: str,
                               confidence: float = None, metadata: Dict = None):
        self._execute("""
            SELECT first_seen, total_entries, total_exits FROM unique_visitors WHERE face_id = %s
        """, (face_id,))
        
        result = self.cursor.fetchone()
        
        metadata_json = json.dumps(metadata) if metadata else None

        if result:
            first_seen, total_entries, total_exits = result
            if event_type == 'entry': total_entries += 1
            elif event_type == 'exit': total_exits += 1
            
            self._execute("""
                UPDATE unique_visitors 
                SET last_seen = %s, total_entries = %s, total_exits = %s, metadata = %s
                WHERE face_id = %s
            """, (timestamp, total_entries, total_exits, metadata_json, face_id))
        else:
            entries = 1 if event_type == 'entry' else 0
            exits = 1 if event_type == 'exit' else 0
            self._execute("""
                INSERT INTO unique_visitors (face_id, first_seen, last_seen, total_entries, total_exits, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (face_id, timestamp, timestamp, entries, exits, metadata_json))
    
    def get_unique_visitor_count(self) -> int:
        self._execute("SELECT COUNT(*) FROM unique_visitors")
        count = self.cursor.fetchone()[0]
        return count
    
    def get_visitor_history(self, face_id: str = None) -> List[Dict]:
        if self.cursor is None:
            return []
            
        if face_id:
            self._execute("""
                SELECT * FROM face_entries WHERE face_id = %s ORDER BY timestamp DESC
            """, (face_id,))
        else:
            self._execute("""
                SELECT * FROM face_entries ORDER BY timestamp DESC LIMIT 100
            """)
        
        rows = self.cursor.fetchall()
        columns = [description[0] for description in self.cursor.description]
        
        results = []
        for row in rows:
            result = dict(zip(columns, row))
            meta = result.get('metadata')
            if meta is not None:
                if isinstance(meta, (str, bytes, bytearray)):
                    result['metadata'] = json.loads(meta)
                else:
                    # Already a dict/list (JSONB automatically parsed by some drivers)
                    result['metadata'] = meta
            results.append(result)
        
        return results
    
    def save_embedding(self, face_id: str, embedding: Any):
        embedding_blob = pickle.dumps(embedding)
        embedding_binary = psycopg2.Binary(embedding_blob)
        
        self._execute("""
            INSERT INTO face_embeddings (face_id, embedding, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (face_id) DO UPDATE 
            SET embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at
        """, (face_id, embedding_binary, datetime.now()))
        
        logger.debug(f"Saved embedding for face {face_id}")
    
    def load_embedding(self, face_id: str) -> Optional[Any]:
        self._execute("""
            SELECT embedding FROM face_embeddings WHERE face_id = %s
        """, (face_id,))
        
        result = self.cursor.fetchone()
        if result:
            embedding_blob = result[0]
            # psycopg2 already handles the conversion from memoryview/bytea
            return pickle.loads(bytes(embedding_blob))
        
        return None
    
    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")