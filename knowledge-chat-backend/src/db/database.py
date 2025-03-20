import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class ChatDatabase:
    """Database manager for chat conversations"""
    
    def __init__(self, db_path=None):
        """Initialize the database connection"""
        if db_path is None:
            # Default to a database in the parent directory
            db_path = str(Path(__file__).resolve().parent.parent.parent / "chat_database.db")
        
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize the database
        self.init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize the database with the conversations table"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create conversations table with a clean, simple schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    question TEXT,
                    answer TEXT,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on thread_id separately
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_id ON conversations(thread_id)
            """)
            
            # Create checkpoints table if it doesn't exist (for langgraph)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT PRIMARY KEY,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized with conversations table")
    
    def save_conversation(self, conversation_data):
        """
        Save a conversation to the database.
        
        Args:
            conversation_data: Dictionary containing conversation data
        """
        try:
            # Extract data from the conversation_data dictionary
            conversation_id = conversation_data.get("conversation_id")
            question = conversation_data.get("question")
            answer = conversation_data.get("answer")
            thread_id = conversation_data.get("thread_id", conversation_id)  # Default to conversation_id if thread_id not provided
            metadata = conversation_data.get("metadata", {})
            
            # Log the save operation
            self.logger.info(f"Saving conversation {conversation_id} with thread {thread_id}")
            
            # Create a timestamp if not provided
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().isoformat()
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Check if this conversation already exists
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if the conversation_id already exists
                cursor.execute(
                    "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update the existing conversation
                    self.logger.info(f"Updating existing conversation {conversation_id}")
                    cursor.execute(
                        """
                        UPDATE conversations 
                        SET thread_id = ?, question = ?, answer = ?, metadata = ?
                        WHERE conversation_id = ?
                        """,
                        (thread_id, question, answer, metadata_json, conversation_id)
                    )
                else:
                    # Insert a new conversation
                    self.logger.info(f"Inserting new conversation {conversation_id}")
                    cursor.execute(
                        """
                        INSERT INTO conversations 
                        (conversation_id, thread_id, question, answer, metadata)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (conversation_id, thread_id, question, answer, metadata_json)
                    )
                
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving conversation: {str(e)}", exc_info=True)
            return False
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        conversation_id,
                        thread_id,
                        question,
                        answer,
                        metadata,
                        timestamp
                    FROM conversations
                    WHERE conversation_id = ?
                ''', (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    # Convert row to dictionary
                    conversation = {
                        "conversation_id": row[0],
                        "thread_id": row[1],
                        "question": row[2],
                        "answer": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                        "timestamp": row[5]
                    }
                    return conversation
                return None
        except Exception as e:
            self.logger.error(f"Error getting conversation: {str(e)}", exc_info=True)
            return None
    
    def get_conversations_by_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all conversations in a thread"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        conversation_id,
                        thread_id,
                        question,
                        answer,
                        metadata,
                        timestamp
                    FROM conversations
                    WHERE thread_id = ?
                    ORDER BY timestamp ASC
                ''', (thread_id,))
                
                rows = cursor.fetchall()
                conversations = []
                
                for row in rows:
                    conversation = {
                        "conversation_id": row[0],
                        "thread_id": row[1],
                        "question": row[2],
                        "answer": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                        "timestamp": row[5]
                    }
                    conversations.append(conversation)
                
                return conversations
        except Exception as e:
            self.logger.error(f"Error getting conversations by thread: {str(e)}", exc_info=True)
            return []
    
    def get_thread_ids(self) -> List[str]:
        """Get all unique thread IDs ordered by most recent activity"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT thread_id, MAX(timestamp) as last_activity
                    FROM conversations
                    GROUP BY thread_id
                    ORDER BY last_activity DESC
                ''')
                
                rows = cursor.fetchall()
                return [row[0] for row in rows if row[0]]
        except Exception as e:
            self.logger.error(f"Error getting thread IDs: {str(e)}", exc_info=True)
            return []
    
    def get_thread_summary(self, thread_id: str) -> Dict[str, Any]:
        """Get a summary of a thread"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get the count of conversations in the thread
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM conversations
                    WHERE thread_id = ?
                ''', (thread_id,))
                count = cursor.fetchone()[0]
                
                # Get the most recent conversation
                cursor.execute('''
                    SELECT 
                        conversation_id,
                        question,
                        timestamp
                    FROM conversations
                    WHERE thread_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (thread_id,))
                
                latest = cursor.fetchone()
                if latest:
                    return {
                        "thread_id": thread_id,
                        "conversation_count": count,
                        "latest_conversation_id": latest[0],
                        "latest_question": latest[1],
                        "latest_timestamp": latest[2]
                    }
                return {
                    "thread_id": thread_id,
                    "conversation_count": 0
                }
        except Exception as e:
            self.logger.error(f"Error getting thread summary: {str(e)}", exc_info=True)
            return {"thread_id": thread_id, "error": str(e)}
    
    def get_all_threads(self) -> List[Dict[str, Any]]:
        """Get summaries of all threads"""
        try:
            thread_ids = self.get_thread_ids()
            return [self.get_thread_summary(thread_id) for thread_id in thread_ids]
        except Exception as e:
            self.logger.error(f"Error getting all threads: {str(e)}", exc_info=True)
            return []
    
    def ensure_parent_checkpoint_column(self):
        """Ensure the parent_checkpoint_id column exists in the checkpoints table"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM pragma_table_info('checkpoints') 
                    WHERE name='parent_checkpoint_id'
                """)
                if cursor.fetchone()[0] == 0:
                    cursor.execute("""
                        ALTER TABLE checkpoints 
                        ADD COLUMN parent_checkpoint_id TEXT
                    """)
                    conn.commit()
                    self.logger.info("Added parent_checkpoint_id column to checkpoints table")
        except Exception as e:
            self.logger.error(f"Error ensuring parent_checkpoint_id column: {str(e)}", exc_info=True)
    
    # Add compatibility methods for backward compatibility
    
    def ensure_feedback_columns(self):
        """Compatibility method - no longer needed with new schema"""
        self.logger.info("ensure_feedback_columns called (no action needed)")
        pass
    
    def ensure_cleared_column(self):
        """Compatibility method - no longer needed with new schema"""
        self.logger.info("ensure_cleared_column called (no action needed)")
        pass
    
    def ensure_architect_response_column(self):
        """Compatibility method - no longer needed with new schema"""
        self.logger.info("ensure_architect_response_column called (no action needed)")
        pass
    
    def ensure_conversation_id_column(self):
        """Compatibility method - no longer needed with new schema"""
        self.logger.info("ensure_conversation_id_column called (no action needed)")
        pass
    
    def get_all_conversations(self):
        """Compatibility method - redirects to get_all_threads"""
        self.logger.info("get_all_conversations called (redirecting to get_all_threads)")
        return self.get_all_threads()

    def get_chat(self, conversation_id):
        """Compatibility method - redirects to get_conversation"""
        self.logger.info(f"get_chat called for {conversation_id} (redirecting to get_conversation)")
        return self.get_conversation(conversation_id)

    def get_recent_chats(self, limit=10, offset=0):
        """Compatibility method - redirects to get_recent_conversations"""
        self.logger.info(f"get_recent_chats called (redirecting to get_recent_conversations)")
        return self.get_recent_conversations(limit, offset)

    def get_recent_conversations(self, limit=10, offset=0) -> List[Dict[str, Any]]:
        """Get a list of recent conversations across all threads"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        conversation_id,
                        thread_id,
                        question,
                        answer,
                        metadata,
                        timestamp
                    FROM conversations
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                ''', (limit, offset))
                
                rows = cursor.fetchall()
                conversations = []
                
                for row in rows:
                    conversation = {
                        "conversation_id": row[0],
                        "thread_id": row[1],
                        "question": row[2],
                        "answer": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                        "timestamp": row[5]
                    }
                    conversations.append(conversation)
                
                return conversations
        except Exception as e:
            self.logger.error(f"Error getting recent conversations: {str(e)}", exc_info=True)
            return []