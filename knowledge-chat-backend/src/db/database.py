import sqlite3
from pathlib import Path
from datetime import datetime
import json
from threading import Lock
from typing import List, Dict, Optional
import uuid

class ChatDatabase:
    def __init__(self):
        self.db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Create messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query TEXT,
                    output TEXT,
                    code_context TEXT,
                    technical_details TEXT
                )
            """)
            
            # Create checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)

    def _get_connection(self):
        """Get a new database connection"""
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Initialize the database with proper chat history table"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Create chat_history table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    analysis TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def save_conversation(self, conversation_id: str, data: dict):
        """Save a conversation to the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO conversations 
                (id, query, output, code_context, technical_details, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    data.get('query', ''),
                    json.dumps(data.get('output', {})),
                    json.dumps(data.get('code_context', {})),
                    json.dumps(data.get('technical_details', '')),
                    datetime.now()
                )
            )
            conn.commit()

    def get_conversation(self, conversation_id: str):
        """Retrieve a conversation from the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'created_at': row[1],
                    'query': row[2],
                    'output': json.loads(row[3]),
                    'code_context': json.loads(row[4]),
                    'technical_details': json.loads(row[5])
                }
            return None

    def get_recent_conversations(self, limit: int = 10):
        """Get recent conversations"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            return [{
                'id': row[0],
                'created_at': row[1],
                'query': row[2],
                'output': json.loads(row[3]),
                'code_context': json.loads(row[4]),
                'technical_details': json.loads(row[5])
            } for row in rows]

    def get_conversation_history(self):
        """Fetch all conversations formatted for history display"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, created_at, query, output, code_context FROM conversations ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
            
            return [{
                'id': row[0],
                'timestamp': row[1],
                'query': row[2],
                'output': json.loads(row[3]),
                'code_context': json.loads(row[4])
            } for row in rows]

    def get_conversation_history_with_checkpoints(self):
        """Fetch conversations with their checkpoints, organized by topic"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    id,
                    strftime('%Y-%m-%d %H:%M:%S', created_at) as created_at,
                    query,
                    output,
                    code_context,
                    technical_details
                FROM conversations 
                ORDER BY created_at DESC
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                # Safely parse JSON fields
                try:
                    output_data = json.loads(row[3]) if row[3] and row[3].strip() else {}
                except (json.JSONDecodeError, TypeError):
                    output_data = {"raw_output": row[3]} if row[3] else {}
                
                try:
                    code_context = json.loads(row[4]) if row[4] and row[4].strip() else {}
                except (json.JSONDecodeError, TypeError):
                    code_context = {"raw_context": row[4]} if row[4] else {}
                
                try:
                    technical_details = json.loads(row[5]) if row[5] and row[5].strip() else {}
                except (json.JSONDecodeError, TypeError):
                    technical_details = {"raw_details": row[5]} if row[5] else {}

                # Get checkpoints
                checkpoint_query = """
                    SELECT 
                        checkpoint_id,
                        parent_checkpoint_id,
                        type,
                        checkpoint,
                        metadata
                    FROM checkpoints 
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id
                """
                cursor.execute(checkpoint_query, (row[0],))
                checkpoints = cursor.fetchall()
                
                processed_checkpoints = []
                if checkpoints:
                    for cp in checkpoints:
                        try:
                            checkpoint_data = {
                                'checkpoint_id': cp[0],
                                'parent_id': cp[1],
                                'type': cp[2],
                                'checkpoint': self._safe_load_binary(cp[3]),
                                'metadata': self._safe_load_binary(cp[4])
                            }
                            processed_checkpoints.append(checkpoint_data)
                        except Exception as e:
                            print(f"Error processing checkpoint {cp[0]}: {str(e)}")
                            continue

                history.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'query': row[2],
                    'output': output_data,
                    'code_context': code_context,
                    'technical_details': technical_details,
                    'checkpoints': processed_checkpoints
                })
            
            return history

    def _safe_load_binary(self, binary_data):
        """Safely load binary data that might be JSON"""
        if not binary_data:
            return {}
        
        try:
            if isinstance(binary_data, bytes):
                # Try UTF-8 first
                try:
                    decoded = binary_data.decode('utf-8')
                except UnicodeDecodeError:
                    decoded = binary_data.decode('latin-1')
                
                # Try parsing as JSON
                try:
                    return json.loads(decoded)
                except json.JSONDecodeError:
                    return {"raw_data": decoded}
                
            elif isinstance(binary_data, str):
                try:
                    return json.loads(binary_data)
                except json.JSONDecodeError:
                    return {"raw_data": binary_data}
            else:
                return {"raw_data": str(binary_data)}
            
        except Exception as e:
            print(f"Error loading binary data: {str(e)}")
            return {"error": "Could not parse data", "raw_data": str(binary_data)}

    def update_conversation(self, conversation_id: str, updated_query: str):
        """Update a conversation's query"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET query = ? WHERE id = ?",
                (updated_query, conversation_id)
            )
            conn.commit()

    def create_checkpoint(self, thread_id: str, checkpoint_id: str, checkpoint_type: str, checkpoint_data: str):
        """Create a new checkpoint"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints (
                    thread_id, checkpoint_id, type, checkpoint
                ) VALUES (?, ?, ?, ?)
                """,
                (thread_id, checkpoint_id, checkpoint_type, checkpoint_data)
            )
            conn.commit()

    def update_conversation_response(self, conversation_id: str, updated_response: str):
        """Update a conversation's response"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # If it's a dict or list, convert to JSON string
                if isinstance(updated_response, (dict, list)):
                    updated_response = json.dumps(updated_response)
                # If it's already a string but looks like JSON, validate it
                elif isinstance(updated_response, str) and (
                    updated_response.strip().startswith('{') or 
                    updated_response.strip().startswith('[')
                ):
                    # Validate JSON format
                    json.loads(updated_response)
                
                cursor.execute(
                    "UPDATE conversations SET output = ? WHERE id = ?",
                    (updated_response, conversation_id)
                )
                conn.commit()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}")

    async def save_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Save a message with optional metadata"""
        message_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, session_id, role, content, metadata_json)
            )
        return message_id

    async def get_recent_chats(self, limit: int = 6) -> List[Dict]:
        """Get recent chat sessions grouped by session_id"""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            # Get recent unique sessions
            cursor = conn.execute("""
                SELECT 
                    session_id,
                    MIN(timestamp) as start_time,
                    COUNT(*) as message_count,
                    GROUP_CONCAT(role) as roles,
                    GROUP_CONCAT(content) as contents
                FROM messages 
                GROUP BY session_id 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = cursor.fetchall()
            chat_history = []
            
            for session in sessions:
                # Get all messages for this session
                cursor = conn.execute("""
                    SELECT 
                        role,
                        content,
                        timestamp,
                        analysis
                    FROM messages 
                    WHERE session_id = ? 
                    ORDER BY timestamp ASC
                """, (session['session_id'],))
                
                messages = []
                for row in cursor:
                    messages.append({
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'analysis': json.loads(row['analysis']) if row['analysis'] else None
                    })
                
                # Get the first user message as preview
                preview = next(
                    (msg['content'] for msg in messages if msg['role'] == 'user'),
                    'Empty conversation'
                )
                
                chat_history.append({
                    'session_id': session['session_id'],
                    'start_time': session['start_time'],
                    'preview': preview[:100] + '...' if len(preview) > 100 else preview,
                    'message_count': session['message_count'],
                    'messages': messages
                })
            
            return chat_history 