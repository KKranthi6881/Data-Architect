import sqlite3
from pathlib import Path
from datetime import datetime
import json
from threading import Lock
from typing import List, Dict, Optional, Any, Union
import uuid
import logging

class ChatDatabase:
    def __init__(self):
        self.db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
        self._init_db()
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)

    def _init_db(self):
        """Initialize the database with required tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query TEXT,
                    output TEXT,
                    code_context TEXT,
                    technical_details TEXT,
                    architect_response TEXT,  -- Add architect_response column
                    feedback_status TEXT,
                    feedback_comments TEXT,
                    thread_id TEXT,
                    cleared INTEGER DEFAULT 0
                )
            ''')
            
            # Create checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT PRIMARY KEY,
                    type TEXT,
                    checkpoint TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add architect_response column if it doesn't exist
            cursor.execute('''
                SELECT COUNT(*) FROM pragma_table_info('conversations') 
                WHERE name='architect_response'
            ''')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    ALTER TABLE conversations 
                    ADD COLUMN architect_response TEXT
                ''')
            
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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

    def save_conversation(self, conversation_id: str, data: Dict[str, Any]):
        """Save or update a conversation with proper logging"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Log the incoming data
                self.logger.info(f"Saving conversation {conversation_id} with data: {json.dumps(data, indent=2)}")
                
                # Extract architect response if present
                architect_response = data.get('architect_response')
                if isinstance(architect_response, dict):
                    architect_response = json.dumps(architect_response)
                
                # Ensure technical_details is JSON string
                technical_details = data.get('technical_details')
                if isinstance(technical_details, dict):
                    technical_details = json.dumps(technical_details)
                
                # Create query with all fields
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations (
                        id,
                        created_at,
                        query,
                        output,
                        code_context,
                        technical_details,
                        architect_response,
                        feedback_status,
                        feedback_comments,
                        thread_id,
                        cleared
                    ) VALUES (?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conversation_id,
                    data.get('created_at'),
                    data.get('query', ''),
                    data.get('output', ''),
                    data.get('code_context', '{}'),
                    technical_details,
                    architect_response,
                    data.get('feedback_status'),
                    data.get('feedback_comments'),
                    data.get('thread_id', conversation_id),  # Default to conversation_id if no thread_id
                    data.get('cleared', 0)
                ))
                
                conn.commit()
                self.logger.info(f"Successfully saved conversation {conversation_id}")
                
        except Exception as e:
            self.logger.error(f"Error saving conversation {conversation_id}: {e}", exc_info=True)
            raise

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, created_at, query, output, code_context, 
                           technical_details, architect_response, feedback_status, 
                           feedback_comments, thread_id, cleared
                    FROM conversations 
                    WHERE id = ?
                ''', (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'created_at': row[1],
                        'query': row[2],
                        'output': row[3],
                        'code_context': row[4],
                        'technical_details': row[5],
                        'architect_response': row[6],
                        'feedback_status': row[7],
                        'feedback_comments': row[8],
                        'thread_id': row[9],
                        'cleared': row[10]
                    }
                return None
                
        except Exception as e:
            print(f"Error getting conversation: {e}")
            raise

    def get_recent_conversations(self, limit=10):
        """Get recent conversations with safe handling of missing columns"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build simple query with essential columns
                query = """
                    SELECT 
                        id,
                        created_at,
                        query,
                        output,
                        feedback_status,
                        cleared
                    FROM conversations 
                    WHERE cleared IS NULL OR cleared = 0
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                print(f"Retrieved {len(rows)} rows from database")
                
                # Convert to list of dictionaries
                conversations = []
                for row in rows:
                    try:
                        # Parse output if it exists
                        output = row[3] if len(row) > 3 else None
                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except json.JSONDecodeError:
                                output = output
                        
                        conversation = {
                            "id": row[0],
                            "created_at": row[1],
                            "query": row[2],
                            "output": output,
                            "feedback_status": row[4] if len(row) > 4 else "pending",
                            "cleared": bool(row[5]) if len(row) > 5 else False
                        }
                        conversations.append(conversation)
                        print(f"Processed conversation: {conversation['id']}")
                        
                    except Exception as e:
                        print(f"Error processing conversation row: {e}")
                        continue
                
                return conversations
                
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []

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

    def ensure_feedback_columns(self):
        """Ensure feedback columns exist in the conversations table"""
        try:
            cursor = self._get_connection().cursor()
            
            # Check if feedback_status column exists
            cursor.execute("PRAGMA table_info(conversations)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add feedback_status column if it doesn't exist
            if "feedback_status" not in columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN feedback_status TEXT")
                print("Added feedback_status column to conversations table")
            
            # Add feedback_comments column if it doesn't exist
            if "feedback_comments" not in columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN feedback_comments TEXT")
                print("Added feedback_comments column to conversations table")
            
            self._get_connection().commit()
        except Exception as e:
            print(f"Error ensuring feedback columns: {e}")

    def ensure_cleared_column(self):
        """Ensure cleared column exists in conversations table"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if cleared column exists
                cursor.execute("PRAGMA table_info(conversations)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Add cleared column if it doesn't exist
                if "cleared" not in columns:
                    cursor.execute("""
                        ALTER TABLE conversations 
                        ADD COLUMN cleared BOOLEAN DEFAULT 0
                    """)
                    conn.commit()
                    print("Added cleared column to conversations table")
        except Exception as e:
            print(f"Error ensuring cleared column: {e}") 

    def get_conversations_by_thread(self, thread_id: str):
        """Get all conversations associated with a thread ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if thread_id column exists
            cursor.execute("PRAGMA table_info(conversations)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if "thread_id" not in columns:
                return []
            
            cursor.execute(
                "SELECT * FROM conversations WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,)
            )
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            conversations = []
            for row in rows:
                conversation = {}
                for i, column in enumerate(column_names):
                    conversation[column] = row[i]
                
                # Handle JSON fields
                for field in ['output', 'code_context', 'technical_details']:
                    if field in conversation and conversation[field]:
                        try:
                            if isinstance(conversation[field], str):
                                conversation[field] = json.loads(conversation[field])
                        except json.JSONDecodeError:
                            # Keep as string if not valid JSON
                            pass
                
                conversations.append(conversation)
            
            return conversations

    def ensure_thread_id_column(self):
        """Ensure thread_id column exists in conversations table"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if thread_id column exists
                cursor.execute("PRAGMA table_info(conversations)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Add thread_id column if it doesn't exist
                if "thread_id" not in columns:
                    cursor.execute("""
                        ALTER TABLE conversations 
                        ADD COLUMN thread_id TEXT
                    """)
                    conn.commit()
                    print("Added thread_id column to conversations table")
        except Exception as e:
            print(f"Error ensuring thread_id column: {e}") 

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create conversations table with architect_response column
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        query TEXT,
                        output TEXT,
                        code_context TEXT,
                        technical_details TEXT,
                        architect_response TEXT,  -- Add this column
                        feedback_status TEXT,
                        feedback_comments TEXT,
                        thread_id TEXT,
                        cleared INTEGER DEFAULT 0
                    )
                ''')
                
                # Add architect_response column if it doesn't exist
                cursor.execute('''
                    SELECT COUNT(*) FROM pragma_table_info('conversations') 
                    WHERE name='architect_response'
                ''')
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        ALTER TABLE conversations 
                        ADD COLUMN architect_response TEXT
                    ''')
                
                conn.commit()
                
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise

    def save_conversation(self, conversation_id: str, data: Dict[str, Any]):
        """Save or update a conversation"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract architect response if present
                architect_response = data.get('architect_response')
                if isinstance(architect_response, dict):
                    architect_response = json.dumps(architect_response)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations (
                        id, query, output, code_context, technical_details, 
                        architect_response, feedback_status, feedback_comments, 
                        thread_id, cleared
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conversation_id,
                    data.get('query', ''),
                    data.get('output', ''),
                    data.get('code_context', '{}'),
                    data.get('technical_details', '{}'),
                    architect_response,
                    data.get('feedback_status'),
                    data.get('feedback_comments'),
                    data.get('thread_id'),
                    data.get('cleared', 0)
                ))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error saving conversation: {e}")
            raise 