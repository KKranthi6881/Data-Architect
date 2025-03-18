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
                    architect_response TEXT,
                    code_context TEXT,
                    technical_details TEXT,
                    feedback_status TEXT,
                    feedback_comments TEXT,
                    thread_id TEXT,
                    cleared INTEGER DEFAULT 0
                )
            ''')
            
            # Create checkpoints table with langgraph compatible schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    type TEXT,
                    checkpoint TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parent_checkpoint_id TEXT,
                    checkpoint_ns TEXT,
                    metadata TEXT
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

    def save_conversation(self, data):
        """Save a conversation to the database."""
        try:
            # Extract conversation_id and content from data
            conversation_id = data.get('conversation_id')
            thread_id = data.get('thread_id', conversation_id)  # Default to conversation_id if thread_id is not provided
            question = data.get('question', '')
            
            # Handle architect_response - ensure it's a JSON string
            architect_response = data.get('architect_response', {})
            if isinstance(architect_response, dict):
                try:
                    architect_response = json.dumps(architect_response)
                except Exception as e:
                    self.logger.warning(f"Failed to convert architect_response to JSON: {e}")
                    architect_response = str(architect_response)
            
            # Handle technical_details - ensure it's a JSON string
            technical_details = data.get('technical_details', {})
            if isinstance(technical_details, dict) or isinstance(technical_details, list):
                try:
                    technical_details = json.dumps(technical_details)
                except Exception as e:
                    self.logger.warning(f"Failed to convert technical_details to JSON: {e}")
                    technical_details = str(technical_details)
            
            # Handle code_context - ensure it's a JSON string
            code_context = data.get('code_context', {})
            if isinstance(code_context, dict) or isinstance(code_context, list):
                try:
                    code_context = json.dumps(code_context)
                except Exception as e:
                    self.logger.warning(f"Failed to convert code_context to JSON: {e}")
                    code_context = str(code_context)
            
            # Get metadata as JSON string
            metadata = data.get('metadata', {})
            if isinstance(metadata, dict) or isinstance(metadata, list):
                try:
                    metadata = json.dumps(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to convert metadata to JSON: {e}")
                    metadata = str(metadata)
            
            # Check if we should use the old schema or new schema
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(conversations)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if "user_question" in columns:
                    # New schema
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO conversations
                        (conversation_id, thread_id, user_question, architect_response, technical_details, code_context, metadata, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conversation_id,
                            thread_id,
                            question,
                            architect_response,
                            technical_details,
                            code_context,
                            metadata,
                            datetime.now().isoformat()
                        )
                    )
                else:
                    # Old schema
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO conversations
                        (id, thread_id, query, architect_response, technical_details, code_context, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conversation_id,
                            thread_id,
                            question,
                            architect_response,
                            technical_details,
                            code_context,
                            datetime.now().isoformat()
                        )
                    )
                
                conn.commit()
                self.logger.info(f"Saved conversation {conversation_id} with thread {thread_id}")
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
                        c.id,
                        c.thread_id,
                        c.query,
                        c.architect_response,
                        c.created_at,
                        c.feedback_status,
                        c.feedback_comments,
                        c.technical_details
                    FROM conversations c
                    WHERE c.id = ?
                ''', (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    # Process architect_response safely
                    architect_response = row[3]
                    processed_response = None
                    if architect_response:
                        try:
                            if isinstance(architect_response, str):
                                processed_response = json.loads(architect_response)
                            else:
                                processed_response = architect_response
                        except json.JSONDecodeError:
                            # If JSON parsing fails, store as raw text
                            processed_response = {
                                "response": architect_response,
                                "type": "raw_text"
                            }

                    # Process technical_details safely
                    technical_details = row[7]
                    processed_details = None
                    if technical_details:
                        try:
                            if isinstance(technical_details, str):
                                processed_details = json.loads(technical_details)
                            else:
                                processed_details = technical_details
                        except json.JSONDecodeError:
                            processed_details = {
                                "details": technical_details,
                                "type": "raw_text"
                            }

                    return {
                        'id': row[0],
                        'thread_id': row[1],
                        'query': row[2],
                        'architect_response': processed_response,
                        'created_at': row[4],
                        'feedback_status': row[5],
                        'feedback_comments': row[6],
                        'technical_details': processed_details
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting conversation: {e}", exc_info=True)
            raise

    def get_recent_conversations(self, limit=10):
        """Get recent conversations with safe handling of missing columns"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Update query to use architect_response instead of output
                query = """
                    SELECT 
                        id,
                        created_at,
                        query,
                        architect_response,
                        feedback_status,
                        cleared
                    FROM conversations 
                    WHERE cleared IS NULL OR cleared = 0
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                
                conversations = []
                for row in rows:
                    try:
                        # Parse architect_response if it exists
                        architect_response = row[3] if len(row) > 3 else None
                        if isinstance(architect_response, str):
                            try:
                                architect_response = json.loads(architect_response)
                            except json.JSONDecodeError:
                                architect_response = architect_response
                        
                        conversation = {
                            "id": row[0],
                            "created_at": row[1],
                            "query": row[2],
                            "output": architect_response,  # Use architect_response as output
                            "feedback_status": row[4] if len(row) > 4 else "pending",
                            "cleared": bool(row[5]) if len(row) > 5 else False
                        }
                        conversations.append(conversation)
                        
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
                """SELECT id, created_at, query, architect_response, code_context 
                   FROM conversations 
                   ORDER BY created_at DESC"""
            )
            rows = cursor.fetchall()
            
            return [{
                'id': row[0],
                'timestamp': row[1],
                'query': row[2],
                'output': json.loads(row[3]) if row[3] else {},  # Load architect_response as output
                'code_context': json.loads(row[4]) if row[4] else {}
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
                    architect_response,
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
                                'metadata': self._safe_load_binary(cp[4]) if len(cp) > 4 else {}
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

    def create_checkpoint(self, thread_id: str, checkpoint_id: str, checkpoint_type: str, checkpoint_data: str, metadata: str = None):
        """Create a new checkpoint"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints (
                    thread_id, checkpoint_id, type, checkpoint, metadata
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, checkpoint_id, checkpoint_type, checkpoint_data, metadata)
            )
            conn.commit()
            self.logger.info(f"Created checkpoint {checkpoint_id} for thread {thread_id}")

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
        """Create necessary tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        thread_id TEXT,
                        user_question TEXT,
                        architect_response TEXT,
                        technical_details TEXT,
                        code_context TEXT,
                        metadata TEXT,
                        timestamp TEXT
                    )
                ''')
                
                # Create checkpoints table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        type TEXT,
                        created_at TEXT,
                        conversation_id TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT,
                        rating INTEGER,
                        comments TEXT,
                        created_at TEXT
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database tables created or verified.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {str(e)}", exc_info=True)

    def ensure_architect_response_column(self):
        """Ensure the architect_response column exists in the conversations table"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check if the column exists
                cursor.execute("PRAGMA table_info(conversations)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if "architect_response" not in columns:
                    # Add the column
                    cursor.execute("ALTER TABLE conversations ADD COLUMN architect_response TEXT")
                    conn.commit()
                    logging.info("Added architect_response column to conversations table")
                
        except Exception as e:
            logging.error(f"Error ensuring architect_response column: {e}")

    def ensure_parent_checkpoint_column(self):
        """Ensure the parent_checkpoint_id column exists in the checkpoints table"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # First, drop the existing checkpoints table to recreate it with correct schema
                cursor.execute("DROP TABLE IF EXISTS checkpoints")
                conn.commit()
                
                # Create the checkpoints table with langgraph-compatible column names
                cursor.execute("""
                    CREATE TABLE checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        thread_id TEXT,
                        type TEXT,
                        checkpoint TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        parent_checkpoint_id TEXT,
                        checkpoint_ns TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
                logging.info("Created checkpoints table with langgraph-compatible schema")
                
        except Exception as e:
            logging.error(f"Error ensuring checkpoint columns: {e}")

    def get_all_conversations(self):
        """Get all conversations with their complete data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        id,
                        created_at,
                        query,
                        technical_details,
                        architect_response,
                        code_context,
                        feedback_status,
                        thread_id,
                        cleared
                    FROM conversations 
                    WHERE cleared = 0 OR cleared IS NULL
                    ORDER BY created_at DESC
                """)
                
                conversations = []
                
                for row in cursor.fetchall():
                    try:
                        # Parse JSON fields
                        technical_details = json.loads(row[3]) if row[3] else {}
                        architect_response = json.loads(row[4]) if row[4] else {}
                        code_context = json.loads(row[5]) if row[5] else {}
                        
                        conversation = {
                            "id": row[0],
                            "created_at": row[1],
                            "query": row[2],
                            "technical_details": technical_details,
                            "architect_response": architect_response,
                            "code_context": code_context,
                            "feedback_status": row[6],
                            "thread_id": row[7] or row[0],  # Use id as thread_id if null
                            "cleared": bool(row[8])
                        }
                        conversations.append(conversation)
                    except Exception as e:
                        self.logger.error(f"Error processing conversation row: {e}")
                        continue
                        
                return conversations
                
        except Exception as e:
            self.logger.error(f"Error getting all conversations: {e}")
            return [] 

    def get_thread_conversations(self):
        """Get all conversations organized by threads"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        c.thread_id,
                        c.id as conversation_id,
                        c.created_at,
                        c.query,
                        c.architect_response,
                        c.feedback_status,
                        COUNT(*) OVER (PARTITION BY c.thread_id) as message_count,
                        MIN(c.query) OVER (PARTITION BY c.thread_id) as initial_query,
                        CASE WHEN c.architect_response IS NOT NULL THEN 1 ELSE 0 END as has_architect_response
                    FROM conversations c
                    WHERE c.cleared = 0 OR c.cleared IS NULL
                    ORDER BY c.created_at DESC
                """)
                
                rows = cursor.fetchall()
                threads = {}
                
                for row in rows:
                    try:
                        thread_id = row[0] or row[1]  # Use conversation_id as thread_id if null
                        
                        # Process architect_response safely
                        architect_response = row[4]
                        processed_response = None
                        
                        if architect_response:
                            if isinstance(architect_response, str):
                                try:
                                    # Try to parse JSON string
                                    processed_response = json.loads(architect_response)
                                except json.JSONDecodeError:
                                    # If parsing fails, wrap the raw text
                                    processed_response = {
                                        "response": architect_response,
                                        "type": "raw_text"
                                    }
                            else:
                                processed_response = architect_response
                        
                        conversation = {
                            "id": row[1],
                            "created_at": row[2],
                            "query": row[3],
                            "architect_response": processed_response or {"response": "", "type": "empty"},
                            "feedback_status": row[5]
                        }
                        
                        if thread_id not in threads:
                            threads[thread_id] = {
                                "thread_id": thread_id,
                                "created_at": row[2],
                                "initial_query": row[7],
                                "message_count": row[6],
                                "has_architect_response": bool(row[8]),
                                "conversations": []
                            }
                        
                        threads[thread_id]["conversations"].append(conversation)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing conversation row: {e}", exc_info=True)
                        continue
                
                # Convert to list and sort by created_at
                thread_list = list(threads.values())
                thread_list.sort(key=lambda x: x["created_at"], reverse=True)
                
                return {
                    "status": "success",
                    "threads": thread_list
                }
                
        except Exception as e:
            self.logger.error(f"Error getting thread conversations: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "threads": []
            }

    def ensure_conversation_id_column(self):
        """Ensure conversation_id column exists in the conversations table"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if conversation_id column exists
                cursor.execute('''
                    SELECT COUNT(*) FROM pragma_table_info('conversations') 
                    WHERE name='conversation_id'
                ''')
                if cursor.fetchone()[0] == 0:
                    # If id column exists, add conversation_id and copy values
                    cursor.execute('''
                        SELECT COUNT(*) FROM pragma_table_info('conversations') 
                        WHERE name='id'
                    ''')
                    if cursor.fetchone()[0] > 0:
                        # Add conversation_id column
                        cursor.execute('''
                            ALTER TABLE conversations 
                            ADD COLUMN conversation_id TEXT
                        ''')
                        # Copy values from id to conversation_id
                        cursor.execute('''
                            UPDATE conversations 
                            SET conversation_id = id
                            WHERE conversation_id IS NULL
                        ''')
                    else:
                        # No id column, rename the table and create a new one
                        cursor.execute('''
                            ALTER TABLE conversations 
                            RENAME TO conversations_old
                        ''')
                        self.create_tables()
                        cursor.execute('''
                            INSERT INTO conversations (
                                conversation_id, thread_id, user_question, 
                                architect_response, technical_details, code_context, 
                                metadata, timestamp
                            )
                            SELECT id, thread_id, query, 
                                   architect_response, technical_details, code_context,
                                   '', created_at
                            FROM conversations_old
                        ''')
                
                conn.commit()
                self.logger.info("Ensured conversation_id column exists")
        except Exception as e:
            self.logger.error(f"Error ensuring conversation_id column: {str(e)}", exc_info=True)