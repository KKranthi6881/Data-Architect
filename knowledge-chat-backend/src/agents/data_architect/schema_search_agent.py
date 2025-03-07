import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Set up logger
logger = logging.getLogger(__name__)

class SchemaSearchAgent:
    """Agent for searching SQL schemas in vector database"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = "nomic-embed-text"
        self.llm_model = "llama3"
        self.collection_name = "sql_schemas"
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "db")
        self.threads_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "threads")
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url="http://localhost:11434"
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the vector store for schema search"""
        try:
            # Create vector store directory if it doesn't exist
            os.makedirs(os.path.join(self.db_path, "vector_db"), exist_ok=True)
            
            # Initialize Chroma with the embeddings
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=os.path.join(self.db_path, "vector_db")
            )
            
            return vector_store
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            raise
    
    def search_schemas(self, parsed_question: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant SQL schemas based on the parsed question"""
        try:
            # Create search query from parsed question
            search_query = self._create_search_query(parsed_question)
            self.logger.info(f"Created search query: {search_query}")
            
            # Search for relevant schemas
            results = self.vector_store.similarity_search_with_score(
                query=search_query,
                k=top_k
            )
            
            # Format search results
            formatted_results = self._format_search_results(results, parsed_question)
            
            # Enhance search results with LLM
            enhanced_results = self._enhance_search_results(formatted_results, parsed_question)
            
            return enhanced_results
        except Exception as e:
            self.logger.error(f"Error searching schemas: {e}")
            return []
    
    def _create_search_query(self, parsed_question: Dict[str, Any]) -> str:
        """Create a search query from the parsed question"""
        try:
            # Extract key information from parsed question
            original_question = parsed_question.get("original_question", "")
            rephrased_question = parsed_question.get("rephrased_question", "")
            business_context = parsed_question.get("business_context", {})
            domain = business_context.get("domain", "")
            primary_objective = business_context.get("primary_objective", "")
            key_entities = business_context.get("key_entities", [])
            
            # Extract query intent
            query_intent = parsed_question.get("query_intent", {})
            primary_intent = query_intent.get("primary_intent", "")
            metrics = query_intent.get("metrics", [])
            grouping = query_intent.get("grouping", [])
            
            # Combine information into a search query
            search_query = f"{rephrased_question} {primary_intent}"
            
            # Add key entities
            if key_entities:
                search_query += f" {' '.join(key_entities)}"
            
            # Add metrics and grouping
            if metrics:
                search_query += f" {' '.join(metrics)}"
            if grouping:
                search_query += f" {' '.join(grouping)}"
            
            return search_query
        except Exception as e:
            self.logger.error(f"Error creating search query: {e}")
            return parsed_question.get("original_question", "")
    
    def _format_search_results(self, results: List[tuple], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format search results for further processing"""
        formatted_results = []
        
        for doc, score in results:
            # Extract schema information from document
            schema_info = {
                "schema_name": doc.metadata.get("schema_name", "Unknown"),
                "table_name": doc.metadata.get("table_name", "Unknown"),
                "columns": doc.metadata.get("columns", []),
                "description": doc.metadata.get("description", ""),
                "content": doc.page_content,
                "relevance_score": float(score),
                "metadata": doc.metadata
            }
            
            formatted_results.append(schema_info)
        
        return formatted_results
    
    def _enhance_search_results(self, search_results: List[Dict[str, Any]], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM explanations"""
        if not search_results:
            return []
        
        try:
            # Prepare context for LLM
            context = {
                "question": parsed_question.get("original_question", ""),
                "rephrased_question": parsed_question.get("rephrased_question", ""),
                "business_context": parsed_question.get("business_context", {}),
                "query_intent": parsed_question.get("query_intent", {}),
                "search_results": search_results
            }
            
            # Convert context to JSON string
            context_json = json.dumps(context, indent=2)
            
            # Prepare prompt for LLM
            prompt = f"""
            You are a data architecture expert. I need you to analyze these schema search results and explain how they relate to the user's question.
            
            For each schema result, provide:
            1. A brief explanation of why this schema is relevant to the question
            2. How the columns in this schema can be used to answer the question
            3. Any potential SQL query patterns that could be used with this schema
            
            Here is the context:
            {context_json}
            
            Please format your response as a JSON array with the following structure for each result:
            [
                {{
                    "schema_name": "original schema name",
                    "table_name": "original table name",
                    "columns": ["original columns"],
                    "relevance_score": original relevance score,
                    "explanation": "Your explanation of why this schema is relevant",
                    "query_pattern": "Example SQL query pattern using this schema",
                    "column_usage": {{
                        "column_name": "explanation of how this column is useful"
                    }}
                }},
                ...
            ]
            
            Return only the JSON array without any additional text.
            """
            
            # Call LLM to enhance results
            response = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error from LLM API: {response.text}")
                return search_results
            
            # Extract JSON from response
            llm_response = response.json()
            llm_content = llm_response.get("message", {}).get("content", "")
            
            # Extract JSON array from content
            json_text = llm_content
            if "```json" in llm_content:
                json_text = llm_content.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_content:
                json_text = llm_content.split("```")[1].strip()
            
            # Parse JSON
            enhanced_results = json.loads(json_text)
            
            return enhanced_results
        except Exception as e:
            self.logger.error(f"Error enhancing search results: {e}")
            return search_results
    
    def save_search_results(self, thread_id: str, conversation_id: str, parsed_question: Dict[str, Any], search_results: List[Dict[str, Any]]):
        """Save search results to thread directory"""
        try:
            # Create thread directory if it doesn't exist
            thread_dir = os.path.join(self.threads_path, thread_id)
            os.makedirs(thread_dir, exist_ok=True)
            
            # Create search results file
            search_results_file = os.path.join(thread_dir, f"schema_results_{conversation_id}.json")
            
            # Prepare data to save
            data_to_save = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "parsed_question": parsed_question,
                "search_results": search_results
            }
            
            # Save to file
            with open(search_results_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.logger.info(f"Saved search results to {search_results_file}")
        except Exception as e:
            self.logger.error(f"Error saving search results: {e}")
    
    def add_schema_to_vector_store(self, schema_info: Dict[str, Any]):
        """Add a schema to the vector store"""
        try:
            # Create document from schema info
            content = f"""
            Schema: {schema_info.get('schema_name', '')}
            Table: {schema_info.get('table_name', '')}
            Description: {schema_info.get('description', '')}
            Columns: {', '.join(schema_info.get('columns', []))}
            """
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "schema_name": schema_info.get("schema_name", ""),
                    "table_name": schema_info.get("table_name", ""),
                    "columns": schema_info.get("columns", []),
                    "description": schema_info.get("description", ""),
                    "primary_key": schema_info.get("primary_key", []),
                    "foreign_keys": schema_info.get("foreign_keys", []),
                    "data_types": schema_info.get("data_types", {}),
                    "source": schema_info.get("source", "manual")
                }
            )
            
            # Add to vector store
            self.vector_store.add_documents([doc])
            
            # Persist changes
            self.vector_store.persist()
            
            return True
        except Exception as e:
            self.logger.error(f"Error adding schema to vector store: {e}")
            return False
    
    def bulk_add_schemas(self, schemas: List[Dict[str, Any]]):
        """Add multiple schemas to the vector store"""
        try:
            documents = []
            
            for schema in schemas:
                # Create content for embedding
                content = f"""
                Schema: {schema.get('schema_name', '')}
                Table: {schema.get('table_name', '')}
                Description: {schema.get('description', '')}
                Columns: {', '.join(schema.get('columns', []))}
                """
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        "schema_name": schema.get("schema_name", ""),
                        "table_name": schema.get("table_name", ""),
                        "columns": schema.get("columns", []),
                        "description": schema.get("description", ""),
                        "primary_key": schema.get("primary_key", []),
                        "foreign_keys": schema.get("foreign_keys", []),
                        "data_types": schema.get("data_types", {}),
                        "source": schema.get("source", "bulk_import")
                    }
                )
                
                documents.append(doc)
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Persist changes
            self.vector_store.persist()
            
            return True
        except Exception as e:
            self.logger.error(f"Error bulk adding schemas: {e}")
            return False 