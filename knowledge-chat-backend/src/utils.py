from src.patches import patch_chromadb_numpy
# Apply patch before importing chromadb
patch_chromadb_numpy()

import os
from typing import Optional, List, Dict, Any, Union
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import Language
import logging
import yaml
from dotenv import load_dotenv
import tempfile
from src.code_analyzer import CodeAnalyzer
import getpass
import os
import requests
import zipfile
import io
from pathlib import Path
import base64
from urllib.parse import urlparse
import git
import shutil
import json
import hashlib
from datetime import datetime
import subprocess
import re




logger = logging.getLogger(__name__)

class OllamaEmbeddingFunction:
    def __init__(self):
        """Initialize the Ollama embedding function"""
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )
            # Test the connection
            self._test_connection()
        except Exception as e:
            raise ValueError(f"Failed to initialize Ollama embeddings: {str(e)}")

    def _test_connection(self):
        """Test the embedding connection"""
        try:
            test_result = self.embeddings.embed_query("test")
            if not test_result or len(test_result) == 0:
                raise ValueError("Empty embedding result")
        except Exception as e:
            raise ValueError(f"Embedding test failed: {str(e)}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB compatible embedding function"""
        if isinstance(input, str):
            input = [input]
        return self.embeddings.embed_documents(input)

class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB with a persistence directory."""
        try:
            # Initialize the Ollama embedding function
            self.embedding_function = OllamaEmbeddingFunction()
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            self.collections = {}
            
            # Initialize text splitters
            self.doc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            self.code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Initialize code analyzer
            from src.code_analyzer import CodeAnalyzer
            self.code_analyzer = CodeAnalyzer()
            
            # Initialize analysis cache
            self._analysis_cache = {}
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDBManager: {str(e)}")
            raise

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get an existing collection or create a new one."""
        if collection_name not in self.collections:
            try:
                self.collections[collection_name] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_function
                )
            except ValueError as e:
                logger.error(f"Error getting/creating collection: {e}")
                raise
        return self.collections[collection_name]

    def process_document(self, file_path: str, doc_type: str) -> List[Document]:
        """Process any document with appropriate chunking."""
        try:
            logger.info(f"Loading {doc_type} file: {file_path}")
            
            if doc_type.lower() == 'pdf':
                # Ensure the file exists and is readable
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"PDF file not found at {file_path}")
                
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    logger.info(f"Successfully loaded PDF with {len(documents)} pages")
                    
                    if not documents:
                        raise ValueError("No content extracted from PDF")
                    
                    chunks = self.doc_splitter.split_documents(documents)
                    logger.info(f"Split PDF into {len(chunks)} chunks")
                    
                    if not chunks:
                        raise ValueError("No chunks created from PDF")
                        
                    return chunks
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                    raise ValueError(f"Failed to process PDF: {str(e)}")
            else:
                loader = TextLoader(file_path)
                documents = loader.load()
                chunks = self.doc_splitter.split_documents(documents)
                logger.info(f"Split {doc_type} into {len(chunks)} chunks")
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing {doc_type}: {str(e)}", exc_info=True)
            raise

    def process_code(self, file_path: str, language: str) -> List[Document]:
        """Process code with language-specific chunking and analysis."""
        try:
            logger.info(f"Loading {language} code file: {file_path}")
            
            # Detect if this is a dbt file
            is_dbt_file = self.code_analyzer._is_dbt_file(file_path)
            
            # Analyze code once and cache the results
            cache_key = f"{file_path}:{language}"
            if cache_key not in self._analysis_cache:
                if is_dbt_file:
                    if file_path.endswith('.yml') or file_path.endswith('.yaml'):
                        analysis = self.code_analyzer.analyze_dbt_schema(file_path)
                    else:
                        analysis = self.code_analyzer.analyze_dbt(file_path)
                else:
                    analysis = self.code_analyzer.analyze_file(file_path, language)
                self._analysis_cache[cache_key] = analysis
            else:
                analysis = self._analysis_cache[cache_key]
            
            logger.info(f"Code analysis complete for {file_path}")
            
            # Flatten and stringify analysis for ChromaDB metadata
            flattened_metadata = self._prepare_metadata_for_chroma(analysis)
            
            # Set language-specific splitting with special handling for dbt files
            if is_dbt_file:
                # For dbt files, use smaller chunks with more overlap for better context
                self.code_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=100
                )
            elif hasattr(Language, language.upper()):
                self.code_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=getattr(Language, language.upper()),
                    chunk_size=500,
                    chunk_overlap=50
                )
            
            # Load and split code
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Create initial document with enhanced metadata for dbt
            metadata = {
                'source': file_path,
                'language': language,
                **flattened_metadata
            }
            
            # Add special metadata for dbt files
            if is_dbt_file:
                metadata['is_dbt'] = True
                metadata['dbt_type'] = 'schema' if file_path.endswith(('.yml', '.yaml')) else 'model'
                
                # For dbt models, add refs and dependencies 
                if 'jinja_references' in analysis:
                    metadata['dbt_refs'] = ','.join(analysis.get('jinja_references', []))
                
                # For schemas, add model definitions
                if 'models' in analysis:
                    model_names = [model.get('name', '') for model in analysis.get('models', [])]
                    metadata['dbt_models'] = ','.join(model_names)
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Split into chunks
            chunks = self.code_splitter.split_documents([doc])
            
            # Enhance chunks with metadata
            for chunk in chunks:
                chunk.metadata.update(metadata)
            
            logger.info(f"Split code into {len(chunks)} chunks with analysis metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing code: {str(e)}", exc_info=True)
            raise

    def _prepare_metadata_for_chroma(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Prepare analysis metadata for ChromaDB with enhanced dbt support."""
        flattened = {}
        
        try:
            # Check if this is a dbt analysis
            if analysis.get('file_type', '').startswith('dbt_'):
                # Handle dbt-specific metadata
                flattened['dbt_type'] = analysis.get('file_type', 'dbt')
                flattened['model_name'] = analysis.get('model_name', '')
                
                # Add model materialization if available
                if 'materialization' in analysis:
                    flattened['materialization'] = analysis['materialization']
                
                # Add model description
                if 'description' in analysis:
                    flattened['description'] = analysis['description']
                
                # Add references to other models
                if 'jinja_references' in analysis:
                    flattened['references'] = ','.join(analysis['jinja_references'])
                
                # Add sources
                if 'jinja_sources' in analysis:
                    sources = [f"{s[0]}.{s[1]}" for s in analysis['jinja_sources']]
                    flattened['sources'] = ','.join(sources)
                
                # Add column information
                if 'columns' in analysis:
                    if isinstance(analysis['columns'], list):
                        flattened['columns'] = ','.join(analysis['columns'])
                
                # Add column descriptions if available
                if 'column_descriptions' in analysis and isinstance(analysis['column_descriptions'], dict):
                    col_descs = []
                    for col, desc in analysis['column_descriptions'].items():
                        if isinstance(desc, dict) and 'description' in desc:
                            col_descs.append(f"{col}: {desc['description']}")
                        else:
                            col_descs.append(f"{col}: {desc}")
                    flattened['column_descriptions'] = ' | '.join(col_descs)
                
                # Handle schema.yml specific data
                if 'models' in analysis:
                    model_names = [model.get('name', '') for model in analysis.get('models', [])]
                    flattened['schema_models'] = ','.join(model_names)
                
                # Add sources from schema.yml
                if 'sources' in analysis:
                    source_names = []
                    for source in analysis.get('sources', []):
                        source_name = source.get('name', '')
                        for table in source.get('tables', []):
                            source_names.append(f"{source_name}.{table.get('name', '')}")
                    flattened['schema_sources'] = ','.join(source_names)
            
            # Original generic metadata handling
            if 'tables' in analysis:
                flattened['tables'] = ','.join(sorted([str(t) for t in analysis['tables']]))
            
            # Handle relationships
            if 'relationships' in analysis and analysis['relationships']:
                relationships = []
                for rel in analysis['relationships']:
                    rel_str = f"{rel.get('left_table')}.{rel.get('left_column')} -> {rel.get('right_table')}.{rel.get('right_column')}"
                    relationships.append(rel_str)
                flattened['relationships'] = '|'.join(relationships)
            
            # Add analysis summary
            if analysis.get('file_type', '').startswith('dbt_'):
                flattened['analysis_summary'] = f"dbt {analysis.get('model_type', 'model')} {analysis.get('model_name', '')} with {len(analysis.get('jinja_references', []))} references and {len(analysis.get('jinja_sources', []))} sources"
            else:
                flattened['analysis_summary'] = f"Analyzed {len(analysis.get('tables', []))} tables with {len(analysis.get('relationships', []))} relationships"
            
            # Ensure all values are strings
            for key, value in flattened.items():
                if not isinstance(value, (str, int, float, bool)):
                    flattened[key] = str(value)
            
            return flattened
            
        except Exception as e:
            logger.error(f"Error preparing metadata: {str(e)}")
            return {
                'error': str(e),
                'analysis_status': 'failed'
            }

    def add_documents(self, collection_name: str, documents: List[Document], metadata: Optional[Dict] = None):
        """Add documents with embeddings to collection."""
        try:
            collection = self.get_or_create_collection(collection_name)
            
            for idx, doc in enumerate(documents):
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata.update(doc.metadata)
                
                try:
                    collection.add(
                        documents=[doc.page_content],
                        metadatas=[doc_metadata],
                        ids=[f"{collection_name}_{idx}"]
                    )
                    logger.info(f"Added document {idx+1}/{len(documents)} to collection {collection_name}")
                except Exception as e:
                    logger.error(f"Error adding document {idx}: {str(e)}")
                    raise
                
        except Exception as e:
            logger.error(f"Error in add_documents: {str(e)}", exc_info=True)
            raise

    def hybrid_search(self, collection_name: str, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Perform hybrid search combining keyword and semantic search."""
        try:
            logger.info(f"Starting hybrid search in collection '{collection_name}' for query: '{query}'")
            collection = self.get_or_create_collection(collection_name)
            
            # Log collection info
            try:
                count = collection.count()
                logger.info(f"Collection has {count} documents")
            except Exception as e:
                logger.warning(f"Could not get collection count: {e}")

            # Perform hybrid search
            results = collection.query(
                query_texts=[query],  # Ensure query is in a list
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"Raw search results: {results}")
            
            # Format and score results
            formatted_results = []
            if results and 'documents' in results and results['documents']:
                # Handle the case where results are nested in lists
                documents = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
                distances = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
                metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                
                for i in range(len(documents)):
                    try:
                        # Ensure we have a numeric distance
                        distance = float(distances[i])
                        
                        # Handle the case where document might be a list
                        document = documents[i]
                        if isinstance(document, list):
                            document = document[0] if document else ""
                        
                        formatted_result = {
                            'content': document,
                            'metadata': metadatas[i],  # Use the correct metadata for this document
                            'similarity': 1 - distance,  # Convert distance to similarity
                        }
                        formatted_results.append(formatted_result)
                        logger.info(f"Formatted result {i}: {formatted_result}")
                    except Exception as e:
                        logger.error(f"Error processing result {i}: {str(e)}")
                        continue
            else:
                logger.warning("No documents found in search results")
            
            # Sort by similarity score
            formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            final_results = {
                'query': query,
                'results': formatted_results
            }
            logger.info(f"Returning {len(formatted_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
            raise

    def code_similarity_search(self, collection_name: str, code_snippet: str, n_results: int = 5) -> Dict[str, Any]:
        """Specialized similarity search for code."""
        try:
            logger.info(f"Starting code search in collection '{collection_name}' for query: '{code_snippet}'")
            
            # If no collection name provided, try to find an appropriate one
            if not collection_name:
                collections = self.client.list_collections()
                code_collections = [c.name for c in collections 
                                  if any(ext in c.name.lower() 
                                        for ext in ['py', 'sql', 'yml', 'yaml'])]
                if code_collections:
                    collection_name = code_collections[0]
                else:
                    return {
                        'query': code_snippet,
                        'results': []
                    }
            
            # Normalize code snippet
            normalized_query = self._normalize_code(code_snippet)
            logger.info(f"Normalized query: {normalized_query}")
            
            # Get collection
            collection = self.get_or_create_collection(collection_name)
            
            # Perform search
            results = collection.query(
                query_texts=[normalized_query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results with code-specific information
            formatted_results = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
                distances = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
                metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                
                for i in range(len(documents)):
                    try:
                        # Get the code content
                        code = documents[i]
                        if isinstance(code, list):
                            code = code[0] if code else ""
                        
                        # Find matching lines with context
                        matched_lines = self._get_matching_lines(code, normalized_query)
                        
                        formatted_result = {
                            'code': code,
                            'file_info': metadatas[i],
                            'similarity': 1 - float(distances[i]),
                            'matched_lines': matched_lines,
                            'language': metadatas[i].get('language', 'unknown')
                        }
                        formatted_results.append(formatted_result)
                    except Exception as e:
                        logger.error(f"Error processing code result {i}: {str(e)}")
                        continue
            
            return {
                'query': code_snippet,
                'results': formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error in code similarity search: {str(e)}")
            return {
                'query': code_snippet,
                'results': []
            }

    def _normalize_code(self, code: str) -> str:
        """Normalize code for better matching."""
        try:
            # Remove comments and empty lines
            lines = []
            for line in code.split('\n'):
                # Remove inline comments
                line = line.split('#')[0].strip()
                # Remove empty lines
                if line:
                    # Normalize whitespace
                    line = ' '.join(line.split())
                    lines.append(line)
            
            # Join lines with space
            normalized = ' '.join(lines)
            
            # Remove extra whitespace
            normalized = ' '.join(normalized.split())
            
            logger.info(f"Normalized code: {normalized}")
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing code: {str(e)}")
            return code

    def _get_matching_lines(self, code: str, query: str) -> List[Dict[str, Any]]:
        """Find and return matching lines in code."""
        try:
            matching_lines = []
            lines = code.split('\n')
            
            # Normalize query for matching
            query_terms = self._normalize_code(query).lower().split()
            
            for i, line in enumerate(lines):
                normalized_line = self._normalize_code(line).lower()
                
                # Check if any query term matches
                if any(term in normalized_line for term in query_terms):
                    matching_lines.append({
                        'line_number': i + 1,
                        'content': line.strip(),
                        'context': self._get_line_context(lines, i)
                    })
            
            return matching_lines
            
        except Exception as e:
            logger.error(f"Error finding matching lines: {str(e)}")
            return []

    def _get_line_context(self, lines: List[str], current_line: int, context_lines: int = 2) -> str:
        """Get context around a matching line."""
        start = max(0, current_line - context_lines)
        end = min(len(lines), current_line + context_lines + 1)
        
        context = []
        for i in range(start, end):
            prefix = '> ' if i == current_line else '  '
            context.append(f"{prefix}{lines[i].strip()}")
        
        return '\n'.join(context)

    def process_github(self, repo_url: str, username: str = "", token: str = "") -> Dict[str, Any]:
        """
        Process a GitHub repository by cloning it and loading its contents.
        Focuses on essential file information for effective searching.
        
        Args:
            repo_url: URL of the GitHub repository
            username: GitHub username for authentication (optional)
            token: GitHub token for authentication (optional)
            
        Returns:
            Dictionary with repository metadata and processed files
        """
        try:
            logger.info(f"Processing GitHub repository: {repo_url}")
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp(prefix="github_repo_", dir="/tmp")
            
            # Clone the repository
            logger.info(f"Cloning repository to {temp_dir}")
            if username and token:
                auth_url = f"https://{username}:{token}@github.com/{repo_url.split('github.com/')[1]}"
                subprocess.run(["git", "clone", auth_url, temp_dir], check=True)
            else:
                subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
            
            # Process the repository files
            file_count = 0
            documents = []
            
            # Supported file types
            supported_extensions = {'.sql', '.py', '.md', '.yml', '.yaml', '.json', '.txt'}
            
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    # Skip hidden files and directories
                    if file.startswith('.') or any(part.startswith('.') for part in Path(root).parts):
                        continue
                        
                    # Check file extension
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in supported_extensions:
                        continue
                    
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Create document ID
                        doc_id = hashlib.md5(f"{repo_url}:{rel_path}".encode()).hexdigest()
                        
                        # Create basic metadata
                        metadata = {
                            "source": "github",
                            "repo_url": repo_url,
                            "file_path": rel_path,
                            "file_name": file,
                            "file_extension": ext.lstrip('.'),
                            "file_size": os.path.getsize(file_path),
                            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        }
                        
                        # Create document
                        document = {
                            "id": doc_id,
                            "content": content,
                            "metadata": metadata
                        }
                        
                        # Add to documents list
                        documents.append(document)
                        
                        # Add to vector store with the same metadata
                        self.add_document(
                            collection_name="github_documents",
                            document_id=doc_id,
                            text=content,
                            metadata=metadata
                        )
                        
                        file_count += 1
                        logger.info(f"Processed file {file_count}: {rel_path}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing file {rel_path}: {str(e)}")
                        continue
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            return {
                "status": "success",
                "files_processed": file_count,
                "repo_url": repo_url,
                "documents": documents
            }
            
        except Exception as e:
            logger.error(f"Error processing GitHub repository: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "repo_url": repo_url,
                "documents": []
            }

    def add_document(self, collection_name: str, document_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document to a ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            document_id: Unique ID for the document
            text: Document text content
            metadata: Document metadata
        """
        try:
            # Get or create the collection
            collection = self.get_or_create_collection(collection_name)
            
            # Add the document
            collection.add(
                ids=[document_id],
                documents=[text],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Error adding document to collection {collection_name}: {str(e)}")
            raise

    def _clean_for_json(self, obj):
        """Recursively convert an object to be JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def _is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read as text
            return False
        except UnicodeDecodeError:
            return True

    def _get_language_from_extension(self, ext: str) -> Optional[str]:
        """Map file extension to programming language."""
        extension_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'java': 'java',
            'c': 'c',
            'cpp': 'cpp',
            'h': 'c',
            'hpp': 'cpp',
            'cs': 'csharp',
            'go': 'go',
            'rb': 'ruby',
            'php': 'php',
            'sql': 'sql',
            'md': 'markdown',
            'json': 'json',
            'yml': 'yaml',
            'yaml': 'yaml',
            'xml': 'xml',
            'html': 'html',
            'css': 'css',
            'sh': 'shell',
            'bat': 'batch',
            'ps1': 'powershell',
            'r': 'r'
        }
        
        # Check for DBT files with compound extensions
        if ext in ['sql.jinja', 'sql.j2', 'sql.jinja2']:
            return 'dbt'
        
        # For files in models/ directory with .sql extension, treat as DBT
        if ext == 'sql' and ('/models/' in self.current_file_path or '/macros/' in self.current_file_path):
            return 'dbt'
        
        # For SQL files with Jinja content, treat as DBT
        if ext == 'sql' and hasattr(self, 'current_file_content') and ('{{' in self.current_file_content or '{%' in self.current_file_content):
            return 'dbt'
        
        return extension_map.get(ext)

    def process_git_zip(self, zip_path: str, filename: str) -> Dict[str, Any]:
        """
        Process a ZIP file containing a Git repository
        
        Args:
            zip_path: Path to the ZIP file
            filename: Original filename of the ZIP
            
        Returns:
            Dictionary with repository metadata and processed files
        """
        logger.info(f"Processing Git ZIP file: {zip_path}")
        
        # Create a temporary directory for extraction
        extract_dir = tempfile.mkdtemp(prefix="git_extract_")
        
        # Track current file path (used by language detection)
        self.current_file_path = ""
        
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Repository metadata
            repo_metadata = {
                "source": "zip_upload",
                "filename": filename,
                "upload_time": datetime.now().isoformat(),
                "file_count": 0,
                "languages": {}
            }
            
            # Track languages for stats
            language_counts = {}
            
            # Process the extracted files
            processed_files = []
            
            # Walk through the extracted directory
            for root, dirs, files in os.walk(extract_dir):
                # Skip hidden directories (like .git)
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    # Skip hidden files
                    if file.startswith('.'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, extract_dir)
                    
                    # Skip binary files and very large files
                    if self._is_binary_file(file_path) or os.path.getsize(file_path) > 1024 * 1024:
                        continue
                    
                    try:
                        # Get file extension
                        _, ext = os.path.splitext(file)
                        ext = ext.lstrip('.').lower()
                        
                        # Set current file path for language detection
                        self.current_file_path = rel_path
                        
                        # Determine the language
                        language = self._get_language_from_extension(ext)
                        
                        # Skip unsupported file types
                        if not language:
                            continue
                        
                        # Update language count
                        language_counts[language] = language_counts.get(language, 0) + 1
                        
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Analyze SQL content if applicable
                        analysis = {}
                        if language == 'sql':
                            analysis = self._analyze_sql(content)
                        elif language == 'python':
                            analysis = self._analyze_python(content)
                        
                        # Store processed file
                        processed_files.append({
                            "path": rel_path,
                            "language": language,
                            "content": content,
                            "analysis": analysis,
                            "size": os.path.getsize(file_path)
                        })
                    
                    except Exception as e:
                        logger.warning(f"Error processing file {rel_path}: {str(e)}")
            
            # Update repository metadata
            repo_metadata["file_count"] = len(processed_files)
            repo_metadata["languages"] = language_counts
            
            # Create documents for vector storage
            documents = []
            zip_id = hashlib.md5(filename.encode()).hexdigest()
            
            # Process each file into chunks
            for file in processed_files:
                # Generate a unique ID for the document
                doc_id = base64.b64encode(f"{zip_id}:{file['path']}".encode()).decode()
                
                # Create chunks from the file content
                if file['language'] in ['python', 'javascript', 'typescript', 'java']:
                    # Use code-specific splitter for programming languages
                    chunks = self.code_splitter.split_text(file['content'])
                else:
                    # Use regular document splitter for other files
                    chunks = self.doc_splitter.split_text(file['content'])
                
                # Create a document for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    
                    # Create metadata with only primitive types
                    metadata = {
                        "source": "git_zip",
                        "filename": filename,
                        "file_path": file['path'],
                        "language": file['language'],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "size": file.get('size', 0)
                    }
                    
                    # Add simplified analysis data if available
                    if file['analysis']:
                        # Convert complex analysis objects to simple strings
                        simple_analysis = {}
                        for key, value in file['analysis'].items():
                            if isinstance(value, (str, int, float, bool)):
                                simple_analysis[key] = value
                            elif isinstance(value, (list, dict)):
                                # Convert to string representation
                                simple_analysis[key] = str(value)
                        
                        metadata["analysis_summary"] = simple_analysis
                    
                    documents.append({
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": metadata
                    })
            
            # Add to vector store
            self.add_git_zip_documents(documents, filename)
            
            return {
                "metadata": repo_metadata,
                "document_count": len(documents)
            }
        
        finally:
            # Clean up
            try:
                shutil.rmtree(extract_dir)
                os.remove(zip_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def add_git_zip_documents(self, documents, filename):
        """Add Git ZIP documents to the vector store"""
        try:
            # Get the collection
            collection = self.get_or_create_collection("github_documents")  # Reuse the GitHub collection
            
            if not documents:
                logger.warning(f"No documents extracted from ZIP: {filename}")
                return
            
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Extract data for ChromaDB
                ids = [doc["id"] for doc in batch]
                texts = [doc["text"] for doc in batch]
                
                # Clean metadata
                cleaned_metadatas = []
                for doc in batch:
                    # Create a simplified metadata dict with only primitive types
                    cleaned_metadata = {
                        "source": "git_zip",
                        "filename": filename,
                        "file_path": doc["metadata"].get("file_path", ""),
                        "language": doc["metadata"].get("language", ""),
                        "chunk_index": doc["metadata"].get("chunk_index", 0),
                        "total_chunks": doc["metadata"].get("total_chunks", 1),
                        "upload_time": datetime.now().isoformat()
                    }
                    
                    # Add any other primitive metadata that might be useful
                    if "size" in doc["metadata"]:
                        cleaned_metadata["size"] = doc["metadata"]["size"]
                    
                    cleaned_metadatas.append(cleaned_metadata)
                
                # Add to collection
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=cleaned_metadatas
                )
            
            logger.info(f"Added {len(documents)} documents from Git ZIP file: {filename}")
            
        except Exception as e:
            logger.error(f"Error adding Git ZIP documents: {str(e)}")
            raise

    def _link_dbt_dependencies(self, processed_files: List[Dict[str, Any]]) -> None:
        """
        Link dbt models and their dependencies after all files have been processed.
        This enhances the analysis with inter-model relationships.
        """
        try:
            # Create a mapping of model names to file paths
            model_map = {}
            schema_map = {}
            
            # First pass to identify all models and schemas
            for file in processed_files:
                analysis = file.get('analysis', {})
                
                if not file.get('is_dbt', False):
                    continue
                
                path = file.get('path', '')
                
                if path.endswith('.sql'):
                    model_name = analysis.get('model_name', '')
                    if model_name:
                        model_map[model_name] = {
                            'path': path,
                            'file': file
                        }
                
                if path.endswith(('.yml', '.yaml')):
                    # Map models defined in this schema file
                    for model in analysis.get('models', []):
                        model_name = model.get('name', '')
                        if model_name:
                            schema_map[model_name] = {
                                'schema_path': path,
                                'schema_file': file,
                                'model_schema': model
                            }
            
            # Second pass to enhance each model with its dependencies
            for file in processed_files:
                if not file.get('is_dbt', False) or not file.get('path', '').endswith('.sql'):
                    continue
                
                analysis = file.get('analysis', {})
                model_name = analysis.get('model_name', '')
                
                if not model_name:
                    continue
                
                # Add schema information if available
                if model_name in schema_map:
                    schema_info = schema_map[model_name]['model_schema']
                    
                    # Only update if not already present in the analysis
                    if 'description' not in analysis and 'description' in schema_info:
                        analysis['description'] = schema_info.get('description', '')
                    
                    if 'columns' in schema_info:
                        analysis['schema_columns'] = schema_info.get('columns', [])
                    
                    if 'tests' in schema_info:
                        analysis['schema_tests'] = schema_info.get('tests', [])
                
                # Add references to models that depend on this model
                dependents = []
                jinja_refs = analysis.get('jinja_references', [])
                for ref_model in jinja_refs:
                    if ref_model in model_map:
                        ref_path = model_map[ref_model]['path']
                        dependents.append({
                            'model': ref_model,
                            'path': ref_path
                        })
                
                if dependents:
                    if 'dependencies' not in analysis:
                        analysis['dependencies'] = {'depends_on': {}, 'supports': []}
                    
                    analysis['dependencies']['depends_on']['models'] = jinja_refs
                    
                    # Update the supports field on referenced models
                    for ref_model in jinja_refs:
                        if ref_model in model_map:
                            ref_file = model_map[ref_model]['file']
                            ref_analysis = ref_file.get('analysis', {})
                            
                            if 'dependencies' not in ref_analysis:
                                ref_analysis['dependencies'] = {'depends_on': {}, 'supports': []}
                            
                            ref_analysis['dependencies']['supports'].append(model_name)
            
            logger.info(f"Linked dependencies for {len(model_map)} dbt models")
            
        except Exception as e:
            logger.error(f"Error linking dbt dependencies: {str(e)}")

    def _enhance_dbt_results_with_lineage(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance dbt results with lineage information by finding related models.
        """
        enhanced_results = []
        
        # Create a mapping of model names to result indices for faster lookups
        model_map = {}
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            model_name = metadata.get('model_name', '')
            if model_name:
                model_map[model_name] = i
        
        # For each result, find its upstream and downstream dependencies
        for result in results:
            metadata = result.get('metadata', {})
            model_name = metadata.get('model_name', '')
            
            if not model_name:
                enhanced_results.append(result)
                continue
            
            # Find upstream models (models this model depends on)
            upstream_models = metadata.get('references', '').split(',') if metadata.get('references', '') else []
            upstream_models = [m.strip() for m in upstream_models if m.strip()]
            
            # Find downstream models (models that depend on this model)
            downstream_models = []
            
            # Check if any other model has this model in its references
            for other_result in results:
                other_metadata = other_result.get('metadata', {})
                other_model = other_metadata.get('model_name', '')
                if not other_model or other_model == model_name:
                    continue
                    
                other_refs = other_metadata.get('references', '').split(',') if other_metadata.get('references', '') else []
                other_refs = [r.strip() for r in other_refs if r.strip()]
                
                # If this model is referenced by the other model, it's a downstream dependency
                if model_name in other_refs:
                    downstream_models.append(other_model)
            
            # Create enhanced result
            enhanced_result = result.copy()
            
            # Add lineage information
            if 'dbt_info' not in enhanced_result:
                enhanced_result['dbt_info'] = {}
            
            # Check if there's more detailed lineage info in the analysis metadata
            analysis = metadata.get('analysis', {})
            lineage_from_analysis = analysis.get('lineage', {})
            
            if lineage_from_analysis and isinstance(lineage_from_analysis, dict):
                # Use the more detailed lineage information if available
                enhanced_result['dbt_info']['lineage'] = lineage_from_analysis
            else:
                # Use what we've gathered
                enhanced_result['dbt_info']['lineage'] = {
                    'upstream': upstream_models,
                    'downstream': downstream_models
                }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results

    def _flatten_dict_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """Flatten and convert nested dictionary to a format suitable for ChromaDB metadata."""
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                # Keep primitive types as is (handle None as empty string)
                flattened[key] = value if value is not None else ""
            elif isinstance(value, dict):
                # For dictionaries, serialize to JSON string
                try:
                    flattened[key] = json.dumps(self._clean_for_json(value))
                except Exception as e:
                    logger.warning(f"Error serializing dict for key {key}: {str(e)}")
                    flattened[key] = str(value)
            elif isinstance(value, list):
                # For lists, serialize to JSON string
                try:
                    flattened[key] = json.dumps(self._clean_for_json(value))
                except Exception as e:
                    logger.warning(f"Error serializing list for key {key}: {str(e)}")
                    flattened[key] = str(value)
            else:
                # For any other type, convert to string
                flattened[key] = str(value)
        
        return flattened

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize JSON strings in metadata back into Python objects."""
        deserialized = {}
        
        for key, value in metadata.items():
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    deserialized[key] = json.loads(value)
                except json.JSONDecodeError:
                    deserialized[key] = value
            else:
                deserialized[key] = value
        
        return deserialized

# SQL Analyzer functions for code enhancement
def analyze_sql_model(sql_content: str, source_column_name: str = None) -> Dict[str, Any]:
    """
    Analyze a SQL model's structure, identifying CTEs, columns, joins, and other components.
    
    Args:
        sql_content (str): The SQL content to analyze
        source_column_name (str, optional): The name of a specific column to track
        
    Returns:
        Dict[str, Any]: A detailed analysis of the SQL structure
    """
    analysis = {
        "isDBTModel": _is_dbt_model(sql_content),
        "hasCTEs": _has_ctes(sql_content),
        "ctes": [],
        "finalCTE": None,
        "finalSelect": None,
        "columns": [],
        "joins": [],
        "groupBys": [],
        "hasSourceColumn": source_column_name is not None and source_column_name in sql_content,
        "sourceColumnLocations": _find_column_references(sql_content, source_column_name) if source_column_name else []
    }
    
    # Extract all CTEs
    analysis["ctes"] = _extract_ctes(sql_content)
    
    # Find the final SELECT or final CTE
    final_cte_match = re.search(r'final\s+as\s+\(\s*\n([\s\S]*?)(?=\)\s*\n+select|\)$)', sql_content)
    final_select_match = re.search(r'select\s+[\s\S]*?from[\s\S]*?$', sql_content)
    
    if final_cte_match:
        analysis["finalCTE"] = {
            "content": final_cte_match.group(1),
            "columns": _extract_columns(final_cte_match.group(1))
        }
    
    if final_select_match:
        analysis["finalSelect"] = {
            "content": final_select_match.group(0),
            "columns": _extract_columns(final_select_match.group(0))
        }
    
    # Extract joins
    join_regex = r'(inner|left|right|full|cross)?\s*join\s+(\w+)\s+(?:as\s+)?(\w+)?\s+on\s+(.*?)(?=\s+(?:inner|left|right|full|cross)?\s*join|\s+where|\s+group\s+by|\s+order\s+by|\s*$)'
    analysis["joins"] = []
    for match in re.finditer(join_regex, sql_content, re.IGNORECASE | re.DOTALL):
        analysis["joins"].append({
            "type": match.group(1) or "inner",
            "table": match.group(2),
            "alias": match.group(3) or match.group(2),
            "condition": match.group(4)
        })
    
    # Extract GROUP BY clauses
    group_by_match = re.search(r'group\s+by\s+(.*?)(?=having|order\s+by|limit|$)', sql_content, re.IGNORECASE)
    if group_by_match:
        analysis["groupBys"] = [col.strip() for col in group_by_match.group(1).split(',')]
    
    # If source column is specified, check specifically where it appears
    if source_column_name:
        for i, cte in enumerate(analysis["ctes"]):
            if source_column_name in cte["content"]:
                cte["hasSourceColumn"] = True
                
                # Check if it's used in an aggregation in this CTE
                agg_pattern = rf'(sum|avg|count|min|max)\s*\(\s*{re.escape(source_column_name)}\s*\)'
                cte["sourceColumnInAggregation"] = bool(re.search(agg_pattern, cte["content"], re.IGNORECASE))
            else:
                cte["hasSourceColumn"] = False
                cte["sourceColumnInAggregation"] = False
    
    return analysis

def _is_dbt_model(sql_content: str) -> bool:
    """Check if the SQL content is a dbt model."""
    return '{{ ref(' in sql_content or '{{ref(' in sql_content

def _has_ctes(sql_content: str) -> bool:
    """Check if the SQL content has CTEs."""
    return 'with ' in sql_content and ' as (' in sql_content

def _extract_ctes(sql_content: str) -> List[Dict[str, Any]]:
    """Extract all CTEs from the SQL content."""
    ctes = []
    cte_regex = r'(\w+)\s+as\s+\(\s*\n([\s\S]*?)(?=\),\s*\n\w+\s+as\s+\(|\),\s*\nfinal\s+as\s+\(|\)\s*\n+select|\)$)'
    
    for match in re.finditer(cte_regex, sql_content, re.DOTALL):
        cte_name = match.group(1)
        cte_content = match.group(2)
        
        # Analyze the CTE content
        cte_type = _determine_cte_type(cte_content)
        cte_columns = _extract_columns(cte_content)
        
        ctes.append({
            "name": cte_name,
            "content": cte_content,
            "type": cte_type,
            "columns": cte_columns,
            "isAggregation": cte_type == 'aggregation'
        })
    
    return ctes

def _determine_cte_type(cte_content: str) -> str:
    """Determine the type of a CTE based on its content."""
    if re.search(r'(sum|avg|count|min|max)\s*\(', cte_content, re.IGNORECASE):
        return 'aggregation' if 'group by' in cte_content.lower() else 'calculation'
    elif 'join' in cte_content.lower():
        return 'join'
    elif 'where' in cte_content.lower():
        return 'filter'
    else:
        return 'base'

def _extract_columns(sql_segment: str) -> List[Dict[str, Any]]:
    """Extract columns from a SQL segment."""
    columns = []
    select_match = re.search(r'select\s+([\s\S]*?)(?=from)', sql_segment, re.IGNORECASE)
    
    if select_match:
        select_clause = select_match.group(1)
        # Split by commas, but handle complex expressions
        depth = 0
        current_column = ''
        
        for char in select_clause:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            if char == ',' and depth == 0:
                columns.append(current_column.strip())
                current_column = ''
            else:
                current_column += char
        
        if current_column.strip():
            columns.append(current_column.strip())
    
    return [_parse_column(col) for col in columns]

def _parse_column(column_definition: str) -> Dict[str, Any]:
    """Parse a column definition into a structured format."""
    as_match = re.search(r'(?:.*\s+as\s+)(\w+)$', column_definition, re.IGNORECASE)
    name = as_match.group(1) if as_match else column_definition.split('.')[-1].strip()
    
    return {
        "fullDefinition": column_definition,
        "name": name,
        "isAggregation": any(agg in column_definition.lower() for agg in ['sum(', 'avg(', 'count(', 'min(', 'max('])
    }

def _find_column_references(sql_content: str, column_name: str) -> List[Dict[str, Any]]:
    """Find all references to a specific column in the SQL content."""
    if not column_name:
        return []
    
    references = []
    lines = sql_content.split('\n')
    
    for i, line in enumerate(lines):
        if column_name in line:
            # Check the context of the reference
            context = "unknown"
            if any(agg + '(' + column_name in line.lower() for agg in ['sum', 'avg', 'count', 'min', 'max']):
                context = "aggregation"
            elif 'select' in line.lower():
                context = "select"
            elif 'where' in line.lower():
                context = "filter"
            elif 'group by' in line.lower():
                context = "groupby"
            elif 'order by' in line.lower():
                context = "orderby"
            elif 'join' in line.lower():
                context = "join"
            
            references.append({
                "lineNumber": i + 1,
                "context": context,
                "line": line.strip()
            })
    
    return references

def find_best_modification_target(analysis: Dict[str, Any], source_column_name: str) -> Dict[str, Any]:
    """
    Find the best target for adding a new aggregation based on an existing column.
    
    Args:
        analysis (Dict[str, Any]): Analysis of the SQL model
        source_column_name (str): The source column to base the aggregation on
        
    Returns:
        Dict[str, Any]: Details of the best modification target
    """
    result = {
        "targetType": None,  # 'aggregation_cte', 'new_cte', 'select_clause'
        "targetName": None,  # Name of CTE or clause
        "targetIndex": -1,   # Index in the list of CTEs
        "reason": "",        # Reason for selection
        "modificationType": None # 'add_to_existing', 'create_new', 'modify_select'
    }
    
    # If there are no CTEs, we'll need to modify the main SELECT
    if not analysis["hasCTEs"]:
        result.update({
            "targetType": "select_clause",
            "reason": "No CTEs found, will modify main SELECT statement",
            "modificationType": "modify_select"
        })
        return result
    
    # Look for the best CTE to modify
    target_cte = None
    target_cte_index = -1
    
    # First priority: aggregation CTEs that already have the source column
    for i, cte in enumerate(analysis["ctes"]):
        if cte["type"] == "aggregation" and "content" in cte and source_column_name in cte["content"]:
            target_cte = cte
            target_cte_index = i
            result.update({
                "targetType": "aggregation_cte",
                "targetName": cte["name"],
                "targetIndex": i,
                "reason": f"Found existing aggregation CTE '{cte['name']}' that already uses column '{source_column_name}'",
                "modificationType": "add_to_existing"
            })
            return result
    
    # Second priority: any aggregation CTE
    for i, cte in enumerate(analysis["ctes"]):
        if cte["type"] == "aggregation":
            target_cte = cte
            target_cte_index = i
            result.update({
                "targetType": "aggregation_cte",
                "targetName": cte["name"],
                "targetIndex": i,
                "reason": f"Using existing aggregation CTE '{cte['name']}' for new aggregation",
                "modificationType": "add_to_existing"
            })
            return result
    
    # Third priority: any CTE with the source column - we'll create a new aggregation CTE after it
    for i, cte in enumerate(analysis["ctes"]):
        if "content" in cte and source_column_name in cte["content"]:
            result.update({
                "targetType": "new_cte",
                "targetName": f"{cte['name']}_agg",
                "targetIndex": i,
                "reason": f"Will create new aggregation CTE after '{cte['name']}' which contains source column",
                "modificationType": "create_new",
                "baseCteName": cte["name"]
            })
            return result
    
    # Last resort: create a new aggregation CTE after the last CTE
    if analysis["ctes"]:
        last_cte = analysis["ctes"][-1]
        result.update({
            "targetType": "new_cte",
            "targetName": "aggregations",
            "targetIndex": len(analysis["ctes"]) - 1,
            "reason": "Will create new aggregation CTE at the end, as no suitable target was found",
            "modificationType": "create_new",
            "baseCteName": last_cte["name"]
        })
    else:
        # Fallback to modifying the main SELECT
        result.update({
            "targetType": "select_clause",
            "reason": "No suitable CTEs found, will modify main SELECT statement",
            "modificationType": "modify_select"
        })
    
    return result

def generate_enhancement_modifications(
    sql_content: str, 
    analysis: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """
    Generate the modifications needed to add a new aggregated column.
    
    Args:
        sql_content (str): Original SQL content
        analysis (Dict[str, Any]): Analysis of the SQL model
        source_column_name (str): Source column for aggregation
        new_column_name (str): Name for the new column
        aggregation_type (str): Type of aggregation (sum, avg, count, min, max)
        
    Returns:
        Dict[str, Any]: Result containing success flag, modified code, and details
    """
    # Find the best modification target
    target = find_best_modification_target(analysis, source_column_name)
    
    if target["targetType"] == "aggregation_cte":
        return _modify_existing_aggregation_cte(
            sql_content, analysis, target, source_column_name, new_column_name, aggregation_type
        )
    elif target["targetType"] == "new_cte":
        return _create_new_aggregation_cte(
            sql_content, analysis, target, source_column_name, new_column_name, aggregation_type
        )
    elif target["targetType"] == "select_clause":
        return _modify_select_clause(
            sql_content, analysis, source_column_name, new_column_name, aggregation_type
        )
    else:
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": "Couldn't determine appropriate modification strategy"
        }

def _modify_existing_aggregation_cte(
    sql_content: str, 
    analysis: Dict[str, Any], 
    target: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """Modify an existing aggregation CTE to add a new aggregated column."""
    modified_code = sql_content
    target_cte = analysis["ctes"][target["targetIndex"]]
    
    # Find existing aggregation pattern for indentation
    agg_pattern = r'(\s+)(?:sum|avg|count|min|max)\s*\([^)]+\)\s+as\s+[\w_]+'
    indentation_match = re.search(agg_pattern, target_cte["content"], re.IGNORECASE)
    indentation = indentation_match.group(1) if indentation_match else '        '
    
    # Create the new aggregation line with proper indentation
    aggregation_line = f"{indentation}{aggregation_type}({source_column_name}) as {new_column_name},"
    
    # Find where to insert in the CTE
    cte_pattern = r'(' + re.escape(target_cte["name"]) + r'\s+as\s+\(\s*\n\s*select[\s\S]*?)(\s+from\s+)'
    cte_match = re.search(cte_pattern, sql_content, re.IGNORECASE)
    
    if not cte_match:
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": f"Could not locate SELECT statement in target CTE '{target_cte['name']}'"
        }
    
    select_portion = cte_match.group(1)
    
    # Find the last aggregation in the SELECT
    last_agg_index = max(
        select_portion.lower().rfind('sum('),
        select_portion.lower().rfind('avg('),
        select_portion.lower().rfind('count('),
        select_portion.lower().rfind('min('),
        select_portion.lower().rfind('max(')
    )
    
    if last_agg_index == -1:
        # No existing aggregations, add after the SELECT
        select_index = select_portion.lower().rfind('select')
        if select_index != -1:
            line_end_index = select_portion.find('\n', select_index)
            if line_end_index != -1:
                insertion = (
                    select_portion[:line_end_index + 1] + 
                    f"{indentation}-- Calculate the {aggregation_type} of {source_column_name}\n" +
                    f"{aggregation_line}\n" + 
                    select_portion[line_end_index + 1:]
                )
                modified_code = modified_code.replace(select_portion, insertion)
            else:
                return {
                    "success": False,
                    "enhancedCode": sql_content,
                    "reason": "Could not find end of SELECT line in CTE"
                }
        else:
            return {
                "success": False,
                "enhancedCode": sql_content,
                "reason": "Could not find SELECT statement in CTE"
            }
    else:
        # Insert after the last aggregation line
        line_end_index = select_portion.find('\n', last_agg_index)
        if line_end_index != -1:
            insertion = (
                select_portion[:line_end_index + 1] + 
                f"{indentation}-- Calculate the {aggregation_type} of {source_column_name}\n" +
                f"{aggregation_line}\n" + 
                select_portion[line_end_index + 1:]
            )
            modified_code = modified_code.replace(select_portion, insertion)
        else:
            return {
                "success": False,
                "enhancedCode": sql_content,
                "reason": "Could not find end of last aggregation line"
            }
    
    # Now add the column to the final SELECT or final CTE
    if analysis["finalCTE"]:
        # Add to the final CTE
        final_pattern = r'(final\s+as\s+\(\s*\n\s*select[\s\S]*?' + re.escape(target_cte["name"]) + r'\.\w+,[\s\S]*?)(?=\s+from\s+)'
        final_match = re.search(final_pattern, modified_code, re.IGNORECASE)
        
        if final_match:
            indent_match = re.search(r'\n(\s+)\w', final_match.group(1))
            final_indent = indent_match.group(1) if indent_match else '        '
            
            # Add the column to the final select
            final_insertion = final_match.group(1) + f"\n{final_indent}{target_cte['name']}.{new_column_name},"
            modified_code = modified_code.replace(final_match.group(1), final_insertion)
            
            return {
                "success": True,
                "enhancedCode": modified_code,
                "details": {
                    "modifiedCte": target_cte["name"],
                    "addedColumn": new_column_name,
                    "addedToFinal": True
                }
            }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Added column to CTE but could not locate target in final SELECT"
            }
    elif analysis["finalSelect"]:
        # Add to the main SELECT statement
        select_index = modified_code.lower().rfind('select')
        from_index = modified_code.lower().find('from', select_index)
        
        if select_index != -1 and from_index != -1:
            select_clause = modified_code[select_index:from_index]
            indent_match = re.search(r'\n(\s+)\w', select_clause)
            select_indent = indent_match.group(1) if indent_match else '    '
            
            # Add column to the select clause
            select_insertion = select_clause + f"\n{select_indent}{target_cte['name']}.{new_column_name},"
            modified_code = modified_code.replace(select_clause, select_insertion)
            
            return {
                "success": True,
                "enhancedCode": modified_code,
                "details": {
                    "modifiedCte": target_cte["name"],
                    "addedColumn": new_column_name,
                    "addedToFinal": True
                }
            }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Added column to CTE but could not locate main SELECT"
            }
    else:
        return {
            "success": False,
            "enhancedCode": modified_code,
            "reason": "Added column to CTE but could not find a final SELECT to add it to"
        }

def _create_new_aggregation_cte(
    sql_content: str, 
    analysis: Dict[str, Any], 
    target: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """Create a new aggregation CTE after an existing CTE."""
    base_cte_name = target.get("baseCteName")
    new_cte_name = target["targetName"]
    
    # Find the base CTE to insert after
    cte_pattern = re.escape(base_cte_name) + r'\s+as\s+\([\s\S]*?\),\s*\n'
    cte_match = re.search(cte_pattern, sql_content, re.IGNORECASE)
    
    if not cte_match:
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": f"Could not locate base CTE '{base_cte_name}' in SQL content"
        }
    
    # Create a new aggregation CTE
    new_cte = (
        f"{base_cte_name} as ([\s\S]*?\\),\n"
        f"{new_cte_name} as (\n"
        f"    select\n"
        f"        {base_cte_name}.*,\n"
        f"        {aggregation_type}({source_column_name}) as {new_column_name}\n"
        f"    from {base_cte_name}\n"
        f"    group by 1\n"
        f"),\n"
    )
    
    modified_code = re.sub(
        re.escape(cte_match.group(0)),
        new_cte,
        sql_content
    )
    
    # Add to final CTE or select
    if analysis["finalCTE"]:
        final_select_pattern = r'final\s+as\s+\(\s*\n\s*select\s+([\s\S]*?)(?=\s+from\s+)'
        final_select_match = re.search(final_select_pattern, modified_code, re.IGNORECASE)
        
        if final_select_match:
            indent_match = re.search(r'\n(\s+)\w', final_select_match.group(1))
            final_indent = indent_match.group(1) if indent_match else '        '
            
            final_insertion = final_select_match.group(1) + f"\n{final_indent}{new_cte_name}.{new_column_name},"
            modified_code = modified_code.replace(final_select_match.group(1), final_insertion)
            
            # Update the from clause to join the new CTE
            from_pattern = r'from\s+([\s\S]*?)(?=where|group|order|$)'
            from_match = re.search(from_pattern, modified_code, re.IGNORECASE)
            
            if from_match and new_cte_name not in from_match.group(1):
                join_indent = final_indent
                join_clause = (
                    from_match.group(1) + 
                    f"\n{join_indent}left join {new_cte_name}\n"
                    f"{join_indent}    on {base_cte_name}.{base_cte_name}_key = {new_cte_name}.{base_cte_name}_key"
                )
                modified_code = modified_code.replace(from_match.group(1), join_clause)
                
                return {
                    "success": True,
                    "enhancedCode": modified_code,
                    "details": {
                        "createdCte": new_cte_name,
                        "basedOn": base_cte_name,
                        "addedColumn": new_column_name,
                        "addedToFinal": True
                    }
                }
            else:
                return {
                    "success": False,
                    "enhancedCode": modified_code,
                    "reason": "Created new CTE but could not update FROM clause with join"
                }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Created new CTE but could not find final SELECT to add column to"
            }
    else:
        return {
            "success": False,
            "enhancedCode": modified_code,
            "reason": "Created new CTE but could not find a final SELECT or final CTE"
        }

def _modify_select_clause(
    sql_content: str, 
    analysis: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """Modify a simple SELECT statement to add a new aggregated column."""
    if not (sql_content.lower().find('select') != -1 and sql_content.lower().find('from') != -1):
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": "Could not find basic SELECT...FROM structure in SQL"
        }
    
    select_index = sql_content.lower().find('select')
    from_index = sql_content.lower().find('from', select_index)
    
    select_clause = sql_content[select_index:from_index]
    
    # Look for indentation pattern
    indent_match = re.search(r'\n(\s+)\w', select_clause)
    select_indent = indent_match.group(1) if indent_match else '    '
    
    # Add column to the select clause
    select_insertion = select_clause + f"\n{select_indent}{aggregation_type}({source_column_name}) as {new_column_name},"
    
    # Check if we need to add a GROUP BY
    has_group_by = 'group by' in sql_content.lower()
    
    if not has_group_by:
        # Add group by if not present
        modified_code = sql_content.replace(select_clause, select_insertion)
        
        # Find a good place to add the GROUP BY
        order_by_index = modified_code.lower().find('order by')
        limit_index = modified_code.lower().find('limit')
        insert_index = order_by_index if order_by_index != -1 else (
            limit_index if limit_index != -1 else len(modified_code)
        )
        
        # Try to identify columns to group by (exclude aggregated columns)
        # This is an approximation - would need more complex parsing for accuracy
        potential_group_cols = []
        for col in select_clause.split(','):
            # Skip columns with aggregations
            if any(agg in col.lower() for agg in ['sum(', 'avg(', 'count(', 'min(', 'max(']):
                continue
            # Extract the column name
            col_name = col.strip()
            if ' as ' in col_name.lower():
                col_name = col_name.split(' as ')[-1].strip()
            potential_group_cols.append(col_name)
        
        group_by_clause = ""
        if potential_group_cols:
            group_by_clause = (
                f"\n{select_indent}group by\n"
                f"{select_indent}    {','.join(potential_group_cols)}\n"
            )
            
            # Insert the GROUP BY
            modified_code = (
                modified_code[:insert_index] + 
                group_by_clause + 
                modified_code[insert_index:]
            )
            
            return {
                "success": True,
                "enhancedCode": modified_code,
                "details": {
                    "addedColumn": new_column_name,
                    "addedGroupBy": True
                }
            }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Added column but could not identify appropriate GROUP BY columns"
            }
    else:
        # Just add the column if GROUP BY already exists
        modified_code = sql_content.replace(select_clause, select_insertion)
        return {
            "success": True,
            "enhancedCode": modified_code,
            "details": {
                "addedColumn": new_column_name,
                "addedGroupBy": False
            }
        }

def generate_suggested_approach(
    analysis: Dict[str, Any], 
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> str:
    """Generate suggested approach for manual implementation when automatic modification fails."""
    suggestions = []
    
    if analysis["ctes"]:
        # Find potential places for modification
        aggregation_ctes = [cte for cte in analysis["ctes"] if cte["type"] == "aggregation"]
        source_ctes = [cte for cte in analysis["ctes"] if source_column_name in cte.get("content", "")]
        
        if aggregation_ctes:
            target_cte = next((cte for cte in aggregation_ctes if source_column_name in cte.get("content", "")), aggregation_ctes[0])
            suggestions.append(f"-- 1. Add to the '{target_cte['name']}' CTE: {aggregation_type}({source_column_name}) as {new_column_name}")
        elif source_ctes:
            source_cte = source_ctes[0]
            suggestions.append(f"-- 1. Create a new aggregation CTE after '{source_cte['name']}' that computes {aggregation_type}({source_column_name}) as {new_column_name}")
        
        # Add suggestion for the final SELECT or final CTE
        if analysis["finalCTE"]:
            suggestions.append(f"-- 2. Add the new column to the final CTE SELECT statement")
        else:
            suggestions.append(f"-- 2. Add the new column to the main SELECT statement")
    else:
        # Simple SQL suggestions
        suggestions.append(f"-- 1. Add {aggregation_type}({source_column_name}) as {new_column_name} to the SELECT clause")
        
        if not analysis["groupBys"]:
            suggestions.append(f"-- 2. Add an appropriate GROUP BY clause for non-aggregated columns")
    
    return '\n'.join(suggestions)

def generate_failure_feedback(
    sql_content: str, 
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> str:
    """Generate detailed feedback when modification fails."""
    feedback = []
    
    # Check for source column existence
    if source_column_name not in sql_content:
        feedback.append(f"-- The source column '{source_column_name}' could not be found in the model.")
        feedback.append(f"-- Check for typos or ensure this column exists before aggregating it.")
    else:
        feedback.append(f"-- The source column '{source_column_name}' was found, but the structure is complex.")
    
    # Analyze model structure
    if 'with ' in sql_content and ' as (' in sql_content:
        feedback.append(f"-- This appears to be a model with CTEs. You should add the aggregation to an appropriate CTE")
        feedback.append(f"-- and then reference it in the final SELECT statement.")
    elif 'select' in sql_content and 'from' in sql_content:
        feedback.append(f"-- This appears to be a simple SELECT query. Add the aggregation to the SELECT clause")
        feedback.append(f"-- and add an appropriate GROUP BY clause if needed.")
    
    return '\n'.join(feedback)