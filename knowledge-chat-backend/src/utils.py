from src.patches import patch_chromadb_numpy
# Apply patch before importing chromadb
patch_chromadb_numpy()

import os
from typing import Optional, List, Dict, Any
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
            
            # Initialize code analyzer if not already done
            if not hasattr(self, 'code_analyzer'):
                self.code_analyzer = CodeAnalyzer()
            
            # Analyze code once and cache the results
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}
            
            cache_key = f"{file_path}:{language}"
            if cache_key not in self._analysis_cache:
                analysis = self.code_analyzer.analyze_file(file_path, language)
                self._analysis_cache[cache_key] = analysis
            else:
                analysis = self._analysis_cache[cache_key]
            
            logger.info(f"Code analysis complete: {analysis}")
            
            # Flatten and stringify analysis for ChromaDB metadata
            flattened_metadata = self._prepare_metadata_for_chroma(analysis)
            
            # Set language-specific splitting
            if hasattr(Language, language.upper()):
                self.code_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=getattr(Language, language.upper()),
                    chunk_size=500,
                    chunk_overlap=50
                )
            
            # Load and split code
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Create initial document
            doc = Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'language': language,
                    **flattened_metadata  # Use flattened metadata
                }
            )
            
            # Split into chunks
            chunks = self.code_splitter.split_documents([doc])
            
            # Enhance chunks with metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'language': language,
                    'file_path': str(file_path),
                    **flattened_metadata  # Use flattened metadata
                })
            
            logger.info(f"Split code into {len(chunks)} chunks with analysis metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing code: {str(e)}", exc_info=True)
            raise

    def _prepare_metadata_for_chroma(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Prepare analysis metadata for ChromaDB by flattening and converting to primitive types."""
        flattened = {}
        
        try:
            # Handle tables
            if 'tables' in analysis:
                flattened['tables'] = ','.join(sorted([str(t) for t in analysis['tables']]))
            
            # Handle relationships
            if 'relationships' in analysis and analysis['relationships']:
                relationships = []
                for rel in analysis['relationships']:
                    rel_str = f"{rel.get('left_table')}.{rel.get('left_column')} -> {rel.get('right_table')}.{rel.get('right_column')}"
                    relationships.append(rel_str)
                flattened['relationships'] = '|'.join(relationships)
            
            # Handle column lineage
            if 'column_lineage' in analysis and analysis['column_lineage']:
                lineage = []
                for item in analysis['column_lineage']:
                    source = item.get('source', {})
                    lineage_str = f"{source.get('table')}.{source.get('column')} -> {item.get('target_column')}"
                    lineage.append(lineage_str)
                flattened['column_lineage'] = '|'.join(lineage)
            
            # Handle business context
            if 'business_context' in analysis and analysis['business_context']:
                contexts = []
                for ctx in analysis['business_context']:
                    ctx_str = f"{ctx.get('category')}: {ctx.get('description')}"
                    contexts.append(ctx_str)
                flattened['business_context'] = '|'.join(contexts)
            
            # Handle query patterns
            if 'query_patterns' in analysis and analysis['query_patterns']:
                flattened['query_patterns'] = str(analysis['query_patterns'])
            
            # Add analysis summary
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

    def _flatten_analysis(self, analysis: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
        """Flatten nested analysis dictionary into string values."""
        flattened = {}
        
        if isinstance(analysis, dict):
            if 'error' in analysis:
                # Handle error case
                flattened['analysis_error'] = str(analysis['error'])
                return flattened
            
            for key, value in analysis.items():
                new_key = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    # Convert complex structures to string representation
                    flattened[new_key] = str(value)
                elif isinstance(value, (str, int, float, bool)):
                    # Keep primitive types as is
                    flattened[new_key] = value
                else:
                    # Convert any other types to string
                    flattened[new_key] = str(value)
        else:
            # If analysis is not a dict, store as string
            flattened['analysis'] = str(analysis)
        
        return flattened

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
        Process a GitHub repository by cloning it and analyzing its contents.
        
        Args:
            repo_url: URL of the GitHub repository
            username: GitHub username for authentication (optional)
            token: GitHub token for authentication (optional)
            
        Returns:
            Dictionary with repository metadata and processed files
        """
        logger.info(f"Processing GitHub repository: {repo_url}")
        
        # Parse the repository URL and remove .git if present
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL format: {repo_url}")
        
        owner = path_parts[0]
        repo_name = path_parts[1]
        
        # Remove .git suffix if present
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        # Create a temporary directory for the repository
        temp_dir = tempfile.mkdtemp(prefix="github_repo_")
        
        try:
            # Construct the clone URL with authentication if provided
            # Make sure we use the clean repo name without .git suffix
            base_clone_url = f"https://github.com/{owner}/{repo_name}.git"
            
            if username and token:
                # Format: https://username:token@github.com/owner/repo.git
                clone_url = f"https://{username}:{token}@github.com/{owner}/{repo_name}.git"
            elif token:
                # Format: https://token@github.com/owner/repo.git
                clone_url = f"https://{token}@github.com/{owner}/{repo_name}.git"
            else:
                clone_url = base_clone_url
            
            # Clone the repository
            logger.info(f"Cloning repository to {temp_dir}")
            repo = git.Repo.clone_from(clone_url, temp_dir)
            
            # Get repository metadata
            repo_metadata = {
                "url": repo_url,
                "owner": owner,
                "name": repo_name,
                "last_commit": str(repo.head.commit.hexsha),
                "last_commit_date": str(repo.head.commit.committed_datetime),
                "description": repo.description if hasattr(repo, 'description') else "",
                "file_count": 0,
                "languages": {}
            }
            
            # Process each file in the repository
            processed_files = []
            language_counts = {}
            
            # Walk through the repository directory
            for root, dirs, files in os.walk(temp_dir):
                # Skip .git directory
                if '.git' in dirs:
                    dirs.remove('.git')
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    
                    # Skip binary files
                    if self._is_binary_file(file_path):
                        continue
                    
                    # Get file extension
                    _, ext = os.path.splitext(file)
                    ext = ext.lstrip('.').lower()
                    
                    # Determine language
                    language = self._get_language_from_extension(ext)
                    
                    # Skip files with unknown language
                    if not language:
                        continue
                    
                    # Update language counts
                    language_counts[language] = language_counts.get(language, 0) + 1
                    
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Skip empty files
                        if not content.strip():
                            continue
                        
                        # Analyze file with CodeAnalyzer if applicable
                        analysis = None
                        try:
                            if language in ['python', 'sql', 'javascript', 'typescript']:
                                analyzer = CodeAnalyzer()
                                # Skip analysis for now, just store the file content
                                analysis = {"file_type": language}
                        except Exception as analysis_error:
                            logger.warning(f"Error analyzing file {rel_path}: {str(analysis_error)}")
                            analysis = {"error": str(analysis_error)}
                        
                        processed_files.append({
                            "path": rel_path,
                            "language": language,
                            "content": content,
                            "analysis": analysis
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing file {rel_path}: {str(e)}")
            
            # Update repository metadata
            repo_metadata["file_count"] = len(processed_files)
            repo_metadata["languages"] = language_counts
            
            # Create documents for vector storage
            documents = []
            
            for file in processed_files:
                # Create a document for each file
                doc_id = f"{owner}_{repo_name}_{base64.urlsafe_b64encode(file['path'].encode()).decode()}"
                
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
                    
                    # Create metadata
                    metadata = {
                        "repo_url": repo_url,
                        "repo_owner": owner,
                        "repo_name": repo_name,
                        "file_path": file['path'],
                        "language": file['language'],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                    
                    # Add analysis data if available
                    if file['analysis']:
                        metadata["analysis"] = file['analysis']
                    
                    # Add the document
                    documents.append({
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": metadata
                    })
            
            return {
                "metadata": repo_metadata,
                "documents": documents
            }
            
        except Exception as e:
            logger.error(f"Error processing GitHub repository: {str(e)}")
            raise
        
        finally:
            # Clean up the temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {str(e)}")

    def add_github_repo(self, repo_url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a processed GitHub repository to the vector store.
        """
        try:
            # Process the repository
            repo_data = self.process_github(repo_url, 
                                           username=metadata.get("username", ""), 
                                           token=metadata.get("token", ""))
            
            # Get the collection
            collection = self.get_or_create_collection("github_documents")
            
            # Add documents to the collection
            documents = repo_data["documents"]
            
            if not documents:
                return {
                    "status": "warning",
                    "message": "No documents were extracted from the repository",
                    "repo_url": repo_url
                }
            
            # Add documents in batches to avoid overwhelming the database
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Extract data for ChromaDB
                ids = [doc["id"] for doc in batch]
                texts = [doc["text"] for doc in batch]
                
                # Clean metadata to ensure all values are valid types
                cleaned_metadatas = []
                for doc in batch:
                    # Create a simplified metadata dict with only primitive types
                    cleaned_metadata = {
                        "repo_url": repo_url,
                        "file_path": doc["metadata"].get("file_path", ""),
                        "language": doc["metadata"].get("language", ""),
                        "chunk_index": doc["metadata"].get("chunk_index", 0),
                        "total_chunks": doc["metadata"].get("total_chunks", 1),
                        "repo_owner": doc["metadata"].get("repo_owner", ""),
                        "repo_name": doc["metadata"].get("repo_name", ""),
                        "upload_time": metadata.get("upload_time", "")
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
            
            return {
                "status": "success",
                "message": f"Added {len(documents)} documents from GitHub repository",
                "repo_url": repo_url,
                "repo_metadata": {
                    "url": repo_url,
                    "owner": repo_data["metadata"].get("owner", ""),
                    "name": repo_data["metadata"].get("name", ""),
                    "file_count": repo_data["metadata"].get("file_count", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error adding GitHub repository: {str(e)}")
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
        return extension_map.get(ext)