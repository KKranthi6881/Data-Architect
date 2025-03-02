from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
import logging
from src.utils import ChromaDBManager
from langchain.tools import Tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SearchTools:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def search_code(self, query: str) -> Dict[str, Any]:
        """
        Search through code documents
        
        Args:
            query: The search query
            
        Returns:
            Dict containing search results
        """
        try:
            # Search in SQL and Python collections
            sql_results = self.db_manager.code_similarity_search(
                collection_name="sql_documents",
                code_snippet=query,
                n_results=3
            )
            python_results = self.db_manager.code_similarity_search(
                collection_name="py_documents",
                code_snippet=query,
                n_results=3
            )
            
            # Format results for the agent
            formatted_results = []
            
            # Format SQL results
            if 'results' in sql_results:
                for result in sql_results['results']:
                    formatted_results.append({
                        'content': result.get('code', ''),
                        'source': 'SQL',
                        'file_info': result.get('file_info', {}),
                        'similarity': result.get('similarity', 0)
                    })
            
            # Format Python results
            if 'results' in python_results:
                for result in python_results['results']:
                    formatted_results.append({
                        'content': result.get('code', ''),
                        'source': 'Python',
                        'file_info': result.get('file_info', {}),
                        'similarity': result.get('similarity', 0)
                    })
            
            return {
                "tool": "code_search",
                "tool_input": query,
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error in code search: {str(e)}")
            return {"error": str(e), "results": []}

    def search_documentation(self, query: str) -> Dict[str, Any]:
        """
        Search through documentation
        
        Args:
            query: The search query
            
        Returns:
            Dict containing documentation results
        """
        try:
            # Search in PDF documents
            doc_results = self.db_manager.hybrid_search(
                collection_name="pdf_documents",
                query=query,
                n_results=3
            )
            
            return {
                "results": doc_results.get("results", [])
            }
        except Exception as e:
            logger.error(f"Error in documentation search: {str(e)}")
            return {"error": str(e), "results": []}

    def search_relationships(self, query: str) -> Dict[str, Any]:
        """
        Search for relationships between code and documentation
        
        Args:
            query: The search query
            
        Returns:
            Dict containing relationship information
        """
        try:
            # Use code similarity search for SQL to find relationships
            sql_results = self.db_manager.code_similarity_search(
                collection_name="sql_documents",
                code_snippet=query,
                n_results=2
            )
            
            relationships = []
            if sql_results and 'results' in sql_results:
                for result in sql_results['results']:
                    if 'file_info' in result and 'relationships' in result['file_info']:
                        relationships.append(result['file_info']['relationships'])
            
            return {
                "results": relationships
            }
        except Exception as e:
            logger.error(f"Error in relationship search: {str(e)}")
            return {"error": str(e), "results": {}}

    # Create Tool objects for the agent
    @property
    def tools(self):
        return [
            Tool(
                name="search_code",
                func=self.search_code,
                description="Search through code files (SQL and Python)"
            ),
            Tool(
                name="search_documentation",
                func=self.search_documentation,
                description="Search through documentation files"
            ),
            Tool(
                name="search_relationships",
                func=self.search_relationships,
                description="Search for relationships between code and documentation"
            )
        ]

    @tool("code_search")
    def search_code_old(self, query: str) -> Dict[str, Any]:
        """
        Search through code collections for relevant code snippets.
        
        Args:
            query: The search query to find relevant code
            
        Returns:
            Dict containing code search results with context
        """
        try:
            collections = self.db_manager.client.list_collections()
            code_collections = [c.name for c in collections 
                              if any(ext in c.name.lower() 
                                    for ext in ['py', 'sql', 'yml', 'yaml'])]
            
            code_results = []
            for collection_name in code_collections:
                try:
                    results = self.db_manager.code_similarity_search(
                        collection_name=collection_name,
                        code_snippet=query,
                        n_results=2
                    )
                    if results and 'results' in results:
                        code_results.extend(results['results'])
                except Exception as e:
                    logger.warning(f"Error searching code collection {collection_name}: {str(e)}")
            
            # Sort results by similarity
            code_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Format code context
            code_context = []
            for result in code_results[:3]:
                context = {
                    'code': result.get('code', ''),
                    'file_info': result.get('file_info', {}),
                    'matched_lines': result.get('matched_lines', []),
                    'similarity': result.get('similarity', 0)
                }
                code_context.append(context)
                
            return {
                "status": "success",
                "results": code_context
            }
            
        except Exception as e:
            logger.error(f"Error in code search: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    @tool("doc_search")
    def search_documentation_old(self, query: str) -> Dict[str, Any]:
        """
        Search through documentation collections for relevant information.
        
        Args:
            query: The search query to find relevant documentation
            
        Returns:
            Dict containing documentation search results with context
        """
        try:
            collections = self.db_manager.client.list_collections()
            doc_collections = [c.name for c in collections 
                             if not any(ext in c.name.lower() 
                                      for ext in ['py', 'sql', 'yml', 'yaml'])]
            
            doc_results = []
            for collection_name in doc_collections:
                try:
                    results = self.db_manager.hybrid_search(
                        collection_name=collection_name,
                        query=query,
                        n_results=2
                    )
                    if results and 'results' in results:
                        doc_results.extend(results['results'])
                except Exception as e:
                    logger.warning(f"Error searching doc collection {collection_name}: {str(e)}")
            
            # Sort results by similarity
            doc_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Format documentation context
            doc_context = []
            for result in doc_results[:3]:
                context = {
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'similarity': result.get('similarity', 0)
                }
                doc_context.append(context)
                
            return {
                "status": "success",
                "results": doc_context
            }
            
        except Exception as e:
            logger.error(f"Error in documentation search: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    @tool("relationship_search")
    def search_relationships_old(self, query: str) -> Dict[str, Any]:
        """
        Search for table and code relationships.
        
        Args:
            query: The search query to find relevant relationships
            
        Returns:
            Dict containing relationship information
        """
        try:
            # Search code collections for relationship information
            collections = self.db_manager.client.list_collections()
            code_collections = [c.name for c in collections 
                              if any(ext in c.name.lower() 
                                    for ext in ['py', 'sql', 'yml', 'yaml'])]
            
            relationships = []
            for collection_name in code_collections:
                try:
                    results = self.db_manager.code_similarity_search(
                        collection_name=collection_name,
                        code_snippet=query,
                        n_results=2
                    )
                    if results and 'results' in results:
                        for result in results['results']:
                            if 'file_info' in result and 'relationships' in result['file_info']:
                                relationships.append(result['file_info']['relationships'])
                except Exception as e:
                    logger.warning(f"Error searching relationships in {collection_name}: {str(e)}")
            
            return {
                "status": "success",
                "results": relationships
            }
            
        except Exception as e:
            logger.error(f"Error in relationship search: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 