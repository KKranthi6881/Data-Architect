from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
import logging
from src.utils import ChromaDBManager
from langchain.tools import Tool
from pydantic import BaseModel, Field
import json
import re

logger = logging.getLogger(__name__)

class SearchTools:
    def __init__(self, db_manager: ChromaDBManager):
        """Initialize search tools with a database manager."""
        self.db_manager = db_manager

    def search_code(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Search for code snippets based on a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Get the code collection
            collection = self.db_manager.get_or_create_collection("code_documents")
            
            # Search for documents
            results = collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            return {
                "status": "success",
                "results": self._format_results(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def search_github_repos(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Search for GitHub repository content based on a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Get the GitHub documents collection
            collection = self.db_manager.get_or_create_collection("github_documents")
            
            # Use hybrid search for better results with code repositories
            results = self.db_manager.hybrid_search(
                collection_name="github_documents",
                query=query,
                n_results=limit
            )
            
            if not results or not results.get('results', []):
                # Fallback to regular search if hybrid search returns no results
                results = collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                if not results or not results['documents'] or len(results['documents'][0]) == 0:
                    return {
                        "status": "success",
                        "results": [],
                        "message": "No GitHub repository content found matching your query."
                    }
                
                # Format regular search results
                return {
                    "status": "success",
                    "results": self._format_github_results(results)
                }
            
            # Return hybrid search results
            return {
                "status": "success",
                "results": results.get('results', [])
            }
            
        except Exception as e:
            logger.error(f"Error searching GitHub content: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _format_github_results(self, results) -> List[Dict[str, Any]]:
        """Format GitHub search results with enhanced metadata."""
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return []
        
        formatted_results = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Extract repository information
            repo_url = metadata.get('repo_url', '')
            repo_name = repo_url.split('/')[-1] if repo_url else 'Unknown'
            file_path = metadata.get('file_path', '')
            language = metadata.get('file_type', '')
            
            # Format the result with proper metadata structure
            formatted_result = {
                "id": i + 1,
                "content": doc,
                "metadata": {
                    "file_path": file_path,
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "file_type": language,
                    "source": "github"
                }
            }
            
            # Add score if available
            if 'distances' in results:
                formatted_result["score"] = results['distances'][0][i]
            
            # Extract dbt-specific information if available
            if language == 'dbt' or metadata.get('is_dbt', False):
                formatted_result["dbt_info"] = {
                    'model_name': metadata.get('model_name', ''),
                    'materialization': metadata.get('materialization', ''),
                    'description': metadata.get('description', ''),
                    'references': metadata.get('references', '').split(',') if metadata.get('references', '') else [],
                    'sources': metadata.get('sources', '').split(',') if metadata.get('sources', '') else []
                }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def search_sql_schema(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Search for SQL schema information based on a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Get the SQL schema collection
            collection = self.db_manager.get_or_create_collection("sql_documents")
            
            # Search for documents using hybrid search for better results
            results = self.db_manager.hybrid_search(
                collection_name="sql_documents",
                query=query,
                n_results=limit
            )
            
            if not results or not results.get('results', []):
                # Try fallback to regular search if hybrid search returns no results
                results = collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                if not results or not results['documents'] or len(results['documents'][0]) == 0:
                    return {
                        "status": "success",
                        "results": [],
                        "message": "No SQL schema information found matching your query."
                    }
                
                # Format regular search results
                return {
                    "status": "success",
                    "results": self._format_sql_results(results)
                }
            
            # Return hybrid search results
            return {
                "status": "success",
                "results": results.get('results', [])
            }
            
        except Exception as e:
            logger.error(f"Error searching SQL schema: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _format_sql_results(self, results) -> List[Dict[str, Any]]:
        """Format SQL search results with enhanced metadata."""
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return []
        
        formatted_results = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Extract table information
            tables = metadata.get('tables', '').split(',') if metadata.get('tables', '') else []
            
            # Extract relationship information
            relationships = []
            if 'relationships' in metadata:
                rel_str = metadata.get('relationships', '')
                if rel_str:
                    # Parse relationship string format: "table1.col1 -> table2.col2|..."
                    for rel in rel_str.split('|'):
                        if '->' in rel:
                            relationships.append(rel.strip())
            
            # Format the result
            formatted_result = {
                "id": i + 1,
                "content": doc,
                "metadata": {
                    "source": metadata.get('source', 'Unknown source'),
                    "language": metadata.get('language', 'sql'),
                    "tables": tables,
                    "relationships": relationships,
                    "analysis_summary": metadata.get('analysis_summary', '')
                },
                "score": results['distances'][0][i] if 'distances' in results else None
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Format search results into a standardized structure."""
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return []
        
        formatted_results = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted_result = {
                "id": i + 1,
                "content": doc,
                "metadata": metadata,
                "score": results['distances'][0][i] if 'distances' in results else None
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results

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
                "status": "success",
                "results": doc_results.get("results", [])
            }
        except Exception as e:
            logger.error(f"Error in documentation search: {str(e)}")
            return {
                "status": "error", 
                "error": str(e), 
                "results": []
            }

    def search_dbt_models(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Search specifically for dbt models and their lineage
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with dbt model search results
        """
        try:
            # Get the GitHub documents collection
            collection = self.db_manager.get_or_create_collection("github_documents")
            
            # Create a more specific query for dbt models
            dbt_query = f"dbt model {query}"
            
            # Search for documents
            results = self.db_manager.hybrid_search(
                collection_name="github_documents",
                query=dbt_query,
                n_results=limit
            )
            
            if not results or not results.get('results', []):
                return {
                    "status": "success",
                    "results": [],
                    "message": "No dbt models found matching your query."
                }
            
            # Filter results to only include dbt models
            dbt_results = []
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('language') == 'dbt' or metadata.get('is_dbt', False):
                    dbt_results.append(result)
            
            # If we filtered out all results, try a more general search
            if not dbt_results:
                return {
                    "status": "success",
                    "results": [],
                    "message": "No dbt models found matching your query."
                }
            
            # Enhance results with lineage information
            enhanced_results = self._enhance_dbt_results_with_lineage(dbt_results)
            
            return {
                "status": "success",
                "results": enhanced_results
            }
            
        except Exception as e:
            logger.error(f"Error searching dbt models: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _enhance_dbt_results_with_lineage(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance dbt results with lineage information by finding related models.
        
        Args:
            results: List of dbt model search results
            
        Returns:
            Enhanced results with lineage information
        """
        enhanced_results = []
        
        # Extract model names from results
        model_names = set()
        for result in results:
            metadata = result.get('metadata', {})
            model_name = metadata.get('model_name', '')
            if model_name:
                model_names.add(model_name)
            
            # Also add referenced models
            references = metadata.get('references', '').split(',') if metadata.get('references', '') else []
            for ref in references:
                if ref.strip():
                    model_names.add(ref.strip())
        
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
            
            # Create enhanced result
            enhanced_result = result.copy()
            
            # Add lineage information
            if 'dbt_info' not in enhanced_result:
                enhanced_result['dbt_info'] = {}
            
            enhanced_result['dbt_info']['lineage'] = {
                'upstream': upstream_models,
                'downstream': downstream_models
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results

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
            
            # Also search for dbt models to get lineage information
            dbt_results = self.search_dbt_models(query, limit=2)
            
            relationships = []
            
            # Extract relationships from SQL results
            if sql_results and 'results' in sql_results:
                for result in sql_results['results']:
                    if 'file_info' in result and 'relationships' in result['file_info']:
                        relationships.append({
                            'source': 'sql',
                            'relationships': result['file_info']['relationships']
                        })
            
            # Extract relationships from dbt results
            if dbt_results and 'results' in dbt_results:
                for result in dbt_results['results']:
                    if 'dbt_info' in result and result['dbt_info'] and 'lineage' in result['dbt_info']:
                        relationships.append({
                            'source': 'dbt',
                            'model': result.get('dbt_info', {}).get('model_name', 'Unknown model'),
                            'lineage': result['dbt_info']['lineage']
                        })
            
            return {
                "status": "success",
                "results": relationships
            }
        except Exception as e:
            logger.error(f"Error in relationship search: {str(e)}")
            return {
                "status": "error", 
                "error": str(e), 
                "results": []
            }

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
                name="search_github_repos",
                func=self.search_github_repos,
                description="Search through GitHub repositories, especially dbt code"
            ),
            Tool(
                name="search_sql_schema",
                func=self.search_sql_schema,
                description="Search for SQL schema information and table relationships"
            ),
            Tool(
                name="search_documentation",
                func=self.search_documentation,
                description="Search through documentation files"
            ),
            Tool(
                name="search_dbt_models",
                func=self.search_dbt_models,
                description="Search specifically for dbt models and their lineage"
            ),
            Tool(
                name="search_relationships",
                func=self.search_relationships,
                description="Search for relationships between code and documentation"
            )
        ]

    # Legacy tools kept for backward compatibility
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

    def _log_search_results(self, source: str, results: Dict[str, Any]) -> None:
        """Log detailed search results for debugging."""
        try:
            status = results.get('status', 'unknown')
            result_list = results.get('results', [])
            
            logger.info(f"{source} search status: {status}, found {len(result_list)} results")
            
            # Log a preview of each result
            for i, result in enumerate(result_list[:3], 1):  # Log first 3 results
                logger.info(f"{source} result {i}:")
                
                # Get metadata from the result
                metadata = result.get('metadata', {})
                
                # Log different fields based on result type
                if source == "GitHub":
                    file_path = metadata.get('file_path', 'Unknown')
                    repo_url = metadata.get('repo_url', 'Unknown')
                    repo_name = repo_url.split('/')[-1] if repo_url != 'Unknown' else 'Unknown'
                    language = metadata.get('file_type', 'Unknown')
                    
                    logger.info(f"  File: {file_path}")
                    logger.info(f"  Repo: {repo_name}")
                    logger.info(f"  Language: {language}")
                    
                    # Log DBT info if available
                    dbt_info = result.get('dbt_info', {})
                    if dbt_info:
                        logger.info(f"  DBT Model: {dbt_info.get('model_name', 'Unknown')}")
                        logger.info(f"  Materialization: {dbt_info.get('materialization', 'Unknown')}")
                
                elif source == "SQL":
                    logger.info(f"  Source: {metadata.get('source', 'Unknown')}")
                    logger.info(f"  Tables: {', '.join(metadata.get('tables', []))[:100]}")
                
                elif source == "DBT":
                    dbt_info = result.get('dbt_info', {})
                    if dbt_info:
                        logger.info(f"  Model: {dbt_info.get('model_name', 'Unknown')}")
                        logger.info(f"  References: {', '.join(dbt_info.get('references', []))[:100]}")
                
                elif source == "Relationships":
                    logger.info(f"  Source: {result.get('source', 'Unknown')}")
                    logger.info(f"  Model: {result.get('model', 'Unknown')}")
                
                # Log content preview for all result types
                content = result.get('content', '')
                if content:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"  Content preview: {content_preview}")
                
                logger.info("  ---")
                
        except Exception as e:
            logger.error(f"Error logging {source} search results: {str(e)}") 