from pathlib import Path
from typing import List, Dict, Any
from src.utils import ChromaDBManager
import logging

logger = logging.getLogger(__name__)

class SearchProcessor:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    def search_documents(self, query: str, collection_name: str = None, n_results: int = 5) -> Dict[str, Any]:
        """Search documents using hybrid search."""
        try:
            # If no collection name is provided, try to find an appropriate one
            if not collection_name:
                try:
                    collections = self.chroma_manager.client.list_collections()
                    if collections:
                        # Use the first available collection
                        collection_name = collections[0].name
                    else:
                        raise ValueError("No collections found. Please upload documents first.")
                except Exception as e:
                    raise ValueError(f"Error accessing collections: {str(e)}")
            
            logger.info(f"Searching in collection: {collection_name}")
            results = self.chroma_manager.hybrid_search(
                collection_name,
                query,
                n_results
            )
            
            if not results.get('results'):
                logger.warning("No results found")
                return {
                    'query': query,
                    'results': []
                }
            
            # Post-process results
            processed_results = []
            for result in results['results']:
                try:
                    # Add relevance indicators
                    result['relevance_factors'] = self._analyze_relevance(
                        query,
                        result['content'],
                        result['similarity']
                    )
                    processed_results.append(result)
                    logger.info(f"Processed result with similarity {result['similarity']}")
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    continue
            
            results['results'] = processed_results
            return results
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}", exc_info=True)
            raise

    def search_code(self, 
                   code_query: str,
                   collection_name: str = None,
                   language: str = "python", 
                   n_results: int = 5) -> Dict[str, Any]:
        """Search code using similarity search.
        
        Args:
            code_query (str): The code snippet or query to search for
            collection_name (str, optional): Name of the collection to search in. Defaults to None.
            language (str, optional): Programming language. Defaults to "python".
            n_results (int, optional): Number of results to return. Defaults to 5.
        
        Returns:
            Dict[str, Any]: Search results with matches and metadata
        """
        try:
            # If no collection name is provided, try to find an appropriate one
            if not collection_name:
                try:
                    collections = self.chroma_manager.client.list_collections()
                    code_collections = [c.name for c in collections if any(ext in c.name.lower() for ext in ['py', 'sql', 'yml', 'yaml'])]
                    if code_collections:
                        collection_name = code_collections[0]
                    else:
                        raise ValueError("No code collections found. Please upload code files first.")
                except Exception as e:
                    raise ValueError(f"Error accessing collections: {str(e)}")
            
            logger.info(f"Searching code in collection: {collection_name}")
            results = self.chroma_manager.code_similarity_search(
                collection_name,
                code_query,
                n_results
            )
            
            # Add language information to results
            if results and 'results' in results:
                for result in results['results']:
                    result['language'] = language
            
            return results
            
        except Exception as e:
            logger.error(f"Error in code search: {str(e)}")
            raise

    def _analyze_relevance(self, query: str, content: str, similarity: float) -> Dict[str, Any]:
        """Analyze why a document is considered relevant."""
        try:
            # Ensure content is a string
            if isinstance(content, list):
                content = ' '.join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
                
            factors = {
                'semantic_similarity': similarity,
                'keyword_matches': self._count_keyword_matches(query, content),
                'context_summary': self._extract_relevant_context(query, content)
            }
            return factors
        except Exception as e:
            logger.error(f"Error in relevance analysis: {str(e)}")
            return {
                'semantic_similarity': similarity,
                'keyword_matches': {},
                'context_summary': "Error analyzing content"
            }

    def _analyze_code_match(self, query: str, code: str, similarity: float) -> Dict[str, Any]:
        """Analyze code match relevance."""
        return {
            'similarity_score': similarity,
            'matching_patterns': self._identify_code_patterns(query, code),
            'function_context': self._extract_function_context(code),
            'complexity': self._estimate_code_complexity(code)
        }

    def _count_keyword_matches(self, query: str, content: str) -> Dict[str, int]:
        """Count keyword matches in content."""
        try:
            # Ensure content is a string
            if isinstance(content, list):
                content = ' '.join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Process query keywords
            keywords = query.lower().split()
            content_lower = content.lower()
            
            return {
                keyword: content_lower.count(keyword)
                for keyword in keywords
            }
        except Exception as e:
            logger.error(f"Error in keyword matching: {str(e)}")
            return {}

    def _extract_relevant_context(self, query: str, content: str, context_size: int = 100) -> str:
        """Extract relevant context around query matches."""
        try:
            # Ensure content is a string
            if isinstance(content, list):
                content = ' '.join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
                
            # Find the first occurrence of any query term
            query_terms = query.lower().split()
            content_lower = content.lower()
            
            # Find the best matching position
            best_pos = -1
            for term in query_terms:
                pos = content_lower.find(term)
                if pos != -1 and (best_pos == -1 or pos < best_pos):
                    best_pos = pos
            
            if best_pos == -1:
                return content[:context_size] + "..."
            
            # Extract context around the match
            start = max(0, best_pos - context_size // 2)
            end = min(len(content), best_pos + context_size // 2)
            
            context = content[start:end]
            if start > 0:
                context = "..." + context
            if end < len(content):
                context = context + "..."
                
            return context
            
        except Exception as e:
            logger.error(f"Error extracting context: {str(e)}")
            return content[:context_size] + "..."

    def _identify_code_patterns(self, query: str, code: str) -> List[str]:
        """Identify relevant code patterns."""
        # Implementation details...
        return []

    def _extract_function_context(self, code: str) -> Dict[str, Any]:
        """Extract context of functions in code."""
        # Implementation details...
        return {}

    def _estimate_code_complexity(self, code: str) -> Dict[str, Any]:
        """Estimate code complexity metrics."""
        # Implementation details...
        return {}

    def process_code_search(self, code_snippet: str, collection_name: str = None) -> Dict[str, Any]:
        """Process code search with enhanced context."""
        try:
            # Get search results
            results = self.chroma_manager.search_code(
                collection_name,
                code_snippet
            )
            
            if not results.get('results'):
                return {
                    'query': code_snippet,
                    'results': []
                }
            
            # Use cached analysis from metadata
            processed_results = []
            for result in results['results']:
                try:
                    # Analysis is already in the metadata
                    analysis = result.get('metadata', {}).get('analysis', {})
                    
                    processed_result = {
                        'content': result['content'],
                        'similarity': result['similarity'],
                        'file_path': result['metadata'].get('file_path'),
                        'language': result['metadata'].get('language'),
                        'context': self._extract_code_context(
                            result['content'],
                            code_snippet,
                            analysis
                        )
                    }
                    processed_results.append(processed_result)
                    logger.info(f"Processed result with similarity {result['similarity']}")
                    
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    continue
            
            return {
                'query': code_snippet,
                'results': processed_results
            }
            
        except Exception as e:
            logger.error(f"Error in code search: {str(e)}")
            raise
