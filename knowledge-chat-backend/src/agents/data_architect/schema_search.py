import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from src.db.database import ChatDatabase
from src.tools import SearchTools
from src.utils import ChromaDBManager
from src.agents.data_architect.question_parser import QuestionParserSystem

# Set up logger
logger = logging.getLogger(__name__)

class SchemaSearchAgent:
    """Agent for searching SQL schemas based on parsed questions"""
    
    def __init__(self):
        """Initialize the schema search agent"""
        self.logger = logging.getLogger(__name__)
        self.db = ChatDatabase()
        
        # Create ChromaDBManager and SearchTools
        self.chroma_manager = ChromaDBManager()
        self.search_tools = SearchTools(self.chroma_manager)
        
        # Initialize LLM
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.1,
            base_url="http://localhost:11434",
            timeout=120,
        )
    
    def search_schemas(self, parsed_question: Dict[str, Any], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant SQL schemas based on the parsed question
        
        Args:
            parsed_question: The output from the question parser
            max_results: Maximum number of results to return
            
        Returns:
            List of schema search results
        """
        try:
            # Create search query from parsed question
            search_query = self._create_search_query(parsed_question)
            self.logger.info(f"Created schema search query: {search_query}")
            
            # Use the search_tools to search for SQL schemas
            search_results = self.search_tools.search_sql_schema(search_query, limit=max_results)
            
            # Check if we got valid results
            if search_results.get("status") != "success" or not search_results.get("results"):
                self.logger.warning(f"No schema search results found or error in search")
                return []
            
            # Format the results for enhancement
            formatted_results = search_results.get("results", [])
            
            # Enhance results with LLM
            enhanced_results = self._enhance_search_results(formatted_results, parsed_question)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in schema search: {e}", exc_info=True)
            return []
    
    def _create_search_query(self, parsed_question: Dict[str, Any]) -> str:
        """Create a search query from the parsed question"""
        try:
            # Extract key information from parsed question
            rephrased_question = parsed_question.get("rephrased_question", "")
            key_points = parsed_question.get("key_points", [])
            business_context = parsed_question.get("business_context", {})
            
            # Build search query
            query_parts = [rephrased_question]
            
            # Add key points (limited to first 3)
            if key_points:
                query_parts.extend(key_points[:3])
            
            # Add key entities from business context
            if business_context and "key_entities" in business_context:
                entities = business_context.get("key_entities", [])
                if entities and isinstance(entities, list):
                    query_parts.append(" ".join(entities[:5]))
            
            # Join all parts
            search_query = " ".join(query_parts)
            
            return search_query
            
        except Exception as e:
            self.logger.error(f"Error creating schema search query: {e}")
            return parsed_question.get("rephrased_question", "")
    
    def _enhance_search_results(self, search_results: List[Dict[str, Any]], parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM analysis"""
        try:
            if not search_results:
                return []
            
            # Create prompt for LLM - using raw string to avoid formatting issues
            prompt = f"""
            # Data Architect Schema Analysis

            You are an expert data architect analyzing database schemas to answer a business question. Think step-by-step like a professional data architect to find the most relevant tables and columns.

            ## Business Question
            "{parsed_question.get('rephrased_question', '')}"

            ## Key Points from Question
            {chr(10).join([f"- {point}" for point in parsed_question.get('key_points', [])])}

            ## Business Context
            {json.dumps(parsed_question.get('business_context', {}), indent=2)}

            ## Available Database Schemas
            """
            
            # Format schema snippets and add them to the prompt
            for i, result in enumerate(search_results):
                # Extract schema information
                schema_name = result.get("schema_name", "unknown_schema")
                table_name = result.get("table_name", "unknown_table")
                columns = result.get("columns", [])
                description = result.get("description", "")
                
                # Format columns as a list
                if isinstance(columns, str):
                    columns = [col.strip() for col in columns.split(",")]
                
                # Create schema snippet text and add directly to prompt
                columns_text = ", ".join(columns) if columns else "No columns available"
                prompt += f"""
            ### Schema {i+1}:
            - **Schema**: {schema_name}
            - **Table**: {table_name}
            - **Columns**: {columns_text}
            - **Description**: {description}
                """
            
            # Add the rest of the prompt
            prompt += """
            ## Analysis Instructions
            Think through this problem step-by-step:

            1. **Understand the Business Need**:
               - What specific data points does the question require?
               - What business metrics or KPIs are being requested?
               - What time periods, filters, or groupings are needed?

            2. **Identify Key Entities**:
               - What business entities (example:customers, products, orders, etc.) are central to this question?
               - What relationships between entities need to be explored?

            3. **Evaluate Each Schema**:
               - For each schema, determine if it contains the required entities
               - Check if the columns provide the metrics or attributes needed
               - Consider if the table would be a primary or supporting table for the query

            4. **Develop Query Patterns**:
               - What SQL patterns would effectively extract the needed information?
               - What joins might be required with other tables?
               - What aggregations or calculations would be needed?

            ## Example Thought Process
            "The question asks about customer purchase patterns by region. I need:
            1. Customer data (demographics, location)
            2. Purchase/order data (amounts, dates)
            3. Product information
            4. Regional classifications

            The customers table has location data but lacks purchase history. The orders table has purchase amounts and dates, with a customer_id that could join to the customers table. The products table would help categorize purchases..."

            ## Analysis Output
            For each schema, provide your analysis in the following JSON format:

            ```json
            [
              {
                "schema_name": "example_schema",
                "table_name": "example_table",
                "columns": ["column1", "column2", "column3"],
                "relevance_score": 8.5,
                "description": "This schema contains...",
                "explanation": "This is relevant because it contains the customer purchase data needed to analyze regional patterns. The customer_id column can be joined with the customers table to get demographic information, while the purchase_date and amount columns provide the core metrics needed.",
                "query_pattern": "SELECT c.region, SUM(o.amount) FROM example_schema.orders o JOIN example_schema.customers c ON o.customer_id = c.id WHERE o.purchase_date BETWEEN '2023-01-01' AND '2023-12-31' GROUP BY c.region ORDER BY SUM(o.amount) DESC"
              }
            ]
            ```

            IMPORTANT: 
            - Your response must be valid JSON that can be parsed
            - Score relevance from 0-10 based on how directly the schema addresses the question
            - Provide detailed explanations that show your reasoning
            - Include realistic SQL query patterns that could actually be executed
            - Do not include any text outside the JSON block
            """
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_text = self._extract_json(response_text)
            
            if not json_text:
                self.logger.warning("Could not extract JSON from LLM response")
                return self._format_raw_results(search_results)
            
            try:
                enhanced_results = json.loads(json_text)
                # Validate and fix the enhanced results
                return self._validate_enhanced_results(enhanced_results, search_results)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error enhancing schema search results: {e}")
                return self._format_raw_results(search_results)
                
        except Exception as e:
            self.logger.error(f"Error enhancing schema search results: {e}", exc_info=True)
            # Return basic results if enhancement fails
            return self._format_raw_results(search_results)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response text"""
        # Try to extract JSON from markdown code block
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        # Try to extract JSON array directly
        array_pattern = r"\[\s*{[\s\S]*}\s*\]"
        matches = re.findall(array_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        # If all else fails, try to extract anything that looks like JSON
        if text.strip().startswith("[") and text.strip().endswith("]"):
            return text.strip()
        
        return ""
    
    def _validate_enhanced_results(self, enhanced_results: List[Dict[str, Any]], original_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix enhanced results"""
        # Ensure we have a list
        if not isinstance(enhanced_results, list):
            self.logger.warning("Enhanced results is not a list, using raw results")
            return self._format_raw_results(original_results)
        
        # Ensure all results have the required fields
        for i, result in enumerate(enhanced_results):
            if "relevance_score" not in result:
                result["relevance_score"] = 0.7
            elif isinstance(result["relevance_score"], int) or isinstance(result["relevance_score"], float):
                # Normalize score to 0-1 range if it's on a 0-10 scale
                if result["relevance_score"] > 1:
                    result["relevance_score"] = result["relevance_score"] / 10.0
            else:
                result["relevance_score"] = 0.7
                
            if "explanation" not in result:
                result["explanation"] = "Automatically extracted from database schema"
                
            if "schema_name" not in result and i < len(original_results):
                result["schema_name"] = original_results[i].get("schema_name", "unknown_schema")
                
            if "table_name" not in result and i < len(original_results):
                result["table_name"] = original_results[i].get("table_name", "unknown_table")
                
            if "columns" not in result and i < len(original_results):
                result["columns"] = original_results[i].get("columns", [])
                
            if "description" not in result and i < len(original_results):
                result["description"] = original_results[i].get("description", "")
                
            if "query_pattern" not in result:
                result["query_pattern"] = f"SELECT * FROM {result.get('schema_name', 'schema')}.{result.get('table_name', 'table')} LIMIT 10"
        
        return enhanced_results
    
    def _format_raw_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw search results when enhancement fails"""
        return [
            {
                "schema_name": result.get("schema_name", "unknown_schema"),
                "table_name": result.get("table_name", "unknown_table"),
                "columns": result.get("columns", []),
                "relevance_score": 0.7,  # Default relevance score
                "description": result.get("description", ""),
                "explanation": "Automatically extracted from database schema",
                "query_pattern": f"SELECT * FROM {result.get('schema_name', 'schema')}.{result.get('table_name', 'table')} LIMIT 10"
            }
            for result in search_results
        ]
    
    def save_search_results(self, thread_id: str, conversation_id: str, parsed_question: Dict[str, Any], search_results: List[Dict[str, Any]]):
        """Save search results to the thread directory"""
        try:
            # Get thread directory
            thread_dir = Path(self.db.db_path).parent / "threads" / thread_id
            thread_dir.mkdir(exist_ok=True)
            
            # Create a search results file
            search_results_file = thread_dir / f"schema_results_{conversation_id}.json"
            
            # Prepare data for JSON storage
            json_data = {
                "id": str(uuid.uuid4()),
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "parsed_question": parsed_question,
                "search_results": search_results
            }
            
            # Write to file
            with open(search_results_file, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            self.logger.info(f"Saved schema search results to {search_results_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving schema search results: {e}", exc_info=True)
            return False 