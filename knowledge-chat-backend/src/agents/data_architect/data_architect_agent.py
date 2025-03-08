import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from src.db.database import ChatDatabase
from src.agents.data_architect.schema_search_agent import SchemaSearchAgent
from src.agents.github_search.code_search_agent import GitHubCodeSearchAgent
from src.agents.data_architect.question_parser import QuestionParserSystem

# Set up logger
logger = logging.getLogger(__name__)

class DataArchitectAgent:
    """
    Data Architect Agent that combines outputs from question parser, 
    schema search, and code search to provide comprehensive answers.
    """
    
    def __init__(self):
        """Initialize the Data Architect Agent"""
        self.logger = logging.getLogger(__name__)
        self.db = ChatDatabase()
        
        # Initialize LLM
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.2,  # Slightly higher temperature for more creative responses
            base_url="http://localhost:11434",
            timeout=180,  # Longer timeout for complex synthesis
        )
        
        # Initialize sub-agents
        self.schema_search_agent = SchemaSearchAgent()
        self.code_search_agent = GitHubCodeSearchAgent()
    
    async def process_question(self, 
                              question: str, 
                              thread_id: str, 
                              conversation_id: str,
                              parsed_question: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a question through the entire pipeline:
        1. Parse the question (if not already parsed)
        2. Search for relevant schemas
        3. Search for relevant code
        4. Synthesize the results
        
        Args:
            question: The user's question
            thread_id: The thread ID
            conversation_id: The conversation ID
            parsed_question: Optional pre-parsed question
            
        Returns:
            Dict containing the final response
        """
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Step 1: Parse the question if not already parsed
            if not parsed_question:
                question_parser = QuestionParserSystem()
                parsed_question = await question_parser.parse_question(
                    question=question,
                    thread_id=thread_id,
                    conversation_id=conversation_id
                )
            
            # Step 2: Search for relevant schemas
            schema_results = self.schema_search_agent.search_schemas(parsed_question)
            
            # Step 3: Search for relevant code
            code_results = self.code_search_agent.search_code(parsed_question)
            
            # Step 4: Synthesize the results
            final_response = self._synthesize_results(
                parsed_question=parsed_question,
                schema_results=schema_results,
                code_results=code_results
            )
            
            # Save the results
            self._save_results(
                thread_id=thread_id,
                conversation_id=conversation_id,
                parsed_question=parsed_question,
                schema_results=schema_results,
                code_results=code_results,
                final_response=final_response
            )
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing question: {str(e)}",
                "parsed_question": parsed_question or {},
                "schema_results": [],
                "code_results": [],
                "final_response": "I encountered an error while analyzing your question. Please try again."
            }
    
    def _synthesize_results(self, 
                           parsed_question: Dict[str, Any],
                           schema_results: List[Dict[str, Any]],
                           code_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize the results from all agents into a comprehensive response
        
        Args:
            parsed_question: The parsed question
            schema_results: Results from schema search
            code_results: Results from code search
            
        Returns:
            Dict containing the synthesized response
        """
        try:
            # Create prompt for LLM
            prompt_template = """
            You are an expert Data Architect helping a user understand how to implement a data solution.
            
            BUSINESS QUESTION:
            {rephrased_question}
            
            KEY POINTS:
            {key_points}
            
            BUSINESS CONTEXT:
            {business_context}
            
            RELEVANT DATABASE SCHEMAS:
            {schema_results}
            
            RELEVANT CODE EXAMPLES:
            {code_results}
            
            TASK:
            Provide a comprehensive data architecture solution that addresses the user's question. Your response should include:
            
            1. A clear explanation of the business problem and requirements
            2. Recommended database schema design with tables and relationships
            3. Implementation approach with code patterns
            4. Best practices and considerations
            5. A sample implementation plan
            
            FORMAT YOUR RESPONSE IN THESE SECTIONS:
            
            ## Business Understanding
            [Provide a clear explanation of the business problem and requirements]
            
            ## Data Architecture Recommendation
            [Recommend database schema design with tables and relationships]
            
            ## Implementation Approach
            [Suggest implementation patterns and code examples]
            
            ## Best Practices
            [Provide best practices and considerations]
            
            ## Implementation Plan
            [Outline a sample implementation plan]
            
            Make your response practical, actionable, and focused on the user's specific needs.
            """
            
            # Format schema results for the prompt
            schema_results_text = ""
            for i, result in enumerate(schema_results):
                schema_name = result.get("schema_name", "unknown_schema")
                table_name = result.get("table_name", "unknown_table")
                columns = result.get("columns", [])
                description = result.get("description", "")
                explanation = result.get("explanation", "")
                relevance_score = result.get("relevance_score", 0)
                
                columns_text = ", ".join(columns) if isinstance(columns, list) else str(columns)
                
                schema_text = f"SCHEMA {i+1} (Relevance: {relevance_score}/10):\nSchema: {schema_name}\nTable: {table_name}\nColumns: {columns_text}\nDescription: {description}\nRelevance: {explanation}\n"
                schema_results_text += schema_text + "\n"
            
            # Format code results for the prompt
            code_results_text = ""
            for i, result in enumerate(code_results):
                file_path = result.get("file_path", "unknown_file")
                code_snippet = result.get("code_snippet", "")
                explanation = result.get("explanation", "")
                relevance_score = result.get("relevance_score", 0)
                repo_info = result.get("repo_info", {})
                
                code_text = f"CODE EXAMPLE {i+1} (Relevance: {relevance_score}/10):\nFile: {file_path}\nRepo: {repo_info.get('repo_name', 'unknown')}\nRelevance: {explanation}\n\n```\n{code_snippet}\n```\n"
                code_results_text += code_text + "\n"
            
            # Format the prompt
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["rephrased_question", "key_points", "business_context", "schema_results", "code_results"]
            )
            
            formatted_prompt = prompt.format(
                rephrased_question=parsed_question.get("rephrased_question", ""),
                key_points="\n".join([f"- {point}" for point in parsed_question.get("key_points", [])]),
                business_context=json.dumps(parsed_question.get("business_context", {}), indent=2),
                schema_results=schema_results_text,
                code_results=code_results_text
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Create the final response
            final_response = {
                "status": "success",
                "parsed_question": parsed_question,
                "schema_results": schema_results,
                "code_results": code_results,
                "response": response_text,
                "sections": self._extract_sections(response_text)
            }
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error synthesizing results: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error synthesizing results: {str(e)}",
                "response": "I encountered an error while synthesizing the results. Please try again."
            }
    
    def _extract_sections(self, response_text: str) -> Dict[str, str]:
        """Extract sections from the response text"""
        sections = {}
        
        # Define section headers
        section_headers = [
            "Business Understanding",
            "Data Architecture Recommendation",
            "Implementation Approach",
            "Best Practices",
            "Implementation Plan"
        ]
        
        # Extract each section
        for i, header in enumerate(section_headers):
            start_marker = f"## {header}"
            end_marker = f"## {section_headers[i+1]}" if i < len(section_headers) - 1 else None
            
            start_idx = response_text.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = response_text.find(end_marker) if end_marker and end_marker in response_text else len(response_text)
                
                section_content = response_text[start_idx:end_idx].strip()
                sections[header] = section_content
        
        # If no sections were found using the ## format, try to extract them differently
        if not sections:
            # Try to find sections with just the header text
            for header in section_headers:
                start_idx = response_text.find(header)
                if start_idx != -1:
                    # Find the next header or end of text
                    next_header_idx = len(response_text)
                    for next_header in section_headers:
                        if next_header != header:
                            temp_idx = response_text.find(next_header, start_idx + len(header))
                            if temp_idx != -1 and temp_idx < next_header_idx:
                                next_header_idx = temp_idx
                    
                    # Extract content
                    start_idx += len(header)
                    section_content = response_text[start_idx:next_header_idx].strip()
                    sections[header] = section_content
        
        return sections
    
    def _save_results(self,
                     thread_id: str,
                     conversation_id: str,
                     parsed_question: Dict[str, Any],
                     schema_results: List[Dict[str, Any]],
                     code_results: List[Dict[str, Any]],
                     final_response: Dict[str, Any]):
        """Save all results to the thread directory"""
        try:
            # Get thread directory
            thread_dir = Path(self.db.db_path).parent / "threads" / thread_id
            thread_dir.mkdir(exist_ok=True)
            
            # Create a results file
            results_file = thread_dir / f"architect_results_{conversation_id}.json"
            
            # Prepare data for JSON storage
            json_data = {
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "parsed_question": parsed_question,
                "schema_results": schema_results,
                "code_results": code_results,
                "final_response": final_response
            }
            
            # Write to file
            with open(results_file, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            self.logger.info(f"Saved architect results to {results_file}")
            
            # Update conversation in database
            conversation_data = {
                "query": parsed_question.get("original_question", ""),
                "output": final_response.get("response", ""),
                "technical_details": json.dumps({
                    "parsed_question": parsed_question,
                    "schema_results": schema_results,
                    "code_results": code_results,
                    "sections": final_response.get("sections", {})
                }),
                "code_context": json.dumps({
                    "schema_results": schema_results,
                    "code_results": code_results
                }),
                "thread_id": thread_id,
                "feedback_status": "completed"
            }
            
            self.db.save_conversation(conversation_id, conversation_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving architect results: {e}", exc_info=True)
            return False

    def generate_response(self, parsed_question: Dict[str, Any], schema_results: List[Dict[str, Any]], code_results: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """Generate a comprehensive response based on schema and code search results"""
        try:
            self.logger.info(f"Generating data architect response for question: {original_question[:50]}...")
            
            # Create prompt for LLM
            prompt_template = """
            You are an expert data architect helping to answer a business question by providing a comprehensive solution.
            
            ORIGINAL QUESTION:
            {original_question}
            
            BUSINESS UNDERSTANDING:
            {rephrased_question}
            
            KEY POINTS:
            {key_points}
            
            BUSINESS CONTEXT:
            {business_context}
            
            RELEVANT DATABASE SCHEMAS:
            {schema_results}
            
            RELEVANT CODE EXAMPLES:
            {code_results}
            
            Please provide a comprehensive solution that addresses the business question. Your response should include:
            
            1. Business Understanding: A clear explanation of the business problem and objectives
            2. Data Architecture Recommendation: The recommended data architecture approach
            3. Schema Design: Proposed schema design or modifications to existing schemas
            4. Implementation Approach: How to implement the solution, including code examples
            5. Testing and Validation: How to test and validate the solution
            
            Format your response with clear section headers (## Business Understanding, etc.) and provide specific, actionable recommendations.
            """
            
            # Format schema results for the prompt
            schema_results_text = ""
            for i, result in enumerate(schema_results):
                schema_name = result.get("schema_name", "unknown_schema")
                table_name = result.get("table_name", "unknown_table")
                columns = result.get("columns", [])
                relevance = result.get("relevance_score", 0)
                explanation = result.get("explanation", "")
                query_pattern = result.get("query_pattern", "")
                
                # Format columns as a list
                if isinstance(columns, str):
                    columns = [col.strip() for col in columns.split(",")]
                
                columns_text = ", ".join(columns) if columns else "No columns available"
                
                schema_results_text += f"""
                SCHEMA {i+1}: {schema_name}.{table_name}
                Relevance: {relevance * 10:.1f}/10
                Columns: {columns_text}
                Explanation: {explanation}
                Query Pattern: {query_pattern}
                """
            
            # Format code results for the prompt
            code_results_text = ""
            for i, result in enumerate(code_results):
                file_path = result.get("file_path", "unknown_file")
                relevance = result.get("relevance_score", 0)
                explanation = result.get("explanation", "")
                code_snippet = result.get("code_snippet", "")
                
                # Truncate long code snippets
                if len(code_snippet) > 500:
                    code_snippet = code_snippet[:500] + "...[truncated]"
                
                code_results_text += f"""
                CODE EXAMPLE {i+1}: {file_path}
                Relevance: {relevance * 10:.1f}/10
                Explanation: {explanation}
                
                ```
                {code_snippet}
                ```
                """
            
            # Format the prompt
            prompt = PromptTemplate.from_template(prompt_template).format(
                original_question=original_question,
                rephrased_question=parsed_question.get("rephrased_question", ""),
                key_points="\n".join([f"- {point}" for point in parsed_question.get("key_points", [])]),
                business_context=json.dumps(parsed_question.get("business_context", {}), indent=2),
                schema_results=schema_results_text,
                code_results=code_results_text
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract sections from the response
            sections = self._extract_sections(response_text)
            
            # Create the final response
            result = {
                "response": response_text,
                "sections": sections,
                "schema_results": schema_results,
                "code_results": code_results
            }
            
            # Log success
            self.logger.info(f"Generated data architect response with {len(sections)} sections")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating data architect response: {e}", exc_info=True)
            # Return a basic response in case of error
            return {
                "response": f"I apologize, but I encountered an error while generating a response. Please try again or rephrase your question.\n\nError details: {str(e)}",
                "sections": {},
                "schema_results": schema_results,
                "code_results": code_results
            } 