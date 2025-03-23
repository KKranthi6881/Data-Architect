from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Union, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import uuid
import logging
from sqlite3 import connect
from threading import Lock
import time
import traceback
from datetime import datetime
import json
import pprint
from src.tools import SearchTools
from src.db.database import ChatDatabase
from src.utils import ChromaDBManager
from src.dbt_tools import DbtSearchTools

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    question_analysis: Annotated[Dict, "Parsed question analysis"]
    dbt_results: Annotated[Dict, "DBT model search results"]
    final_response: Annotated[str, "Final formatted response to the user"]

# Define structured outputs for question analysis
class QuestionAnalysis(BaseModel):
    question_type: str = Field(
        description="Type of question (MODEL_INFO, LINEAGE, DEPENDENCIES, CODE_ENHANCEMENT, DOCUMENTATION, DEVELOPMENT, GENERAL)"
    )
    entities: List[str] = Field(
        description="Key entities mentioned in the question (models, etc.)"
    )
    intent: str = Field(
        description="Primary intent of the question"
    )
    rephrased_question: str = Field(
        description="Rephrased question for better search"
    )

class DataArchitectAgent:
    """
    Data Architect Agent that processes questions about data architecture,
    searches relevant sources, and provides comprehensive answers.
    """
    
    def __init__(self, repo_url: str = "", username: str = "", token: str = ""):
        """Initialize the data architect agent."""
        self.llm = ChatOllama(model="gemma3:latest")
        self.repo_url = repo_url
        self.username = username
        self.token = token
        
        # Initialize DBT tools only if repo_url is provided and not empty
        self.dbt_tools = None
        if repo_url and isinstance(repo_url, str) and repo_url.strip():
            try:
                # Validate GitHub URL format
                if not repo_url.startswith(('https://github.com/', 'git@github.com:')):
                    raise ValueError("Invalid GitHub repository URL format")
                
                # Check if required credentials are provided for private repos
                if repo_url.startswith('https://github.com/') and not (username and token):
                    logger.warning("GitHub credentials not provided. Access to private repositories may be limited.")
                
                self.dbt_tools = DbtSearchTools(repo_url, username, token)
                logger.info(f"Initialized DBT tools with repository: {repo_url}")
            except ValueError as e:
                logger.error(f"Invalid repository configuration: {str(e)}")
                self.dbt_tools = None
            except Exception as e:
                logger.error(f"Error initializing DBT tools: {str(e)}")
                self.dbt_tools = None
        else:
            logger.info("No valid repository URL provided, skipping DBT tools initialization")
        
        self.agent_graph = self._create_agent_graph()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the agent workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_question", self._parse_question)
        workflow.add_node("search_dbt", self._search_dbt)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.add_edge("parse_question", "search_dbt")
        workflow.add_edge("search_dbt", "generate_response")
        
        # Set entry point
        workflow.set_entry_point("parse_question")
        
        return workflow.compile()

    def _parse_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the user's question to determine intent and entities."""
        try:
            # Get the last message
            messages = state["messages"]
            last_message = messages[-1].content
            
            # Create prompt for question analysis
            prompt = f"""
            Analyze the following question about DBT models and determine:
            1. Question type (MODEL_INFO, LINEAGE, DEPENDENCIES, CODE_ENHANCEMENT, DOCUMENTATION, DEVELOPMENT, GENERAL)
            2. Key entities (model names, code references, etc.)
            3. Primary intent
            4. Rephrased question for better search

            Question: {last_message}

            Return ONLY a JSON object with these fields:
            {{
                "question_type": "TYPE",
                "entities": ["entity1", "entity2"],
                "intent": "intent description",
                "rephrased_question": "rephrased version"
            }}

            Question types:
            - MODEL_INFO: Questions about model structure, purpose, and functionality
            - LINEAGE: Questions about model relationships and dependencies
            - DEPENDENCIES: Questions about model dependencies and impact
            - CODE_ENHANCEMENT: Questions about improving or optimizing code
            - DOCUMENTATION: Questions about generating or improving documentation
            - DEVELOPMENT: Questions about implementation and development tasks
            - GENERAL: Other types of questions
            """
            
            # Get analysis from LLM
            response = self._safe_llm_call(self.llm, prompt, "question_analysis")
            
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse the response
            analysis = QuestionAnalysis.parse_raw(response)
            
            # Update state
            state["question_analysis"] = analysis.dict()
            return state
            
        except Exception as e:
            logger.error(f"Error parsing question: {str(e)}")
            state["question_analysis"] = {
                "question_type": "GENERAL",
                "entities": [],
                "intent": "general inquiry",
                "rephrased_question": last_message
            }
            return state

    def _search_dbt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Search DBT models based on the question analysis."""
        try:
            analysis = state["question_analysis"]
            entities = analysis.get("entities", [])
            
            results = {}
            
            # Only proceed with DBT search if we have DBT tools initialized
            if self.dbt_tools:
                # Handle different question types
                if analysis["question_type"] == "MODEL_INFO":
                    # Search for specific models or columns
                    for entity in entities:
                        # First try to find the model containing this column
                        model_result = self.dbt_tools.search_model(entity)
                        if model_result["status"] == "success":
                            results[entity] = model_result
                            
                            # If this was found by column search, add more context
                            if model_result.get("search_type") == "column":
                                # Search for related calculations
                                related_models = self.dbt_tools.search_by_pattern(entity)
                                if related_models:
                                    results[entity]["related_models"] = related_models
                                    
                                # Extract calculation details if available
                                model_info = model_result["model_info"]
                                for column in model_info.get("columns", []):
                                    if column["name"].lower() == entity.lower():
                                        if column.get("calculation"):
                                            results[entity]["calculation_details"] = column["calculation"]
                                        if column.get("description"):
                                            results[entity]["column_description"] = column["description"]
                                        results[entity]["sql_expression"] = column["expression"]

                elif analysis["question_type"] == "LINEAGE":
                    # Get lineage information
                    for entity in entities:
                        lineage_result = self.dbt_tools.dependency_tracer.get_model_lineage(entity)
                        if lineage_result:
                            results[entity] = lineage_result

                elif analysis["question_type"] == "DEPENDENCIES":
                    # Get dependency information
                    for entity in entities:
                        dep_result = self.dbt_tools.get_model_dependencies(entity)
                        if dep_result["status"] == "success":
                            results[entity] = dep_result

                else:
                    # General search using pattern matching
                    pattern = "|".join(entities) if entities else ".*"
                    matching_models = self.dbt_tools.search_by_pattern(pattern)
                    results["matching_models"] = matching_models
            else:
                # If no DBT tools available, provide a more helpful error message
                logger.warning("DBT tools not initialized - GitHub repository configuration required")
                results = {
                    "status": "no_dbt_tools",
                    "message": "Unable to search DBT models. Please configure a GitHub repository containing your DBT models.",
                    "setup_required": True,
                    "setup_steps": [
                        "Configure a GitHub repository in the settings",
                        "Ensure the repository contains your DBT models",
                        "Provide valid GitHub credentials"
                    ]
                }

            # Update state
            state["dbt_results"] = results
            return state
            
        except Exception as e:
            logger.error(f"Error searching DBT: {str(e)}")
            state["dbt_results"] = {
                "error": str(e),
                "status": "error",
                "message": f"Error searching DBT models: {str(e)}"
            }
            return state

    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response based on the search results."""
        try:
            analysis = state["question_analysis"]
            results = state["dbt_results"]
            
            # Check if DBT tools are not available
            if results.get("status") == "no_dbt_tools":
                response = f"""
## Setup Required
I notice that DBT tools are not properly configured. To help you understand how `{analysis.get("entities", [""])[0]}` is calculated, I need access to your DBT models.

### Required Setup Steps:
{chr(10).join(f"- {step}" for step in results.get("setup_steps", []))}

### Next Steps:
1. Configure your GitHub repository in the settings
2. Ensure your DBT models are in the repository
3. Once configured, I can provide detailed information about your specific model calculations

Would you like me to help you with the setup process?
"""
            else:
                # Get appropriate instructions based on question type
                instructions = self._get_instructions_for_type(
                    analysis["question_type"],
                    analysis["rephrased_question"]
                )

                # Create response prompt with detailed results
                prompt = f"""
                Based on the following question and search results, generate a comprehensive response.
                
                QUESTION: {analysis["rephrased_question"]}
                QUESTION TYPE: {analysis["question_type"]}
                
                SEARCH RESULTS:
                {json.dumps(results, indent=2)}
                
                INSTRUCTIONS:
                {instructions}
                
                Please provide a well-structured response that includes:
                1. A clear overview of the calculation or logic
                2. Step-by-step explanation of the code
                3. Code blocks with syntax highlighting
                4. Related dependencies and their roles
                5. Any important considerations or notes
                
                Format the response in markdown with the following structure:
                
                ## Overview
                [Brief explanation of what the code does]
                
                ## Code Analysis
                ### Step 1: [First major step]
                ```sql
                [Relevant SQL code with comments]
                ```
                [Explanation of this step]
                
                ### Step 2: [Second major step]
                ```sql
                [Relevant SQL code with comments]
                ```
                [Explanation of this step]
                
                ## Dependencies
                - [List of dependencies and their roles]
                
                ## Notes
                - [Important considerations]
                - [Performance implications]
                - [Best practices]
                """

                # Generate response
                response = self._safe_llm_call(self.llm, prompt, "response_generation")
            
            # Update state
            state["final_response"] = response
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["final_response"] = f"I apologize, but I encountered an error: {str(e)}"
            return state

    def _get_instructions_for_type(self, question_type: str, question: str) -> str:
        """Get specific instructions based on question type."""
        instruction_map = {
            "MODEL_INFO": self._get_model_explanation_instructions,
            "LINEAGE": self._get_lineage_instructions,
            "DEPENDENCIES": self._get_dependency_instructions,
            "CODE_ENHANCEMENT": self._get_code_enhancement_instructions,
            "DOCUMENTATION": self._get_documentation_instructions,
            "DEVELOPMENT": self._get_development_instructions,
            "GENERAL": self._get_general_instructions
        }
        
        instruction_func = instruction_map.get(question_type, self._get_general_instructions)
        return instruction_func(question)

    def _get_model_explanation_instructions(self, query: str) -> str:
        """Get instructions for model explanation questions."""
        return """
        Provide a concise DBT model explanation focusing on:
        1. Purpose and business logic
        2. Key transformations
        3. Dependencies and lineage
        4. Performance considerations

        FORMAT:
        ## Overview
        [2-3 sentences about the model's purpose]

        ## Key Components
        - Transformations
        - Dependencies
        - Performance Notes

        ## Business Logic
        [2-3 sentences about the business rules]
        """

    def _get_lineage_instructions(self, query: str) -> str:
        """Get instructions for lineage questions."""
        return """
        Provide a clear lineage analysis focusing on:
        1. Direct dependencies
        2. Impact analysis
        3. Critical paths

        FORMAT:
        ## Lineage Overview
        [1-2 sentences about the model's position]

        ## Dependencies
        - Upstream: [List key dependencies]
        - Downstream: [List key consumers]

        ## Critical Paths
        [Highlight important dependency chains]
        """

    def _get_dependency_instructions(self, query: str) -> str:
        """Get instructions for dependency questions."""
        return """
        Analyze dependencies focusing on:
        1. Direct dependencies
        2. Impact assessment
        3. Optimization opportunities

        FORMAT:
        ## Dependencies
        - Direct: [List immediate dependencies]
        - Indirect: [List transitive dependencies]

        ## Impact Analysis
        [Brief impact assessment]

        ## Optimization
        [Key optimization suggestions]
        """

    def _get_code_enhancement_instructions(self, query: str) -> str:
        """Get instructions for code enhancement tasks."""
        return """
        Provide comprehensive code enhancement guidance following these steps:

        1. Code Analysis
        - Current implementation review
        - Performance bottlenecks
        - Code quality issues
        - Best practices violations

        2. Enhancement Strategy
        - Prioritized improvements
        - Impact assessment
        - Risk evaluation
        - Dependencies consideration

        3. Implementation Plan
        FORMAT:
        ## Current State
        [Brief analysis of existing code]

        ## Enhancement Areas
        - Performance
          * Current bottlenecks
          * Optimization opportunities
          * Expected improvements
        - Code Quality
          * Issues identified
          * Refactoring needs
          * Best practices to implement
        - Architecture
          * Design improvements
          * Pattern applications
          * Scalability considerations

        ## Step-by-Step Implementation
        1. [First step with code example]
        2. [Second step with code example]
        3. [Subsequent steps...]

        ## Testing Strategy
        - Unit tests to add/modify
        - Integration test scenarios
        - Performance benchmarks

        ## Validation Steps
        1. [Validation step 1]
        2. [Validation step 2]
        3. [Final verification]

        ## Rollout Plan
        - Development phase
        - Testing phase
        - Deployment steps
        - Monitoring plan
        """

    def _get_documentation_instructions(self, query: str) -> str:
        """Get instructions for documentation tasks."""
        return """
        Generate documentation focusing on:
        1. Purpose and usage
        2. Technical details
        3. Examples and references

        FORMAT:
        ## Overview
        [Brief purpose and usage]

        ## Technical Details
        - Parameters
        - Dependencies
        - Configuration

        ## Examples
        [Usage examples]
        """

    def _get_development_instructions(self, query: str) -> str:
        """Get instructions for development tasks."""
        return """
        Provide comprehensive development guidance following these steps:

        1. Requirements Analysis
        - Functional requirements
        - Technical requirements
        - Constraints and limitations
        - Dependencies

        2. Architecture Design
        - System components
        - Data flow
        - Integration points
        - Security considerations

        3. Implementation Plan
        FORMAT:
        ## Overview
        [Brief project/feature description]

        ## Technical Stack
        - Frontend: [Technologies]
        - Backend: [Technologies]
        - Database: [Technologies]
        - Infrastructure: [Requirements]

        ## Architecture
        - Component diagram
        - Data flow
        - Security model
        - Scalability approach

        ## Development Phases
        1. Setup Phase
           - Environment setup
           - Dependencies installation
           - Configuration setup
           - Initial project structure

        2. Core Development
           - Database schema
           - API endpoints
           - Business logic
           - Frontend components

        3. Integration
           - Component integration
           - API integration
           - Third-party services
           - Data migration

        4. Testing
           - Unit tests
           - Integration tests
           - End-to-end tests
           - Performance tests

        5. Deployment
           - Build process
           - Deployment pipeline
           - Monitoring setup
           - Backup strategy

        ## Code Structure
        ```
        project/
        ├── src/
        │   ├── components/
        │   ├── services/
        │   ├── utils/
        │   └── config/
        ├── tests/
        ├── docs/
        └── scripts/
        ```

        ## Implementation Steps
        1. [Step 1 with code example]
        2. [Step 2 with code example]
        3. [Subsequent steps...]

        ## Testing Strategy
        - Unit testing approach
        - Integration testing
        - Performance testing
        - Security testing

        ## Deployment Checklist
        - [ ] Environment setup
        - [ ] Dependencies installed
        - [ ] Configuration verified
        - [ ] Tests passing
        - [ ] Security checks
        - [ ] Performance benchmarks
        - [ ] Documentation updated
        - [ ] Backup verified
        - [ ] Monitoring configured
        - [ ] Rollback plan ready

        ## Maintenance Plan
        - Monitoring strategy
        - Update schedule
        - Backup procedures
        - Performance optimization
        - Security patches
        """

    def _get_general_instructions(self, query: str) -> str:
        """Get instructions for general questions."""
        return """
        Provide a focused response addressing:
        1. Direct answer to the question
        2. Relevant context
        3. Next steps if applicable

        FORMAT:
        ## Answer
        [Direct response]

        ## Context
        [Relevant background]

        ## Next Steps
        [Suggested actions]
        """

    def _get_response_prompt(self, question: str, question_type: str, results: str, instructions: str) -> str:
        """Create the prompt for generating the final response."""
        return f"""
        Based on the following question and search results, generate a comprehensive response.
        
        QUESTION: {question}
        QUESTION TYPE: {question_type}
        
        SEARCH RESULTS:
        {results}
        
        INSTRUCTIONS:
        {instructions}
        
        Please provide a well-structured response that directly addresses the question while incorporating the search results.
        """

    def _safe_llm_call(self, model: ChatOllama, prompt: str, purpose: str = "general", max_retries: int = 2) -> str:
        """Safely call the LLM with retries."""
        for attempt in range(max_retries):
            try:
                response = model.invoke(prompt)
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error in LLM call for {purpose}: {str(e)}")
                    return f"I apologize, but I encountered an error while processing your request: {str(e)}"
                logger.warning(f"Retry {attempt + 1} for {purpose} after error: {str(e)}")

    def process_question(self, question: str, conversation_id: str = None, thread_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user question and generate a response."""
        try:
            # Initialize state with proper message type
            state = {
                "messages": [HumanMessage(content=question)],
                "question_analysis": {},
                "dbt_results": {},
                "final_response": ""
            }

            # Run the agent graph
            final_state = self.agent_graph.invoke(state)

            # Create response metadata
            response_metadata = self._create_response_metadata(
                final_state,
                "GENERAL",
                final_state["question_analysis"].get("question_type", "GENERAL")
            )

            return {
                "status": "success",
                "response": final_state["final_response"],
                "metadata": response_metadata,
                "conversation_id": conversation_id,
                "thread_id": thread_id
            }

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return self._create_error_response(
                state,
                "GENERAL",
                str(e)
            )

    def _create_response_metadata(self, state: Dict[str, Any], initial_type: str, question_type: str) -> Dict[str, Any]:
        """Create metadata for the response."""
        return {
            "question_type": question_type,
            "initial_type": initial_type,
            "entities": state["question_analysis"].get("entities", []),
            "intent": state["question_analysis"].get("intent", ""),
            "timestamp": datetime.now().isoformat()
        }

    def _create_error_response(self, state: Dict[str, Any], question_type: str, error: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "status": "error",
            "response": f"I apologize, but I encountered an error: {error}",
            "metadata": {
                "question_type": question_type,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
        }

def create_data_architect_agent(repo_url_or_tools: Union[str, SearchTools] = "", username: str = "", token: str = ""):
    """Factory function to create a data architect agent.
    
    Args:
        repo_url_or_tools: Either a repository URL string or a SearchTools object
        username: GitHub username (optional)
        token: GitHub token (optional)
    """
    # If repo_url_or_tools is a SearchTools object, extract the repo URL
    if isinstance(repo_url_or_tools, SearchTools):
        # Get the latest GitHub configuration from the database
        try:
            with ChatDatabase()._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value FROM settings 
                    WHERE key = 'github_connectors' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                
                if result:
                    configs = json.loads(result[0])
                    if configs:
                        # Use the most recent configuration
                        latest_config = configs[-1]
                        repo_url = latest_config.get('repoUrl', '')
                        username = latest_config.get('username', '')
                        token = latest_config.get('token', '')
                        return DataArchitectAgent(repo_url, username, token)
        except Exception as e:
            logger.error(f"Error getting GitHub configuration: {str(e)}")
    
    # If we get here, either repo_url_or_tools is a string or we couldn't get the config
    return DataArchitectAgent(
        repo_url=repo_url_or_tools if isinstance(repo_url_or_tools, str) else "",
        username=username,
        token=token
    ) 