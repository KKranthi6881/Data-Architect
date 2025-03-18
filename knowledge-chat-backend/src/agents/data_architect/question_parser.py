from typing import Dict, List, Any, Annotated, Sequence, TypedDict
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
import json
from datetime import datetime
import re

from src.tools import SearchTools
from src.db.database import ChatDatabase

logger = logging.getLogger(__name__)

# Define default analysis structure at module level
DEFAULT_ANALYSIS = {
    "rephrased_question": "",
    "question_type": "unknown",  # New field to identify question type
    "key_points": [],
    "business_context": {
        "domain": "Unknown",
        "primary_objective": "Not specified",
        "key_entities": [],
        "business_impact": "Not analyzed"
    },
    "technical_context": {  # New section for technical analysis
        "data_stack": [],
        "relevant_components": [],
        "dependencies": [],
        "technical_considerations": []
    },
    "implementation_guidance": {  # New section for code/implementation guidance
        "approach": "",
        "suggested_steps": [],
        "code_references": []
    },
    "assumptions": [],
    "clarifying_questions": [],
    "confidence_score": 0.0
}

# Define state type
class ParserState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    doc_context: Annotated[Dict, "Documentation context"]
    sql_context: Annotated[Dict, "SQL schema context"]
    github_context: Annotated[Dict, "Github context"]
    business_analysis: Annotated[Dict, "Business analysis results"]
    feedback_status: Annotated[str, "Current status"]
    confidence_score: Annotated[float, "Confidence in analysis"]

# Define structured outputs
class BusinessAnalysis(BaseModel):
    rephrased_question: str = Field(description="Clear restatement of the question")
    question_type: str = Field(description="Type of question (business, technical, hybrid)")
    key_points: List[str] = Field(description="Key points and objectives")
    business_context: Dict[str, Any] = Field(description="Business context information")
    technical_context: Dict[str, Any] = Field(description="Technical context information")
    implementation_guidance: Dict[str, Any] = Field(description="Implementation and code guidance")
    assumptions: List[str] = Field(description="Assumptions to verify")
    clarifying_questions: List[str] = Field(description="Questions about requirements")
    confidence_score: float = Field(description="Confidence in analysis (0-1)")

def create_parser_agent(tools: SearchTools):
    # Initialize model
    llm = ChatOllama(
        model="deepseek-r1:8b",  
        temperature=0.1,
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parser
    output_parser = PydanticOutputParser(pydantic_object=BusinessAnalysis)

    # Create prompt template with corrected JSON instructions
    template = """You are an expert data consultant helping users understand and solve both business and technical data questions.

USER QUESTION:
{question}

BUSINESS DOCUMENTATION:
{doc_context}

SQL SCHEMA INFORMATION:
{sql_context}

GITHUB CODE REPOSITORY:
{github_context}

TASK:
Analyze this question, determine if it's a business question, technical question, or hybrid, and provide a structured understanding with relevant information.

GUIDELINES:
- First, identify the question type (business, technical, or hybrid)
- Never generalize the content. If you do not find specific info, acknowledge the limitation
- For business questions: focus on objectives, requirements, and impact
- For technical questions: focus on implementation details, code patterns, and technical considerations
- For hybrid questions: balance both aspects appropriately
- Only reference information found in documentation and schema
- Identify any missing context or information
- Provide specific guidance relevant to the tech stack (especially Snowflake and dbt)

QUESTION TYPES TO IDENTIFY:
1. BUSINESS QUESTIONS: About metrics, KPIs, reporting needs, business processes
2. TECHNICAL QUESTIONS: About implementation, code, data pipelines, testing, architecture
   - Development: Creating new code/models/tables
   - Enhancement: Modifying existing code/models/tables
   - Debugging: Fixing issues or understanding errors
   - Performance: Optimization and efficiency
   - Architecture: Design patterns and structure
3. HYBRID QUESTIONS: Combining business and technical aspects

FORMATTING INSTRUCTIONS:
Respond ONLY with a valid JSON object with no other text. Format your response exactly like this example:

{{
  "rephrased_question": "Clear restatement of the question",
  "question_type": "business|technical|hybrid",
  "key_points": [
    "Key point 1",
    "Key point 2"
  ],
  "business_context": {{
    "domain": "Business domain area",
    "primary_objective": "Main business goal",
    "key_entities": ["Entity 1", "Entity 2"],
    "business_impact": "How this affects business"
  }},
  "technical_context": {{
    "data_stack": ["Snowflake", "dbt", "Other relevant technologies"],
    "relevant_components": ["Tables", "Models", "Scripts"],
    "dependencies": ["Related systems", "Prerequisites"],
    "technical_considerations": ["Performance concerns", "Design patterns"]
  }},
  "implementation_guidance": {{
    "approach": "High-level approach to solving",
    "suggested_steps": ["Step 1", "Step 2"],
    "code_references": ["Reference to relevant code patterns"]
  }},
  "assumptions": [
    "Assumption 1",
    "Assumption 2"
  ],
  "clarifying_questions": [
    "Question about requirement 1",
    "Question about requirement 2"
  ],
  "confidence_score": 0.85
}}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "doc_context", "sql_context", "github_context"]
    )

    def process_question(state: ParserState) -> Dict:
        """Process the question and generate business analysis"""
        try:
            messages = state['messages']
            question = messages[-1].content if messages else ""
            doc_context = state.get('doc_context', {})
            sql_context = state.get('sql_context', {})
            github_context = state.get('github_context', {})  # Get GitHub context
            
            # Debug: print number of results in each context
            doc_count = len(doc_context.get('results', []))
            sql_count = len(sql_context.get('results', []))
            github_count = len(github_context.get('results', []))
            logger.info(f"Context counts - Doc: {doc_count}, SQL: {sql_count}, GitHub: {github_count}")
            
            # Debug github_context structure if it seems empty but shouldn't be
            if not github_context:
                logger.warning(f"GitHub context is empty: {github_context}")
            elif 'results' not in github_context:
                logger.warning(f"GitHub context missing 'results' key. Keys found: {list(github_context.keys())}")
                # Try to fix it by finding any array in the context that might be the results
                for key, value in github_context.items():
                    if isinstance(value, list):
                        logger.info(f"Found potential results array under key '{key}' with {len(value)} items")
                        github_context['results'] = value
                        break
            elif not github_context.get('results'):
                logger.warning(f"GitHub 'results' is empty or null: {github_context.get('results')}")
            else:
                logger.info(f"GitHub context has {len(github_context['results'])} results")
            
            # Check if we have any context documents or schemas or GitHub repos
            has_context = (doc_count > 0 or sql_count > 0 or github_count > 0)
            logger.info(f"Has context: {has_context}")
            
            # If github context exists but others don't, log a confirmation
            if github_count > 0 and doc_count == 0 and sql_count == 0:
                logger.info("Using only GitHub context for analysis")
            
            # Format documentation context
            formatted_doc_context = "\n".join([
                f"[{doc.get('type', 'Doc')}]\n{doc.get('content', '')}\n---"
                for doc in doc_context.get('results', [])
            ])
            
            # Format SQL schema context
            formatted_sql_context = "\n".join([
                f"[{sql.get('type', 'SQL')}]\n{sql.get('content', '')}\n---"
                for sql in sql_context.get('results', [])
            ])
            
            # Format GitHub context
            formatted_github_context = ""
            for github in github_context.get('results', []):
                # Extract repo name from the repository object
                repo_name = "Unknown Repo"
                if isinstance(github.get('repository'), dict):
                    repo_name = github.get('repository', {}).get('name', 'Unknown Repo')
                
                # Get the file path if available
                file_path = "Unknown File"
                if isinstance(github.get('file'), dict):
                    file_path = github.get('file', {}).get('path', 'Unknown File')
                
                # Get the content
                content = github.get('content', '')
                
                # Format the GitHub context entry
                formatted_github_context += f"[GitHub: {repo_name} - {file_path}]\n{content}\n---\n"
            
            # Add a message if no GitHub context is available
            if not formatted_github_context:
                formatted_github_context = "No relevant GitHub repository code found."
            
            # If no context is available, create a special response
            if not has_context:
                logger.warning("No context available (documents, schemas, or GitHub) for question processing")
                no_context_analysis = DEFAULT_ANALYSIS.copy()
                no_context_analysis.update({
                    "rephrased_question": question,
                    "question_type": "no_context",
                    "key_points": ["No relevant documentation or schema information is available"],
                    "business_context": {
                        "domain": "Information Required",
                        "primary_objective": "Please upload relevant files or schemas first",
                        "key_entities": [],
                        "business_impact": "Cannot analyze without context"
                    },
                    "technical_context": {
                        "data_stack": [],
                        "relevant_components": [],
                        "dependencies": [],
                        "technical_considerations": ["Upload dbt models, SQL files, or documentation to enable analysis"]
                    },
                    "implementation_guidance": {
                        "approach": "Please upload relevant files to proceed",
                        "suggested_steps": [
                            "Upload dbt models and schema files to provide context",
                            "Upload SQL files if working directly with Snowflake",
                            "Upload documentation files for business context",
                            "Or connect GitHub repositories for code examples"
                        ],
                        "code_references": []
                    },
                    "assumptions": ["No data available to analyze"],
                    "clarifying_questions": [
                        "Do you have dbt models you can upload?",
                        "Do you have Snowflake SQL scripts available?",
                        "Is there documentation that describes your data models?"
                    ],
                    "confidence_score": 0.0
                })
                return {
                    **state,
                    "business_analysis": no_context_analysis,
                    "feedback_status": "pending",
                    "confidence_score": 0.0
                }
            
            # Generate analysis
            formatted_prompt = prompt.format(
                question=question,
                doc_context=formatted_doc_context,
                sql_context=formatted_sql_context,
                github_context=formatted_github_context
            )
            
            # Log the first 500 characters of the prompt for debugging
            logger.debug(f"Prompt first 500 chars: {formatted_prompt[:500]}...")
            
            # Call the LLM with error handling
            try:
                response = llm.invoke(formatted_prompt)
                response_text = response.content
                
                # Log the first 500 characters of the response for debugging
                logger.debug(f"LLM response first 500 chars: {response_text[:500]}...")
                
                # Pre-process response text to fix common JSON issues
                response_text = preprocess_json_response(response_text)
                
                # Parse the preprocessed JSON
                try:
                    analysis = json.loads(response_text)
                    
                    # Validate minimum structure requirements
                    if not isinstance(analysis, dict):
                        logger.warning(f"LLM returned non-dict object: {type(analysis)}")
                        analysis = {"rephrased_question": question}
                    
                    # Ensure the analysis has all required fields
                    analysis = {**DEFAULT_ANALYSIS, **analysis}
                    logger.info(f"Successfully parsed analysis with keys: {list(analysis.keys())}")
                    
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parse error: {json_err} in text: {response_text[:200]}...")
                    # Fall back to error handling
                    raise ValueError(f"Failed to parse LLM response as JSON: {json_err}")
                    
            except Exception as llm_error:
                logger.error(f"LLM error: {llm_error}")
                raise ValueError(f"LLM processing error: {llm_error}")
                
            # Return the processed state
            return {
                **state,
                "business_analysis": analysis,
                "feedback_status": "pending",
                "confidence_score": float(analysis["confidence_score"])
            }
            
        except Exception as e:
            logger.error(f"Error in process_question: {e}", exc_info=True)
            # Return an error analysis as the fallback
            error_analysis = DEFAULT_ANALYSIS.copy()
            error_analysis.update({
                "rephrased_question": question if 'question' in locals() else "Error processing question",
                "question_type": "error",
                "key_points": ["Error in analysis process"],
                "business_context": {
                    "domain": "Error",
                    "primary_objective": str(e),
                    "key_entities": [],
                    "business_impact": "Analysis failed"
                },
                "assumptions": ["Please try again"],
                "clarifying_questions": ["Could you rephrase your question?"],
                "confidence_score": 0.0
            })
            
            return {
                **state,
                "business_analysis": error_analysis,
                "feedback_status": "pending",
                "confidence_score": 0.0
            }

    def preprocess_json_response(text: str) -> str:
        """
        Preprocess LLM response text to fix common JSON issues.
        """
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Check if the text is wrapped in ```json ... ``` code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, text)
        if matches:
            text = matches[0].strip()
        
        # Check if the text starts with a proper JSON object
        if not text.startswith('{'):
            # If it starts with "rephrased_question" or another key, wrap it in braces
            if '"rephrased_question"' in text or "'rephrased_question'" in text:
                logger.info("Adding missing opening brace to JSON response")
                text = '{' + text
        
        # Check if the text ends with a proper JSON object closure
        if not text.endswith('}'):
            # Add closing brace if needed
            if text.count('{') > text.count('}'):
                logger.info("Adding missing closing brace to JSON response")
                text = text + '}'
        
        # Balance braces if needed
        open_braces = text.count('{')
        close_braces = text.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            text = text + ('}' * (open_braces - close_braces))
        elif close_braces > open_braces:
            # Remove excess closing braces from the end
            excess = close_braces - open_braces
            if text.endswith('}' * excess):
                text = text[:-excess]
        
        # Ensure proper quotes for all keys
        # This is a simple fix for cases where keys are not properly quoted
        key_fix_pattern = r'([{,]\s*)(\w+)(\s*:)'
        text = re.sub(key_fix_pattern, r'\1"\2"\3', text)
        
        # Fix trailing commas before closing braces (which are invalid in JSON)
        text = re.sub(r',\s*}', '}', text)
        
        # Log the preprocessed text for debugging (truncated)
        logger.debug(f"Preprocessed JSON: {text[:200]}...")
        
        return text

    # Build the graph
    graph = StateGraph(ParserState)
    
    # Add nodes
    graph.add_node("process_question", process_question)
    
    # Add edges
    graph.add_edge(START, "process_question")
    graph.add_edge("process_question", END)

    # Create SQLite saver
    db_path = str(Path(__file__).parent.parent.parent.parent / "chat_history.db")
    conn = connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile()

class QuestionParserSystem:
    def __init__(self, tools: SearchTools):
        self.app = create_parser_agent(tools)
        self.tools = tools
        self.db = ChatDatabase()
        self._lock = Lock()
        self.active_threads = {}  # Store active thread information

    async def parse_question(self, question: str, thread_id: str = None, conversation_id: str = None) -> Dict[str, Any]:
        """Process a question through the parser system"""
        try:
            # Generate unique IDs if not provided
            if not thread_id:
                thread_id = str(uuid.uuid4())
            
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            else:
                # If conversation_id is provided but no thread_id, use it as thread_id too
                if not thread_id:
                    thread_id = conversation_id
            
            # Store the relationship between conversation and thread
            self.active_threads[conversation_id] = {
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Get documentation context
            doc_results = self.tools.search_documentation(question)
            
            # Get SQL schema context
            sql_results = {}
            try:
                sql_results = self.tools.search_sql_schema(question)
            except Exception as sql_error:
                logger.warning(f"Error searching SQL schema: {sql_error}. Continuing with documentation only.")
                # Create empty results if SQL search fails
                sql_results = {"results": []}
            
            # Log available collections
            try:
                collections = self.tools.db_manager.client.list_collections()
                collection_names = [c.name for c in collections]
                logger.info(f"Available collections: {collection_names}")
            except Exception as e:
                logger.warning(f"Error listing collections: {e}")
                
            # Get GitHub repository context
            github_results = {}
            try:
                logger.info(f"Starting GitHub repository search for query: '{question}'")
                github_results = self.tools.search_github_repos(question)
                github_result_count = len(github_results.get('results', []))
                logger.info(f"GitHub search returned {github_result_count} results")
                
                # Debug github_results structure
                if 'results' not in github_results:
                    logger.warning("GitHub search didn't return a 'results' key")
                    github_results['results'] = []
                elif not github_results['results']:
                    logger.warning("GitHub search returned empty results array")
                else:
                    logger.info(f"GitHub search found {len(github_results['results'])} results")
                    if github_results['results']:
                        logger.debug(f"First GitHub result keys: {list(github_results['results'][0].keys())}")
                
                # Make sure we have the correct structure for github_results
                if isinstance(github_results, dict) and 'status' in github_results and 'results' in github_results:
                    # Extract just the results array
                    github_results = {'results': github_results['results']}
                elif isinstance(github_results, dict) and 'results' not in github_results:
                    # Create a results array if missing
                    github_results = {'results': list(github_results.values())[0] if github_results else []}
                
                # Add type field to each GitHub result for better identification
                for result in github_results.get('results', []):
                    if isinstance(result, dict) and 'type' not in result:
                        result['type'] = 'GitHub'
                
                # Log the structure being passed to invoke
                logger.info(f"GitHub context structure: {len(github_results.get('results', []))} results with keys: {list(github_results.keys())}")
                
            except Exception as github_error:
                logger.warning(f"Error searching GitHub repositories: {github_error}. Continuing without GitHub context.")
                # Create empty results if GitHub search fails
                github_results = {"results": []}
            
            # Execute the query with better error handling
            try:
                with self._lock:
                    result = self.app.invoke(
                        {
                            "messages": [HumanMessage(content=question)],
                            "doc_context": doc_results,
                            "sql_context": sql_results,
                            "github_context": github_results,  # Add GitHub context
                            "business_analysis": {},
                            "feedback_status": None,
                            "confidence_score": 0.0
                        },
                        {"configurable": {"thread_id": thread_id}}
                    )
                
                # Save conversation to database
                business_analysis = result.get("business_analysis", {})
                
                # Save conversation to database
                github_result_count = len(github_results.get('results', []))
                logger.info(f"Saving conversation with {github_result_count} GitHub results")

                response_data = {
                    "query": question,
                    "output": business_analysis.get("rephrased_question", "No response available"),
                    "technical_details": json.dumps(business_analysis),
                    "code_context": json.dumps({
                        "documentation": doc_results,
                        "sql_schema": sql_results,
                        "github_code": github_results  # Add GitHub context
                    }),
                    "thread_id": thread_id  # Store thread_id in the conversation data
                }

                # Verify code_context before saving
                code_context_json = json.dumps({
                    "documentation": doc_results,
                    "sql_schema": sql_results,
                    "github_code": github_results
                })
                logger.info(f"Code context JSON length: {len(code_context_json)}")
                logger.info(f"GitHub results in code_context: {len(github_results.get('results', []))}")

                # Save to database using the conversations table
                with self._lock:
                    self.db.save_conversation(conversation_id, response_data)
                
                # Add thread_id to the returned business analysis
                business_analysis["thread_id"] = thread_id
                business_analysis["conversation_id"] = conversation_id
                
                return business_analysis
                
            except Exception as e:
                # Log more details about the error
                logger.error(f"Error in question processing: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                
                # If we have GitHub results, create a GitHub-only analysis
                if len(github_results.get('results', [])) > 0:
                    logger.info(f"Creating fallback analysis with {len(github_results.get('results', []))} GitHub results")
                    
                    # Create fallback business analysis with GitHub info
                    business_analysis = DEFAULT_ANALYSIS.copy()
                    business_analysis.update({
                        "rephrased_question": question,
                        "question_type": "github_only",
                        "key_points": ["Analysis based only on GitHub code examples"],
                        "business_context": {
                            "domain": "Code Analysis",
                            "primary_objective": "Understand relevant code patterns",
                            "key_entities": [],
                            "business_impact": "Code implementation guidance"
                        },
                        "technical_context": {
                            "data_stack": ["dbt", "Snowflake"],
                            "relevant_components": [],
                            "dependencies": [],
                            "technical_considerations": ["Based on GitHub code examples"]
                        },
                        "implementation_guidance": {
                            "approach": "Study similar implementations in the code examples",
                            "suggested_steps": ["Review the GitHub code examples below"],
                            "code_references": []
                        },
                        "assumptions": ["Processing based on GitHub code examples only"],
                        "clarifying_questions": ["Could you provide more business context?"],
                        "confidence_score": 0.4
                    })
                    
                    # Add GitHub information to the analysis
                    repo_names = []
                    file_paths = []
                    code_snippets = []
                    
                    for result in github_results.get('results', [])[:3]:  # Use top 3 results
                        if isinstance(result.get('repository'), dict):
                            repo_name = result.get('repository', {}).get('name', 'Unknown')
                            repo_names.append(repo_name)
                        
                        if isinstance(result.get('file'), dict):
                            file_path = result.get('file', {}).get('path', 'Unknown')
                            file_paths.append(file_path)
                        
                        # Get a snippet of the content (first 200 chars)
                        content = result.get('content', '')[:200]
                        if content:
                            code_snippets.append(f"From {repo_name}/{file_path}: {content}...")
                    
                    business_analysis['technical_context']['relevant_components'] = file_paths
                    business_analysis['implementation_guidance']['code_references'] = [
                        f"Example in {repo} repository" for repo in repo_names
                    ]
                    
                    # Add snippets to the key points for more context
                    if code_snippets:
                        business_analysis['key_points'].extend([
                            "Found relevant code examples that may help with your implementation"
                        ])
                else:
                    # No GitHub results, return a generic error
                    business_analysis = DEFAULT_ANALYSIS.copy()
                    business_analysis.update({
                        "rephrased_question": question,
                        "question_type": "error",
                        "key_points": ["Error analyzing the question"],
                        "business_context": {
                            "domain": "Error",
                            "primary_objective": "Please try again or rephrase your question",
                            "key_entities": [],
                            "business_impact": "Unable to analyze"
                        },
                        "confidence_score": 0.0
                    })
            
            # Save conversation to database
            response_data = {
                "query": question,
                "output": business_analysis.get("rephrased_question", "No response available"),
                "technical_details": json.dumps(business_analysis),
                "code_context": json.dumps({
                    "documentation": doc_results,
                    "sql_schema": sql_results,
                    "github_code": github_results  # Add GitHub context
                }),
                "thread_id": thread_id  # Store thread_id in the conversation data
            }
            
            # Save to database using the conversations table
            with self._lock:
                self.db.save_conversation(conversation_id, response_data)
            
            # Add thread_id to the returned business analysis
            business_analysis["thread_id"] = thread_id
            business_analysis["conversation_id"] = conversation_id
            
            return business_analysis
            
        except Exception as e:
            logger.error(f"Error in parser system: {e}")
            error_analysis = DEFAULT_ANALYSIS.copy()
            error_analysis.update({
                "rephrased_question": "Error analyzing question",
                "key_points": ["Unable to analyze business requirements"],
                "business_context": {
                    "domain": "Error",
                    "primary_objective": str(e),
                    "key_entities": [],
                    "business_impact": "Analysis failed"
                },
                "assumptions": ["Please try again"],
                "clarifying_questions": ["Could you rephrase your question?"],
                "confidence_score": 0.0,
                "thread_id": thread_id if thread_id else str(uuid.uuid4()),
                "conversation_id": conversation_id if conversation_id else str(uuid.uuid4())
            })
            
            # Save error to database using conversations table
            error_conversation_id = conversation_id if conversation_id else str(uuid.uuid4())
            error_response = {
                "query": question,
                "output": "Error analyzing question",
                "technical_details": json.dumps(error_analysis),
                "code_context": "{}",
                "thread_id": thread_id if thread_id else str(uuid.uuid4())
            }
            
            with self._lock:
                self.db.save_conversation(error_conversation_id, error_response)
                
            return error_analysis

    def get_thread_conversations(self, thread_id: str) -> List[Dict]:
        """Get all conversations associated with a thread"""
        try:
            return self.db.get_conversations_by_thread(thread_id)
        except Exception as e:
            logger.error(f"Error getting thread conversations: {e}")
            return []
