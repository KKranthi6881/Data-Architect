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
        model="llama3.2:latest",  # Changed to llama2
        temperature=0.1,
        base_url="http://localhost:11434",
        timeout=120,
    )

    # Create parser
    output_parser = PydanticOutputParser(pydantic_object=BusinessAnalysis)

    # Create prompt template
    template = """You are an expert data consultant helping users understand and solve both business and technical data questions.

USER QUESTION:
{question}

BUSINESS DOCUMENTATION:
{doc_context}

SQL SCHEMA INFORMATION:
{sql_context}

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

IMPORTANT: Your response must be a valid JSON object with the following structure (no markdown, no explanations, just the JSON):
{
    "rephrased_question": "Clear restatement of the question",
    "question_type": "business|technical|hybrid",
    "key_points": [
        "Key point 1",
        "Key point 2"
    ],
    "business_context": {
        "domain": "Business domain area",
        "primary_objective": "Main business goal",
        "key_entities": ["Entity 1", "Entity 2"],
        "business_impact": "How this affects business"
    },
    "technical_context": {
        "data_stack": ["Snowflake", "dbt", "Other relevant technologies"],
        "relevant_components": ["Tables", "Models", "Scripts"],
        "dependencies": ["Related systems", "Prerequisites"],
        "technical_considerations": ["Performance concerns", "Design patterns"]
    },
    "implementation_guidance": {
        "approach": "High-level approach to solving",
        "suggested_steps": ["Step 1", "Step 2"],
        "code_references": ["Reference to relevant code patterns"]
    },
    "assumptions": [
        "Assumption 1",
        "Assumption 2"
    ],
    "clarifying_questions": [
        "Question about requirement 1",
        "Question about requirement 2"
    ],
    "confidence_score": 0.85
}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "doc_context", "sql_context"]
    )

    def process_question(state: ParserState) -> Dict:
        """Process the question and generate business analysis"""
        try:
            messages = state['messages']
            question = messages[-1].content if messages else ""
            doc_context = state.get('doc_context', {})
            sql_context = state.get('sql_context', {})
            
            # Check if we have any context documents or schemas
            has_context = (len(doc_context.get('results', [])) > 0 or 
                          len(sql_context.get('results', [])) > 0)
            
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
            
            # If no context is available, create a special response
            if not has_context:
                logger.warning("No context documents or schemas available for question processing")
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
                            "Upload documentation files for business context"
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
                sql_context=formatted_sql_context
            )
            response = llm.invoke(formatted_prompt)
            
            # Parse response with better error handling
            try:
                response_text = response.content if isinstance(response, BaseMessage) else str(response)
                
                # Clean the response text
                response_text = response_text.strip()
                
                # Log the raw response for debugging
                logger.debug(f"Raw LLM response: {response_text}")
                
                # Parse JSON
                try:
                    # First, try to extract a JSON block if the response contains markdown code blocks
                    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                    json_match = re.search(json_pattern, response_text)
                    
                    if json_match:
                        # Found a JSON code block, use its content
                        analysis = json.loads(json_match.group(1))
                    else:
                        # Check if the response starts with a JSON object
                        response_text = response_text.strip()
                        if response_text.startswith('{') and response_text.endswith('}'):
                            analysis = json.loads(response_text)
                        else:
                            # Try to clean up the response and parse again
                            cleaned_text = response_text
                            # Replace single quotes with double quotes for JSON
                            cleaned_text = re.sub(r"'([^']*)':", r'"\1":', cleaned_text)
                            # Remove newlines and extra whitespace
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                            # Try to extract anything that looks like JSON
                            json_obj_match = re.search(r'({.*})', cleaned_text)
                            if json_obj_match:
                                analysis = json.loads(json_obj_match.group(1))
                            else:
                                raise ValueError("Could not find valid JSON in the response")
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error: {je}\nAttempting to fix malformed JSON...")
                    # Try to fix common JSON formatting issues
                    response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes
                    response_text = re.sub(r'\n\s*', ' ', response_text)  # Remove newlines
                    response_text = re.sub(r',\s*}', '}', response_text)  # Remove trailing commas
                    response_text = re.sub(r',\s*]', ']', response_text)  # Remove trailing commas in arrays
                    
                    # Try one more time with the fixed JSON
                    try:
                        analysis = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If all else fails, create a minimal valid response
                        logger.error(f"Failed to parse JSON after fixing: {response_text[:100]}...")
                        analysis = {
                            "rephrased_question": question,
                            "question_type": "unknown",
                            "key_points": ["Error parsing response"],
                            "business_context": {
                                "domain": "Error recovery",
                                "primary_objective": "Failed to parse model response",
                                "key_entities": [],
                                "business_impact": "Unknown"
                            },
                            "technical_context": {
                                "data_stack": [],
                                "relevant_components": [],
                                "dependencies": [],
                                "technical_considerations": ["Error occurred during analysis"]
                            },
                            "implementation_guidance": {
                                "approach": "Please try again with a clearer question",
                                "suggested_steps": [],
                                "code_references": []
                            },
                            "assumptions": ["Processing error occurred"],
                            "clarifying_questions": ["Could you rephrase your question?"],
                            "confidence_score": 0.0
                        }
                
                # Create a new analysis with defaults
                validated_analysis = DEFAULT_ANALYSIS.copy()
                validated_analysis["rephrased_question"] = question
                
                # Validate and ensure required fields
                required_fields = {
                    "rephrased_question": str,
                    "question_type": str,
                    "key_points": list,
                    "business_context": dict,
                    "technical_context": dict,
                    "implementation_guidance": dict,
                    "assumptions": list,
                    "clarifying_questions": list,
                    "confidence_score": (int, float)
                }
                
                # Update with valid fields from the response
                for field, expected_type in required_fields.items():
                    if field in analysis and isinstance(analysis[field], expected_type):
                        if field in ["business_context", "technical_context", "implementation_guidance"]:
                            validated_analysis[field].update(analysis[field])
                        else:
                            validated_analysis[field] = analysis[field]
                
                return {
                    **state,
                    "business_analysis": validated_analysis,
                    "feedback_status": "pending",
                    "confidence_score": float(validated_analysis["confidence_score"])
                }
                
            except Exception as json_error:
                logger.error(f"Error parsing LLM response: {json_error}\nResponse text: {response_text}")
                error_analysis = DEFAULT_ANALYSIS.copy()
                error_analysis.update({
                    "rephrased_question": question,
                    "key_points": ["Error parsing analysis output"],
                    "business_context": {
                        "domain": "Data Analysis",
                        "primary_objective": "Market Share Analysis",
                        "key_entities": ["Nation", "Region", "Revenue"],
                        "business_impact": f"Error: {str(json_error)}"
                    }
                })
                return {
                    **state,
                    "business_analysis": error_analysis,
                    "feedback_status": "error",
                    "confidence_score": 0.0
                }
                
        except Exception as e:
            # Create a better error message for the user
            logger.error(f"Error in question processing: {e}")
            error_analysis = DEFAULT_ANALYSIS.copy()
            error_analysis.update({
                "rephrased_question": question,
                "question_type": "error",
                "key_points": ["An error occurred processing your question"],
                "business_context": {
                    "domain": "System Error",
                    "primary_objective": "Please try again with a clearer or simpler question",
                    "key_entities": [],
                    "business_impact": "Unable to process request"
                },
                "technical_context": {
                    "data_stack": [],
                    "relevant_components": [],
                    "dependencies": [],
                    "technical_considerations": ["Internal processing error occurred"]
                },
                "implementation_guidance": {
                    "approach": "Please try a different approach to your question",
                    "suggested_steps": [
                        "Simplify your question",
                        "Break complex questions into smaller parts",
                        "Check if you've uploaded relevant context documents"
                    ],
                    "code_references": []
                },
                "assumptions": [],
                "clarifying_questions": [
                    "Could you rephrase your question more simply?",
                    "Are you looking for business or technical information?"
                ],
                "confidence_score": 0.0
            })
            return {
                **state,
                "business_analysis": error_analysis,
                "feedback_status": "error",
                "confidence_score": 0.0
            }

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
            
            with self._lock:
                result = self.app.invoke(
                    {
                        "messages": [HumanMessage(content=question)],
                        "doc_context": doc_results,
                        "sql_context": sql_results,
                        "business_analysis": {},
                        "feedback_status": None,
                        "confidence_score": 0.0
                    },
                    {"configurable": {"thread_id": thread_id}}
                )
            
            # Save conversation to database
            business_analysis = result.get("business_analysis", {})
            response_data = {
                "query": question,
                "output": business_analysis.get("rephrased_question", "No response available"),
                "technical_details": json.dumps(business_analysis),
                "code_context": json.dumps({
                    "documentation": doc_results,
                    "sql_schema": sql_results
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
