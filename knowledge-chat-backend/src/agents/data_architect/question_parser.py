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

from src.tools import SearchTools
from src.db.database import ChatDatabase

logger = logging.getLogger(__name__)

# Define default analysis structure at module level
DEFAULT_ANALYSIS = {
    "rephrased_question": "",
    "key_points": [],
    "business_context": {
        "domain": "Unknown",
        "primary_objective": "Not specified",
        "key_entities": [],
        "business_impact": "Not analyzed"
    },
    "assumptions": [],
    "clarifying_questions": [],
    "confidence_score": 0.0
}

# Define state type
class ParserState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    doc_context: Annotated[Dict, "Documentation context"]
    business_analysis: Annotated[Dict, "Business analysis results"]
    feedback_status: Annotated[str, "Current status"]
    confidence_score: Annotated[float, "Confidence in analysis"]

# Define structured outputs
class BusinessAnalysis(BaseModel):
    rephrased_question: str = Field(description="Clear business-focused restatement")
    key_points: List[str] = Field(description="Key business points and objectives")
    business_context: Dict[str, Any] = Field(description="Business context information")
    assumptions: List[str] = Field(description="Business assumptions to verify")
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
    template = """You are an expert business analyst helping users understand their data requirements.

USER QUESTION:
{question}

BUSINESS DOCUMENTATION:
{context}

TASK:
Analyze this question using our business documentation and provide a structured understanding.

GUIDELINES:
- Focus on business objectives and requirements
- Only reference information found in our documentation
- Identify any missing business context
- Keep technical details for later analysis

IMPORTANT: Your response must be a valid JSON object with the following structure (no markdown, no explanations, just the JSON):
{{
    "rephrased_question": "Clear business-focused restatement of the question",
    "key_points": [
        "Key business point 1",
        "Key business point 2"
    ],
    "business_context": {{
        "domain": "Business domain area",
        "primary_objective": "Main business goal",
        "key_entities": ["Entity 1", "Entity 2"],
        "business_impact": "How this affects business"
    }},
    "assumptions": [
        "Business assumption 1",
        "Business assumption 2"
    ],
    "clarifying_questions": [
        "Question about requirement 1",
        "Question about requirement 2"
    ],
    "confidence_score": 0.85
}}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "context"]
    )

    def process_question(state: ParserState) -> Dict:
        """Process the question and generate business analysis"""
        try:
            messages = state['messages']
            question = messages[-1].content if messages else ""
            context = state.get('doc_context', {})
            
            # Format context
            formatted_context = "\n".join([
                f"[{doc.get('type', 'Doc')}]\n{doc.get('content', '')}\n---"
                for doc in context.get('results', [])
            ])
            
            # Generate analysis
            formatted_prompt = prompt.format(
                question=question,
                context=formatted_context
            )
            response = llm.invoke(formatted_prompt)
            
            # Parse response with better error handling
            try:
                response_text = response.content if isinstance(response, BaseMessage) else str(response)
                
                # Clean the response text
                response_text = response_text.strip()
                
                # Log the raw response for debugging
                logger.debug(f"Raw LLM response: {response_text}")
                
                # Extract JSON if wrapped in any markers
                if '```' in response_text:
                    # Find the last occurrence of ``` before the JSON
                    start = response_text.rfind('```') + 3
                    # Find the next occurrence of ``` after the JSON
                    end = response_text.find('```', start)
                    if end == -1:  # If no closing ```, take the rest of the text
                        response_text = response_text[start:].strip()
                    else:
                        response_text = response_text[start:end].strip()
                
                # Remove any language identifier if present
                if response_text.startswith('json'):
                    response_text = response_text[4:].strip()
                
                # Remove any leading/trailing whitespace or quotes
                response_text = response_text.strip('`\'" \n\t')
                
                # Log cleaned text for debugging
                logger.debug(f"Cleaned response text: {response_text}")
                
                # Parse JSON
                try:
                    analysis = json.loads(response_text)
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error: {je}\nAttempting to fix malformed JSON...")
                    # Try to fix common JSON formatting issues
                    response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes
                    response_text = response_text.replace('\n', '')  # Remove newlines
                    analysis = json.loads(response_text)
                
                # Create a new analysis with defaults
                validated_analysis = DEFAULT_ANALYSIS.copy()
                validated_analysis["rephrased_question"] = question
                
                # Validate and ensure required fields
                required_fields = {
                    "rephrased_question": str,
                    "key_points": list,
                    "business_context": dict,
                    "assumptions": list,
                    "clarifying_questions": list,
                    "confidence_score": (int, float)
                }
                
                # Update with valid fields from the response
                for field, expected_type in required_fields.items():
                    if field in analysis and isinstance(analysis[field], expected_type):
                        if field == "business_context":
                            validated_analysis["business_context"].update(analysis["business_context"])
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
            logger.error(f"Error in question processing: {e}")
            error_analysis = DEFAULT_ANALYSIS.copy()
            error_analysis.update({
                "rephrased_question": "Error analyzing question",
                "key_points": ["Unable to analyze business requirements"],
                "business_context": {
                    "domain": "Error",
                    "primary_objective": str(e),
                    "key_entities": [],
                    "business_impact": "Analysis failed"
                }
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

    return graph.compile(checkpointer=checkpointer)

class QuestionParserSystem:
    def __init__(self, tools: SearchTools):
        self.app = create_parser_agent(tools)
        self.tools = tools
        self.db = ChatDatabase()
        self._lock = Lock()

    async def parse_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the parser system"""
        try:
            # Generate unique ID
            conversation_id = str(uuid.uuid4())
            
            # Get documentation context
            doc_results = self.tools.search_documentation(question)
            
            with self._lock:
                result = self.app.invoke(
                    {
                        "messages": [HumanMessage(content=question)],
                        "doc_context": doc_results,
                        "business_analysis": {},
                        "feedback_status": None,
                        "confidence_score": 0.0
                    },
                    {"configurable": {"thread_id": conversation_id}}
                )
            
            return result.get("business_analysis", {})
            
        except Exception as e:
            logger.error(f"Error in parser system: {e}")
            return {
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
                "confidence_score": 0.0
            }
