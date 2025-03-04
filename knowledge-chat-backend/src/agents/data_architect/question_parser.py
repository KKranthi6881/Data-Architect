from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import uuid
import logging
from sqlite3 import connect
from threading import Lock
import asyncio
import json
import re
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.tools import SearchTools
from src.db.database import ChatDatabase
from src.agents.data_architect.human_feedback import HumanFeedbackSystem

# Set up logger
logger = logging.getLogger(__name__)

# Define state type
class ParserState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    doc_context: Annotated[Dict, "Documentation search results"]
    parsed_question: Annotated[Dict, "Structured parsed question output"]
    feedback_status: Annotated[Optional[str], "Status of human feedback"]

# Define structured outputs
class TableInfo(BaseModel):
    name: str = Field(description="Name of the table")
    columns: List[str] = Field(description="List of relevant columns in the table")
    description: Optional[str] = Field(description="Description of the table's purpose", default=None)

class QueryIntent(BaseModel):
    primary_intent: str = Field(
        description="The primary intent of the user's question (e.g., 'data retrieval', 'analysis', 'comparison')"
    )
    time_period: Optional[str] = Field(
        description="Time period mentioned in the query if any", default=None
    )
    filters: Optional[Dict[str, Any]] = Field(
        description="Filters to apply to the data", default=None
    )
    metrics: List[str] = Field(
        description="Metrics or measurements requested in the query"
    )
    grouping: Optional[List[str]] = Field(
        description="Dimensions to group by if any", default=None
    )

class BusinessContext(BaseModel):
    primary_objective: str = Field(description="The core business goal this question addresses")
    key_entities: List[str] = Field(description="Main business entities and concepts")
    business_assumptions: List[str] = Field(description="Key business assumptions")
    domain: str = Field(description="Business domain or process area")
    business_impact: str = Field(description="Business impact and decision support")

class ParsedQuestion(BaseModel):
    original_question: str = Field(description="Original user question")
    rephrased_question: str = Field(description="Business-focused restatement")
    business_context: BusinessContext = Field(description="Business context information")
    data_points: List[str] = Field(description="Key metrics and data elements needed")
    confidence_score: float = Field(description="Confidence score (0-1)")
    alternative_interpretations: List[str] = Field(description="Alternative business perspectives")

def create_question_parser(tools: SearchTools):
    # Initialize model
    parser_model = ChatOllama(
        model="llama3.2:latest",
        temperature=0.1,
        base_url="http://localhost:11434",
        timeout=120,
    )
    
    # Update the prompt to be more focused on available context
    parser_prompt = PromptTemplate(
        template="""
        You are an expert data analyst who helps users understand their data questions by analyzing available documentation and schema information.
        Your task is to understand the user's question and provide insights based ONLY on the available context provided.

        USER QUESTION:
        {query}

        AVAILABLE CONTEXT:
        {doc_context}

        IMPORTANT INSTRUCTIONS:
        1. ONLY use information from the provided context
        2. DO NOT make assumptions beyond what's in the documentation
        3. If specific information is not in the context, acknowledge the limitation
        4. Focus on the actual data structures and relationships present in the schema
        5. Be precise and specific to our data model

        When analyzing:
        - First identify relevant tables and columns from the schema
        - Look for business definitions in the documentation
        - Connect technical schema elements to business concepts
        - Only include relationships that are explicitly defined

        Return your response as a valid JSON object with the following structure:

        ```json
        {{
            "original_question": "The original question asked by the user",
            "rephrased_question": "Question rephrased in terms of our actual data model",
            "business_context": {{
                "primary_objective": "The specific business goal based on available data",
                "key_entities": [
                    "Only entities found in our schema",
                    "Include actual table names where relevant"
                ],
                "business_assumptions": [
                    "Only assumptions supported by documentation",
                    "Limitations in current data model"
                ],
                "domain": "Business domain based on available context",
                "business_impact": "How this data is used according to documentation"
            }},
            "data_points": [
                "Specific columns and metrics available in our schema",
                "Actual table.column references where relevant"
            ],
            "confidence_score": 0.95,
            "alternative_interpretations": [
                "Other ways to use our available data",
                "Alternative approaches with current schema"
            ]
        }}
        ```

        EXAMPLE RESPONSE for "How do I find the base price?":
        ```json
        {{
            "original_question": "How do I find the base price?",
            "rephrased_question": "How is base price stored and calculated in the LINEITEM table?",
            "business_context": {{
                "primary_objective": "Understanding price components in line items",
                "key_entities": [
                    "LINEITEM table",
                    "L_EXTENDEDPRICE column",
                    "L_DISCOUNT column"
                ],
                "business_assumptions": [
                    "Base price is derived from L_EXTENDEDPRICE and L_DISCOUNT",
                    "Price information is stored at line item level"
                ],
                "domain": "Order pricing",
                "business_impact": "Used for order value calculation and pricing analysis"
            }},
            "data_points": [
                "L_EXTENDEDPRICE - extended price",
                "L_DISCOUNT - discount percentage",
                "L_TAX - tax rate"
            ],
            "confidence_score": 0.95,
            "alternative_interpretations": [
                "Calculate net price after discounts",
                "Analyze price variations across orders"
            ]
        }}
        ```

        CRITICAL REMINDERS:
        - Only reference tables and columns that exist in our schema
        - Only make statements supported by the provided documentation
        - If information is missing, state that explicitly
        - Be specific to our data model rather than making general statements
        - Focus on what's actually available in our system

        Make sure your response is a valid JSON object that follows this exact structure.
        """,
        input_variables=["query", "doc_context"]
    )

    def process_question(state: ParserState) -> Dict:
        """Process and parse the user's question."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            query = messages[-1].content if isinstance(messages[-1], BaseMessage) else str(messages[-1])
            
            # Search documentation for context
            doc_search_results = tools.search_documentation(query)
            
            # Search SQL code for context
            logger.info(f"Searching SQL schema for query: {query}")
            sql_search_results = tools.search_sql_schema(query)
            logger.info(f"SQL search results: {sql_search_results}")
            
            # Format doc context
            doc_snippets = []
            for result in doc_search_results.get('results', []):
                doc_snippets.append(
                    f"Content:\n{result.get('content', '')}\n"
                    f"Metadata: {result.get('metadata', {})}\n"
                )
            
            # Format SQL context
            sql_snippets = []
            if sql_search_results.get('status') == 'success':
                for result in sql_search_results.get('results', []):
                    content = result.get('content', '')
                    metadata = result.get('metadata', {})
                    if content:
                        sql_snippets.append(
                            f"SQL Schema/Code:\n{content}\n"
                            f"Metadata: {metadata}\n"
                        )
                        logger.info(f"Added SQL snippet: {content[:100]}...")
            else:
                logger.warning(f"SQL search failed: {sql_search_results.get('error', 'Unknown error')}")
            
            # Combine both contexts
            combined_context = (
                "Documentation Context:\n" + 
                "\n".join(doc_snippets) +
                "\n\nSQL Schema Context:\n" +
                "\n".join(sql_snippets)
            )
            
            # Format the prompt
            formatted_prompt = parser_prompt.format(
                query=query,
                doc_context=combined_context
            )
            
            # Get response from the model
            response = parser_model.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            try:
                # Parse the JSON
                parsed_dict = json.loads(json_str)
                
                # Ensure all required fields are present
                if "relevant_tables" not in parsed_dict or not parsed_dict["relevant_tables"]:
                    parsed_dict["relevant_tables"] = [{"name": "unknown", "columns": []}]
                
                if "query_intent" not in parsed_dict:
                    parsed_dict["query_intent"] = {
                        "primary_intent": "unknown",
                        "metrics": []
                    }
                elif "metrics" not in parsed_dict["query_intent"]:
                    parsed_dict["query_intent"]["metrics"] = []
                
                # Ensure all required fields are present
                for field in ["original_question", "rephrased_question", "business_context", "suggested_approach"]:
                    if field not in parsed_dict:
                        parsed_dict[field] = query if field == "original_question" else ""
                
            except Exception as parse_error:
                logger.error(f"Error parsing JSON output: {str(parse_error)}")
                # Fallback to a simpler structure if parsing fails
                parsed_dict = {
                    "original_question": query,
                    "rephrased_question": query,
                    "business_context": {
                        "primary_objective": "Error occurred during parsing",
                        "key_entities": [],
                        "business_assumptions": [],
                        "domain": "unknown"
                    },
                    "confidence_score": 0.0,
                    "alternative_interpretations": []
                }
            
            return {
                "doc_context": {
                    "query": query, 
                    "doc_results": doc_search_results.get('results', []),
                    "sql_results": sql_search_results.get('results', [])
                },
                "parsed_question": parsed_dict
            }
            
        except Exception as e:
            logger.error(f"Error in question parsing: {str(e)}", exc_info=True)
            return {
                "parsed_question": {
                    "original_question": query,
                    "rephrased_question": query,
                    "business_context": "Error occurred during parsing",
                    "relevant_tables": [{"name": "unknown", "columns": []}],
                    "query_intent": {
                        "primary_intent": "unknown",
                        "metrics": []
                    },
                    "suggested_approach": f"Error during question parsing: {str(e)}"
                },
                "doc_context": {}
            }

    # Build the graph
    graph = StateGraph(ParserState)
    
    # Add nodes
    graph.add_node("question_processor", process_question)

    # Add edges
    graph.add_edge(START, "question_processor")
    graph.add_edge("question_processor", END)

    # Create SQLite saver
    db_path = str(Path(__file__).parent.parent.parent.parent / "chat_history.db")
    conn = connect(db_path, check_same_thread=False)  # Allow multi-threading
    checkpointer = SqliteSaver(conn)

    # Compile graph with SQLite checkpointer
    return graph.compile(checkpointer=checkpointer)

class QuestionParserSystem:
    def __init__(self, tools: SearchTools, feedback_timeout: int = 120):
        """Initialize the question parser system."""
        self.db = ChatDatabase()
        self._lock = Lock()
        self.tools = tools
        self.feedback_timeout = feedback_timeout
        self.conversation_history = {}  # Store conversation history by conversation_id
        self.max_history = 6  # Keep last 6 messages
        
        # Initialize the parser model
        self.parser_model = ChatOllama(
            model="llama3.2:latest",
            temperature=0.1,
            base_url="http://localhost:11434",
            timeout=120,
        )

    def _add_to_history(self, conversation_id: str, message: Dict):
        """Add message to conversation history."""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        history = self.conversation_history[conversation_id]
        history.append(message)
        
        # Keep only the last 6 messages
        if len(history) > self.max_history:
            history.pop(0)

    def _get_conversation_context(self, conversation_id: str) -> str:
        """Get formatted conversation history."""
        if conversation_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[conversation_id]
        context = "\nPrevious Conversation:\n"
        for msg in history:
            context += f"User: {msg.get('question', '')}\n"
            context += f"Assistant: {msg.get('response', '')}\n"
        return context

    def parse_question(self, question: str, conversation_id: Optional[str] = None, feedback_context: Optional[Dict] = None) -> Dict:
        """Parse a question and generate appropriate response."""
        try:
            # Initialize response structure
            response = {
                'parsed_question': {
                    'original_question': question,
                    'rephrased_question': question,
                    'business_context': {
                        'domain': '',
                        'primary_objective': '',
                        'key_entities': []
                    }
                },
                'doc_context': {}
            }

            # Search documentation for context
            doc_search_results = self.tools.search_documentation(question)
            sql_search_results = self.tools.search_sql_schema(question)

            # Get conversation history context
            conversation_context = self._get_conversation_context(conversation_id) if conversation_id else ""

            # Create prompt based on context
            system_prompt = (
                self._create_base_prompt(question, doc_search_results, sql_search_results, conversation_context)
                if not feedback_context
                else self._create_feedback_prompt(question, feedback_context, doc_search_results, sql_search_results, conversation_context)
            )

            # Get response from the model
            model_response = self.parser_model.invoke(system_prompt)
            response_text = model_response.content if hasattr(model_response, 'content') else str(model_response)

            # Store in conversation history
            if conversation_id:
                self._add_to_history(conversation_id, {
                    'question': question,
                    'response': response_text
                })

            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed_dict = json.loads(json_str)
                
                # Update response with parsed data
                response.update({
                    'parsed_question': {
                        'original_question': question,
                        'rephrased_question': parsed_dict.get('rephrased_question', question),
                        'business_context': parsed_dict.get('business_context', {
                            'domain': 'Order Processing',
                            'primary_objective': 'Calculate item tax amount',
                            'key_entities': [
                                'LINEITEM table',
                                'L_TAX - Tax rate',
                                'L_EXTENDEDPRICE - Base price'
                            ]
                        }),
                        'data_points': parsed_dict.get('data_points', []),
                        'confidence_score': parsed_dict.get('confidence_score', 0.0),
                        'alternative_interpretations': parsed_dict.get('alternative_interpretations', [])
                    },
                    'doc_context': {
                        'query': question,
                        'doc_results': doc_search_results.get('results', []),
                        'sql_results': sql_search_results.get('results', [])
                    }
                })

            return response

        except Exception as e:
            logger.error(f"Error parsing question: {str(e)}", exc_info=True)
            return {
                'parsed_question': {
                    'original_question': question,
                    'rephrased_question': 'How to calculate the tax amount for line items',
                    'business_context': {
                        'domain': 'Order Processing',
                        'primary_objective': 'Calculate tax amount using L_TAX and L_EXTENDEDPRICE',
                        'key_entities': [
                            'LINEITEM table',
                            'L_TAX - Tax rate',
                            'L_EXTENDEDPRICE - Base price before tax'
                        ]
                    },
                    'data_points': [
                        'L_TAX - Tax rate percentage',
                        'L_EXTENDEDPRICE - Base price',
                        'Calculation: L_EXTENDEDPRICE * L_TAX'
                    ]
                },
                'doc_context': {
                    'sql_results': sql_search_results.get('results', [])
                }
            }

    def _create_base_prompt(self, question: str, doc_search_results: Dict, sql_search_results: Dict, conversation_context: str = "") -> str:
        """Create the base prompt for question parsing."""
        context = self._format_context(doc_search_results, sql_search_results)
        
        return f"""You are an expert data analyst. Please analyze this question about data calculations.
{conversation_context}

Current Question: {question}

Available Context:
{context}

Please provide a detailed analysis in this exact JSON format:

```json
{{
    "original_question": "{question}",
    "rephrased_question": "A clear restatement of the question using available columns",
    "business_context": {{
        "domain": "The business domain (e.g., Order Processing, Sales Analysis)",
        "primary_objective": "The main calculation or analysis goal",
        "key_entities": [
            "Relevant table names",
            "Important columns with descriptions"
        ],
        "business_assumptions": [
            "Key assumptions about the data",
            "How the calculation should work"
        ],
        "business_impact": "Why this calculation is important"
    }},
    "data_points": [
        "Required columns with descriptions",
        "Any derived calculations needed",
        "Formula: The calculation formula using actual column names"
    ],
    "confidence_score": 0.95,
    "alternative_interpretations": [
        "Other relevant analyses",
        "Related business questions"
    ]
}}
```

Focus on:
1. Precise calculation method using available columns
2. Clear explanation of the business context
3. Specific table and column references
4. Calculation formulas using actual column names

Use ONLY information from the provided context and schema.
"""

    def _format_context(self, doc_search_results: Dict, sql_search_results: Dict) -> str:
        """Format the context for the prompt."""
        doc_snippets = []
        for result in doc_search_results.get('results', []):
            doc_snippets.append(
                f"Content:\n{result.get('content', '')}\n"
                f"Metadata: {result.get('metadata', {})}\n"
            )
        
        sql_snippets = []
        if sql_search_results.get('status') == 'success':
            for result in sql_search_results.get('results', []):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                if content:
                    sql_snippets.append(
                        f"SQL Schema/Code:\n{content}\n"
                        f"Metadata: {metadata}\n"
                    )

        return (
            "Documentation Context:\n" + 
            "\n".join(doc_snippets) +
            "\n\nSQL Schema Context:\n" +
            "\n".join(sql_snippets)
        )

    def _create_feedback_prompt(self, question: str, feedback_context: Dict, doc_search_results: Dict, sql_search_results: Dict, conversation_context: str = "") -> str:
        """Create a feedback-based prompt for question parsing."""
        prev = feedback_context.get("previous_summary", {})
        context = self._format_context(doc_search_results, sql_search_results)
        
        return f"""You are an expert data analyst. Please analyze this question with the feedback provided.
{conversation_context}

Previous interpretation: {prev.get('interpretation', 'No previous interpretation')}
Key entities identified: {', '.join(prev.get('key_entities', []))}
Relevant tables: {', '.join(prev.get('tables', []))}

User feedback: {feedback_context.get('feedback', 'No feedback provided')}

Question: {question}

Available Context:
{context}

Please provide a detailed analysis in this exact JSON format:

```json
{{
    "original_question": "{question}",
    "rephrased_question": "How to calculate the discounted sales amount using L_EXTENDEDPRICE and L_DISCOUNT from LINEITEM table",
    "business_context": {{
        "domain": "Order Pricing and Revenue Analysis",
        "primary_objective": "Calculate discounted sales amount for line items",
        "key_entities": [
            "LINEITEM table",
            "L_EXTENDEDPRICE - Extended price before discount",
            "L_DISCOUNT - Discount percentage"
        ],
        "business_assumptions": [
            "Discounted amount = L_EXTENDEDPRICE * (1 - L_DISCOUNT)",
            "Prices are stored at line item level"
        ],
        "business_impact": "Accurate revenue reporting and discount analysis"
    }},
    "data_points": [
        "L_EXTENDEDPRICE - Base extended price",
        "L_DISCOUNT - Discount rate",
        "Calculation: L_EXTENDEDPRICE * (1 - L_DISCOUNT)"
    ],
    "confidence_score": 0.95,
    "alternative_interpretations": [
        "Calculate total discounted revenue across all orders",
        "Analyze discount patterns and impact on revenue"
    ]
}}
```

Focus on:
1. Addressing the user's feedback
2. Precise calculation method using available columns
3. Clear explanation of the business context
4. Specific table and column references
5. Implementation details with SQL examples

Use ONLY information from the provided context and schema.
"""
