import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from src.db.database import ChatDatabase
from src.agents.data_architect.schema_search import SchemaSearchAgent
from src.agents.data_architect.github_search import GitHubCodeSearchAgent
from src.agents.data_architect.question_parser import QuestionParserSystem

# Set up logger
logger = logging.getLogger(__name__)

class DataArchitectAgent:
    """
    Data Architect Agent that provides dbt and Snowflake specific solutions
    based on schema and code analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOllama(
            model="deepseek-r1:8b",
            temperature=0.1,  # Lower temperature for more focused responses
            base_url="http://localhost:11434",
            timeout=180,
        )

    def generate_response(self, parsed_question: Dict[str, Any], 
                         schema_results: List[Dict[str, Any]], 
                         code_results: List[Dict[str, Any]], 
                         original_question: str) -> Dict[str, Any]:
        """Generate a dbt and Snowflake focused solution"""
        try:
            self.logger.info(f"Analyzing request: {original_question[:50]}...")
            
            # Analyze available context
            context_analysis = self._analyze_context(schema_results, code_results)
            if not context_analysis["has_sufficient_context"]:
                return self._generate_context_request()

            # Create focused prompt
            prompt = self._create_architect_prompt(
                original_question=original_question,
                parsed_question=parsed_question,
                schema_info=context_analysis["schema_summary"],
                code_info=context_analysis["code_summary"]
            )

            # Generate response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Process and structure the response
            processed_response = self._process_response(
                response_text=response_text,
                context_analysis=context_analysis
            )

            return processed_response
            
        except Exception as e:
            self.logger.error(f"Error in architect analysis: {e}", exc_info=True)
            return self._generate_error_response(str(e))

    def _analyze_context(self, schema_results: List[Dict], code_results: List[Dict]) -> Dict:
        """Analyze available context and create summaries"""
        schema_info = []
        code_info = []
        
        # Process schema information
        for schema in schema_results:
            if schema.get("relevance_score", 0) > 0.5:  # Only include relevant schemas
                schema_info.append({
                    "name": f"{schema.get('schema_name', '')}.{schema.get('table_name', '')}",
                    "columns": schema.get("columns", []),
                    "description": schema.get("explanation", ""),
                    "relevance": schema.get("relevance_score", 0)
                })

        # Process code examples
        for code in code_results:
            if code.get("relevance_score", 0) > 0.5:  # Only include relevant code
                code_info.append({
                    "file": code.get("file_path", ""),
                    "snippet": code.get("code_snippet", ""),
                    "context": code.get("explanation", ""),
                    "relevance": code.get("relevance_score", 0)
                })

        return {
            "has_sufficient_context": bool(schema_info or code_info),
            "schema_summary": self._format_schema_summary(schema_info),
            "code_summary": self._format_code_summary(code_info),
            "available_schemas": schema_info,
            "available_code": code_info
        }

    def _create_architect_prompt(self, original_question: str, 
                               parsed_question: Dict, 
                               schema_info: str, 
                               code_info: str) -> str:
        """Create a focused dbt and Snowflake solution prompt"""
        return """
You are a senior data architect specializing in dbt and Snowflake development.
Your task is to provide a detailed, step-by-step solution focusing on dbt best practices and Snowflake optimization.

BUSINESS QUESTION:
{original_question}

BUSINESS CONTEXT:
{business_context}

AVAILABLE DATA STRUCTURES:
{schema_info}

RELEVANT CODE EXAMPLES:
{code_info}

Provide a complete solution with the following detailed sections:

## Business Requirements
- Detailed business requirements analysis
- Specific metrics and KPIs to be calculated
- Data granularity and freshness requirements
- Expected output format and use cases
- Data quality expectations
- Any business rules or transformations needed

## Source Data Analysis
- Review of source tables and their relationships
- Data quality assessment
- Known data limitations or gaps
- Required data transformations
- Source freshness requirements

## Data Model Design
1. Model Architecture:
   - Model type (staging, intermediate, mart)
   - Materialization strategy with rationale
   - Incremental strategy if applicable
   - Dependencies and DAG structure

2. Table Design:
   ```sql
   -- Column definitions with types
   column_name_1 STRING COMMENT 'Detailed column description',
   column_name_2 TIMESTAMP COMMENT 'Format and business context'
   ```

3. Primary/Foreign Keys:
   - Identify unique keys
   - Define relationships
   - Handle duplicates strategy

## Implementation Guide
1. dbt Model Creation:
   ```sql
   -- Complete SQL implementation
   WITH source_data AS (
       -- Source CTE
   ),
   transformed AS (
       -- Transformation logic
   )
   
   SELECT 
       -- Final output columns
   FROM transformed
   ```

2. dbt Configuration:
   ```yaml
   version: 2
   
   models:
     - name: model_name
       description: "Detailed model description"
       config:
         materialized: table|incremental|view
         schema: target_schema
         tags: [tag1, tag2]
         snowflake_warehouse: compute_wh
       
       columns:
         - name: column_name
           description: "Detailed description"
           tests:
             - unique
             - not_null
             - relationships:
                 to: ref('parent_model')
                 field: id
   
       tests:
         - dbt_utils.equal_rowcount:
             compare_model: ref('source_model')
   ```

## Testing Strategy
1. Data Quality Tests:
   ```yaml
   # Generic tests
   - unique
   - not_null
   - relationships
   - accepted_values
   
   # Custom tests
   - test_name:
       config:
         severity: warn|error
         threshold: value
   ```

2. Business Logic Tests:
   - Row count validations
   - Sum amount checks
   - Date range tests
   - Custom business rule tests

3. Reconciliation Tests:
   - Source to target count matches
   - Sum of metrics matches
   - Sample record comparisons

## Performance Optimization
1. Snowflake Specific:
   - Clustering keys selection
   - Partition strategy
   - Warehouse sizing recommendations

2. dbt Optimization:
   - Materialization strategy rationale
   - Incremental strategy details
   - Dependencies optimization

3. Query Optimization:
   - Join optimization
   - Filter push-down
   - Proper indexing

## Deployment Guide
1. Development Steps:
   - Initial model creation
   - Test implementation
   - Documentation updates

2. Testing Process:
   - Unit testing approach
   - Integration testing steps
   - Performance testing

3. Production Deployment:
   - Deployment checklist
   - Rollback plan
   - Monitoring setup

## Documentation
1. Model Documentation:
   ```yaml
   description: >
     Detailed model description including:
     - Business context
     - Update frequency
     - Dependencies
     - Known limitations
     - Usage examples
   ```

2. Column Documentation:
   ```yaml
   columns:
     - name: column_name
       description: >
         Detailed description including:
         - Business definition
         - Calculation logic
         - Valid values
         - Example values
   ```

Ensure all code is complete and ready to implement. Include specific dbt and Snowflake best practices.
If any information is missing, clearly state what additional context is needed.
""".format(
            original_question=original_question,
            business_context=json.dumps(parsed_question.get("business_context", {}), indent=2),
            schema_info=schema_info,
            code_info=code_info
        )

    def _process_response(self, response_text: str, context_analysis: Dict) -> Dict:
        """Process and structure the architect response"""
        sections = self._extract_enhanced_sections(response_text)
        
        # Extract implementation details
        implementation = self._extract_code_blocks(sections.get("implementation_guide", ""))
        dbt_config = self._extract_code_blocks(sections.get("dbt_configuration", ""))
        
        return {
            "response": response_text,
            "sections": sections,
            "implementation": {
                "sql": implementation.get("sql", []),
                "yaml": dbt_config.get("yaml", []),
                "tests": sections.get("testing_strategy", ""),
                "performance": sections.get("performance_optimization", "")
            },
            "context": {
                "has_schema_data": bool(context_analysis["available_schemas"]),
                "has_code_examples": bool(context_analysis["available_code"]),
                "schemas_used": [s["name"] for s in context_analysis["available_schemas"]],
                "code_references": [c["file"] for c in context_analysis["available_code"]]
            }
        }

    def _extract_code_blocks(self, text: str) -> Dict[str, List[str]]:
        """Extract code blocks by type from text"""
        code_blocks = {
            "sql": [],
            "yaml": [],
            "python": []
        }
        
        pattern = r"```(\w+)?\s*(.*?)```"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            lang = match.group(1) or "sql"
            code = match.group(2).strip()
            if lang.lower() in code_blocks:
                code_blocks[lang.lower()].append(code)
        
        return code_blocks

    def _extract_enhanced_sections(self, response_text: str) -> Dict[str, str]:
        """Extract sections from the response text using markdown headers"""
        try:
            sections = {}
            current_section = "overview"
            current_content = []
            
            # Split text into lines and process
            lines = response_text.split('\n')
            for line in lines:
                # Check for section headers (## Header)
                if line.strip().startswith('##'):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    # Start new section
                    current_section = line.strip('#').strip().lower().replace(' ', '_')
                    current_content = []
                else:
                    current_content.append(line)
            
            # Save the last section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # If no sections were found, put everything in overview
            if not sections:
                sections["overview"] = response_text.strip()
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error extracting sections: {e}", exc_info=True)
            return {"error": str(e), "overview": response_text}

    def _generate_context_request(self) -> Dict[str, Any]:
        """Generate a request for more context"""
        return {
            "response": """
I apologize, but I don't have enough context about your data models or existing code to provide specific recommendations. To help you better, please provide:

1. Your existing dbt model definitions
2. Snowflake table schemas
3. Any current SQL transformations you're using

This will allow me to give you targeted, actionable advice for your specific use case.
            """,
            "sections": {
                "limitations": "No schema or code context available",
                "next_steps": "Please provide dbt models and schema information"
            }
        }

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate an error response"""
        return {
            "response": f"Error generating response: {error_message}",
            "sections": {
                "error": error_message,
                "status": "failed"
            }
        }

    def _format_schema_summary(self, schema_info: List[Dict]) -> str:
        """Format schema summary for prompt"""
        return "\n".join([
            f"Table: {schema['name']}\n"
            f"Columns: {', '.join(schema['columns'])}\n"
            f"Description: {schema['description']}\n"
            f"Relevance: {schema['relevance']*10:.1f}/10"
            for schema in schema_info
        ])

    def _format_code_summary(self, code_info: List[Dict]) -> str:
        """Format code summary for prompt"""
        return "\n".join([
            f"File: {code['file']}\n"
            f"Snippet: {code['snippet']}\n"
            f"Context: {code['context']}\n"
            f"Relevance: {code['relevance']*10:.1f}/10"
            for code in code_info
        ]) 