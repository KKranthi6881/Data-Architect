from typing import Dict, List, Any, Optional, Set
import sqlglot
from sqlglot import parse_one, exp
import ast
import logging
from pathlib import Path
import re
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    description: str
    is_nullable: bool
    constraints: List[str]
    business_terms: List[str]
    sample_values: List[str]
    validation_rules: List[str]

@dataclass
class TableMetadata:
    name: str
    description: str
    columns: Dict[str, ColumnMetadata]
    primary_key: List[str]
    foreign_keys: List[Dict[str, str]]
    update_frequency: str
    business_domain: str
    data_owners: List[str]
    sample_size: Optional[int]

class CodeAnalyzer:
    def __init__(self):
        """Initialize the code analyzer."""
        self.supported_sql_dialects = ['sqlite', 'postgres', 'mysql']
        self.business_glossary = self._initialize_business_glossary()
        self.table_aliases = {}  # Track table aliases
        self.column_references = defaultdict(set)  # Track column references
        
    def _initialize_business_glossary(self) -> Dict[str, str]:
        """Initialize business term mappings."""
        return {
            'revenue': 'Total monetary value of sales',
            'churn': 'Customer discontinuation rate',
            'mrr': 'Monthly Recurring Revenue',
            'arr': 'Annual Recurring Revenue',
            'cac': 'Customer Acquisition Cost',
            'ltv': 'Lifetime Value'
        }

    def analyze_sql(self, content: str) -> Dict[str, Any]:
        """Analyze SQL code and extract metadata."""
        try:
            analysis = {
                'tables': set(),
                'columns': defaultdict(dict),
                'relationships': [],
                'column_lineage': [],
                'transformations': [],
                'business_context': [],
                'query_patterns': []
            }
            
            # Reset tracking dictionaries for each analysis
            self.table_aliases.clear()
            self.column_references.clear()
            
            # Extract business context from comments with improved parsing
            analysis['business_context'] = self._extract_business_context(content)
            
            # Parse SQL with different dialects
            parsed_statements = None
            for dialect in self.supported_sql_dialects:
                try:
                    parsed_statements = sqlglot.parse(content, dialect=dialect)
                    break
                except Exception as e:
                    logger.debug(f"Failed parsing with {dialect}: {str(e)}")
                    continue
            
            if not parsed_statements:
                logger.error("Failed to parse SQL with any dialect")
                return analysis
            
            # First pass: collect table aliases and references
            self._collect_table_info(parsed_statements)
            
            # Second pass: analyze relationships and lineage
            for statement in parsed_statements:
                self._analyze_statement(statement, analysis)
            
            # Post-process relationships to remove duplicates and resolve aliases
            analysis['relationships'] = self._deduplicate_relationships(analysis['relationships'])
            analysis['column_lineage'] = self._resolve_column_lineage(analysis['column_lineage'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in SQL analysis: {str(e)}")
            return {"error": str(e)}

    def _collect_table_info(self, statements):
        """First pass to collect table aliases and references."""
        for statement in statements:
            # Collect table aliases
            for table in statement.find_all(exp.Table):
                if hasattr(table, 'alias'):
                    self.table_aliases[table.alias] = table.name
                
            # Collect column references
            for col in statement.find_all(exp.Column):
                if hasattr(col, 'table') and hasattr(col, 'name'):
                    self.column_references[col.table].add(col.name)

    def _analyze_statement(self, statement, analysis):
        """Analyze a single SQL statement comprehensively."""
        try:
            # Extract tables
            for table in statement.find_all(exp.Table):
                table_name = self._resolve_table_name(table)
                analysis['tables'].add(table_name)
                
                # Extract columns for this table
                if hasattr(table, 'columns'):
                    for col in table.columns:
                        analysis['columns'][table_name][col.name] = {
                            'data_type': getattr(col, 'type', None),
                            'constraints': self._extract_column_constraints(col)
                        }
            
            # Analyze joins with improved relationship detection
            if isinstance(statement, exp.Select):
                self._analyze_select_statement(statement, analysis)
                
        except Exception as e:
            logger.error(f"Error analyzing statement: {str(e)}")

    def _analyze_select_statement(self, statement, analysis):
        """Analyze a SELECT statement comprehensively."""
        try:
            # Analyze explicit joins
            for join in statement.find_all(exp.Join):
                relationship = self._analyze_join(join)
                if relationship:
                    analysis['relationships'].append(relationship)
            
            # Analyze columns and their lineage
            self._analyze_columns(statement, analysis)
            
            # Analyze WHERE clause for relationships and filters
            where_clause = statement.find(exp.Where)
            if where_clause:
                # Get implicit joins
                implicit_joins = self._analyze_where_clause_relationships(where_clause)
                analysis['relationships'].extend(implicit_joins)
                
                # Get filters
                filters = self._analyze_filters(where_clause)
                if filters:
                    if 'filters' not in analysis:
                        analysis['filters'] = []
                    analysis['filters'].extend(filters)
            
            # Analyze GROUP BY
            group_by = statement.find(exp.Group)
            if group_by:
                if 'aggregations' not in analysis:
                    analysis['aggregations'] = []
                analysis['aggregations'].append({
                    'type': 'group_by',
                    'columns': [str(col) for col in group_by.expressions]
                })
            
            # Analyze ORDER BY
            order_by = statement.find(exp.Order)
            if order_by:
                if 'sort_criteria' not in analysis:
                    analysis['sort_criteria'] = []
                analysis['sort_criteria'].append({
                    'columns': [str(col) for col in order_by.expressions],
                    'direction': str(order_by.direction) if hasattr(order_by, 'direction') else 'ASC'
                })
            
        except Exception as e:
            logger.error(f"Error analyzing SELECT statement: {str(e)}")

    def _analyze_join(self, join) -> Optional[Dict[str, Any]]:
        """Enhanced join analysis with alias resolution."""
        try:
            if not hasattr(join, 'on'):
                return None
            
            on_clause = join.on
            if hasattr(on_clause, 'left') and hasattr(on_clause, 'right'):
                left = on_clause.left
                right = on_clause.right
                
                left_table = self._resolve_table_name(left.table) if hasattr(left, 'table') else None
                right_table = self._resolve_table_name(right.table) if hasattr(right, 'table') else None
                
                if left_table and right_table:
                    return {
                        'type': 'join',
                        'join_type': join.__class__.__name__.lower(),
                        'left_table': left_table,
                        'left_column': left.name if hasattr(left, 'name') else str(left),
                        'right_table': right_table,
                        'right_column': right.name if hasattr(right, 'name') else str(right),
                        'condition': str(on_clause)
                    }
            return None
        except Exception as e:
            logger.error(f"Error analyzing join: {str(e)}")
            return None

    def _analyze_where_clause_relationships(self, where_clause) -> List[Dict[str, Any]]:
        """Analyze WHERE clause for implicit joins."""
        relationships = []
        try:
            conditions = where_clause.find_all(exp.Binary)
            for condition in conditions:
                if hasattr(condition, 'left') and hasattr(condition, 'right'):
                    left = condition.left
                    right = condition.right
                    
                    # Check if condition represents a relationship
                    if (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                        relationship = {
                            'type': 'implicit_join',
                            'left_table': self._resolve_table_name(left.table) if hasattr(left, 'table') else None,
                            'left_column': left.name if hasattr(left, 'name') else str(left),
                            'right_table': self._resolve_table_name(right.table) if hasattr(right, 'table') else None,
                            'right_column': right.name if hasattr(right, 'name') else str(right),
                            'condition': str(condition)
                        }
                        if relationship['left_table'] and relationship['right_table']:
                            relationships.append(relationship)
        except Exception as e:
            logger.error(f"Error analyzing where clause: {str(e)}")
        
        return relationships

    def _analyze_columns(self, statement, analysis: Dict) -> None:
        """Analyze columns in SELECT statement for lineage and transformations."""
        try:
            # Get all column expressions from SELECT
            select_expressions = []
            if hasattr(statement, 'expressions'):
                select_expressions = statement.expressions
            elif hasattr(statement, 'selects'):
                select_expressions = statement.selects
            
            for expr in select_expressions:
                try:
                    # Handle aliased expressions
                    target_name = expr.alias if hasattr(expr, 'alias') else None
                    if not target_name and hasattr(expr, 'name'):
                        target_name = expr.name
                    
                    # Skip if no target name found
                    if not target_name:
                        continue
                    
                    # Find source columns
                    source_columns = []
                    for col in expr.find_all(exp.Column):
                        if hasattr(col, 'table') and hasattr(col, 'name'):
                            source_table = self._resolve_table_name(col.table)
                            source_columns.append({
                                'table': source_table,
                                'column': col.name
                            })
                    
                    # Analyze transformations
                    transformation = None
                    if hasattr(expr, 'args'):
                        transformation = {
                            'type': 'function',
                            'name': expr.name if hasattr(expr, 'name') else str(expr),
                            'arguments': [str(arg) for arg in expr.args]
                        }
                    elif isinstance(expr, exp.Binary):
                        transformation = {
                            'type': 'operation',
                            'operator': str(expr.op),
                            'left': str(expr.left),
                            'right': str(expr.right)
                        }
                    
                    # Add to column lineage
                    for source in source_columns:
                        lineage = {
                            'target_column': target_name,
                            'source': source,
                            'transformation': transformation
                        }
                        analysis['column_lineage'].append(lineage)
                    
                    # Track transformations
                    if transformation:
                        analysis['transformations'].append({
                            'target': target_name,
                            'type': transformation['type'],
                            'details': transformation
                        })
                    
                except Exception as e:
                    logger.error(f"Error analyzing column expression: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in column analysis: {str(e)}")

    def _analyze_filters(self, where_clause) -> List[Dict[str, Any]]:
        """Analyze WHERE clause filters."""
        filters = []
        try:
            for condition in where_clause.find_all(exp.Binary):
                filter_info = {
                    'type': 'condition',
                    'operator': str(condition.op),
                    'left': str(condition.left),
                    'right': str(condition.right)
                }
                filters.append(filter_info)
        except Exception as e:
            logger.error(f"Error analyzing filters: {str(e)}")
        return filters

    def _resolve_table_name(self, table_reference: str) -> str:
        """Resolve table name from alias."""
        if table_reference in self.table_aliases:
            return self.table_aliases[table_reference]
        return table_reference

    def _extract_column_constraints(self, column) -> List[str]:
        """Extract column constraints."""
        constraints = []
        if hasattr(column, 'constraints'):
            for constraint in column.constraints:
                constraints.append(str(constraint))
        return constraints

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships while preserving the most detailed information."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a normalized key for comparison
            key = tuple(sorted([
                f"{rel['left_table']}.{rel['left_column']}",
                f"{rel['right_table']}.{rel['right_column']}"
            ]))
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships

    def _resolve_column_lineage(self, lineage: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve and deduplicate column lineage information."""
        resolved_lineage = []
        seen = set()
        
        for item in lineage:
            source = item['source']
            target = item['target_column']
            
            # Resolve table aliases
            if source['table'] in self.table_aliases:
                source['table'] = self.table_aliases[source['table']]
            
            key = f"{source['table']}.{source['column']}->{target}"
            if key not in seen:
                seen.add(key)
                resolved_lineage.append(item)
        
        return resolved_lineage

    def _extract_business_context(self, content: str) -> List[Dict[str, str]]:
        """Enhanced business context extraction from SQL comments."""
        context = []
        current_block = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Handle different comment styles
            if line.startswith('--') or line.startswith('#'):
                comment = line.lstrip('-#').strip()
                if comment:
                    current_block.append(comment)
            elif line.startswith('/*') or line.endswith('*/'):
                # Handle block comments
                comment = line.replace('/*', '').replace('*/', '').strip()
                if comment:
                    current_block.append(comment)
            else:
                # Process accumulated block if it exists
                if current_block:
                    combined_comment = ' '.join(current_block)
                    if len(combined_comment) > 10:  # Filter out short comments
                        context_type = self._categorize_business_context(combined_comment)
                        context.append({
                            'category': context_type,
                            'description': combined_comment
                        })
                    current_block = []
        
        # Process any remaining block
        if current_block:
            combined_comment = ' '.join(current_block)
            if len(combined_comment) > 10:
                context_type = self._categorize_business_context(combined_comment)
                context.append({
                    'category': context_type,
                    'description': combined_comment
                })
        
        return context

    def _categorize_business_context(self, comment: str) -> str:
        """Categorize business context based on content."""
        comment_lower = comment.lower()
        
        categories = {
            'metric': ['revenue', 'sales', 'conversion', 'rate', 'count', 'total'],
            'process': ['workflow', 'process', 'step', 'procedure', 'pipeline'],
            'business_rule': ['rule', 'policy', 'requirement', 'must', 'should', 'validate'],
            'definition': ['means', 'defined as', 'refers to', 'represents'],
            'technical': ['join', 'table', 'column', 'query', 'database']
        }
        
        for category, keywords in categories.items():
            if any(keyword in comment_lower for keyword in keywords):
                return category
        
        return 'general'

    def analyze_file(self, file_path: str, language: str) -> Dict[str, Any]:
        """Analyze code file and extract metadata."""
        try:
            # Cache key based on file content hash
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Use cached analysis if available
            if hasattr(self, '_file_analysis_cache'):
                content_hash = hash(content)
                if content_hash in self._file_analysis_cache:
                    return self._file_analysis_cache[content_hash]
            else:
                self._file_analysis_cache = {}
            
            # Perform analysis
            language = language.lower()
            if language in ['sql', 'mysql', 'postgresql', 'sqlite']:
                analysis = self.analyze_sql(content)
            elif language in ['py', 'python']:
                analysis = self.analyze_python(content)
            else:
                analysis = {
                    "language": language,
                    "supported": False,
                    "message": f"Basic analysis only for {language}"
                }
            
            # Cache the results
            self._file_analysis_cache[hash(content)] = analysis
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing {language} file: {str(e)}")
            return {
                "language": language,
                "error_message": str(e),
                "supported": False
            }