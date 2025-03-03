from typing import Dict, List, Any, Optional, Set, Tuple
import sqlglot
from sqlglot import parse_one, exp
import ast
import logging
from pathlib import Path
import re
from dataclasses import dataclass
from collections import defaultdict
import os
import yaml
import json

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
        self.supported_dialects = [
            'sqlite', 'mysql', 'postgres', 'bigquery', 
            'snowflake', 'redshift', 'duckdb'
        ]
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

    def analyze_file(self, file_path: str, language: str = None) -> Dict[str, Any]:
        """
        Analyze a file based on its extension or specified language.
        
        Args:
            file_path: Path to the file
            language: Optional language override
            
        Returns:
            Dictionary with analysis results
        """
        if not language:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')
            
            # Map extension to language
            if ext == 'py':
                language = 'python'
            elif ext in ['sql', 'ddl', 'dml']:
                language = 'sql'
            elif ext in ['yml', 'yaml']:
                language = 'yaml'
            elif ext in ['js', 'jsx']:
                language = 'javascript'
            elif ext in ['ts', 'tsx']:
                language = 'typescript'
            elif ext in ['md', 'markdown']:
                language = 'markdown'
            elif ext in ['sql.jinja', 'sql.j2', 'sql.jinja2', 'sql.dbt']:
                language = 'dbt'
            
        # Analyze based on language
        if language == 'python':
            return self.analyze_python(file_path)
        elif language == 'sql':
            return self.analyze_sql(file_path)
        elif language == 'dbt':
            return self.analyze_dbt(file_path)
        else:
            # Basic analysis for unsupported languages
            return {
                "file_path": file_path,
                "language": language or "unknown",
                "size": os.path.getsize(file_path)
            }

    def analyze_python(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file to extract classes, functions, imports, etc.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the Python code
            tree = ast.parse(content)
            
            # Extract information
            imports = []
            classes = []
            functions = []
            variables = []
            
            for node in ast.walk(tree):
                # Extract imports
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        imports.append(f"{module}.{name.name}")
                
                # Extract classes
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        "line": node.lineno
                    }
                    classes.append(class_info)
                
                # Extract functions
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "args": [a.arg for a in node.args.args],
                        "line": node.lineno
                    }
                    functions.append(func_info)
                
                # Extract global variables
                elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append({
                                "name": target.id,
                                "line": node.lineno
                            })
            
            return {
                "file_path": file_path,
                "language": "python",
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "variables": variables,
                "size": os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing python file: {str(e)}")
            return {
                "file_path": file_path,
                "language": "python",
                "error": str(e),
                "size": os.path.getsize(file_path)
            }

    def analyze_sql(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a SQL file to extract tables, columns, relationships, etc.
        
        Args:
            file_path: Path to the SQL file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract tables and relationships
            tables, relationships = self._extract_sql_metadata(content)
            
            return {
                "file_path": file_path,
                "language": "sql",
                "tables": tables,
                "relationships": relationships,
                "size": os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SQL file: {str(e)}")
            return {
                "file_path": file_path,
                "language": "sql",
                "error": str(e),
                "size": os.path.getsize(file_path)
            }

    def analyze_dbt(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a DBT SQL file to extract metadata, handling Jinja templates.
        
        Args:
            file_path: Path to the DBT file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract YAML frontmatter if present (between --- markers)
            yaml_data = {}
            yaml_match = re.search(r'---\s+(.*?)\s+---', content, re.DOTALL)
            if yaml_match:
                try:
                    yaml_text = yaml_match.group(1)
                    yaml_data = yaml.safe_load(yaml_text)
                    # Remove the YAML frontmatter for SQL parsing
                    content = content.replace(yaml_match.group(0), '')
                except Exception as e:
                    logger.warning(f"Error parsing YAML frontmatter: {str(e)}")
            
            # Extract Jinja macros and references
            jinja_refs = re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', content)
            jinja_sources = re.findall(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', content)
            jinja_macros = re.findall(r'{{\s*([a-zA-Z0-9_]+)\(', content)
            
            # Extract model name from file path or YAML
            model_name = os.path.basename(file_path).split('.')[0]
            if yaml_data and 'name' in yaml_data:
                model_name = yaml_data['name']
            
            # Create a basic analysis result with the information we can reliably extract
            dbt_analysis = {
                "file_path": file_path,
                "language": "dbt",
                "model_name": model_name,
                "yaml_config": yaml_data,
                "jinja_references": jinja_refs,
                "jinja_sources": jinja_sources,
                "jinja_macros": jinja_macros,
                "size": os.path.getsize(file_path)
            }
            
            # Extract model type and materialization
            if yaml_data:
                if 'config' in yaml_data and 'materialized' in yaml_data['config']:
                    dbt_analysis["materialization"] = yaml_data['config']['materialized']
                
                if 'description' in yaml_data:
                    dbt_analysis["description"] = yaml_data['description']
                    
                # Extract column descriptions if available
                if 'columns' in yaml_data:
                    dbt_analysis["column_descriptions"] = yaml_data['columns']
            
            # Try to extract SQL by removing/simplifying Jinja
            # Replace {{ ref('...') }} with table_name
            simplified_sql = re.sub(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', r'\1', content)
            # Replace {{ source('...', '...') }} with source_table_name
            simplified_sql = re.sub(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', r'\2', simplified_sql)
            # Remove Jinja control structures
            simplified_sql = re.sub(r'{%.*?%}', '', simplified_sql)
            # Replace other Jinja expressions with placeholders
            simplified_sql = re.sub(r'{{.*?}}', 'placeholder', simplified_sql)
            
            # Extract tables using regex (more reliable for DBT files)
            tables = self._extract_tables_basic(simplified_sql)
            dbt_analysis["tables"] = tables
            
            # Extract relationships between tables based on refs and sources
            relationships = []
            
            # Add relationships from ref() calls
            for ref in jinja_refs:
                relationships.append({
                    "source_model": ref,
                    "target_model": model_name,
                    "type": "ref"
                })
            
            # Add relationships from source() calls
            for source in jinja_sources:
                source_name, table_name = source
                relationships.append({
                    "source": source_name,
                    "source_table": table_name,
                    "target_model": model_name,
                    "type": "source"
                })
            
            dbt_analysis["relationships"] = relationships
            
            # Don't try to parse with SQLGlot for DBT files - it's too error-prone
            # Instead, extract columns using regex
            columns = self._extract_columns_from_sql(simplified_sql)
            if columns:
                dbt_analysis["columns"] = columns
            
            return dbt_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing DBT file: {str(e)}")
            return {
                "file_path": file_path,
                "language": "dbt",
                "error": str(e),
                "size": os.path.getsize(file_path)
            }

    def _preprocess_jinja_sql(self, content: str) -> str:
        """
        Preprocess Jinja templated SQL to make it parseable by SQL parsers.
        
        Args:
            content: The SQL content with Jinja templating
            
        Returns:
            Preprocessed SQL content
        """
        # Replace Jinja control structures with empty strings
        content = re.sub(r'{%.*?%}', ' ', content, flags=re.DOTALL)
        
        # Replace Jinja expressions with placeholder values
        content = re.sub(r'{{.*?}}', 'NULL', content, flags=re.DOTALL)
        
        # Replace Jinja comments with SQL comments
        content = re.sub(r'{#.*?#}', '/* Jinja comment */', content, flags=re.DOTALL)
        
        return content

    def _extract_dbt_metadata(self, yaml_data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Extract DBT-specific metadata from YAML frontmatter and file path.
        
        Args:
            yaml_data: Parsed YAML frontmatter
            file_path: Path to the DBT file
            
        Returns:
            Dictionary with DBT metadata
        """
        dbt_metadata = {
            "model_name": os.path.basename(file_path).split('.')[0],
            "model_type": self._determine_dbt_model_type(file_path),
            "config": yaml_data.get('config', {}),
            "description": yaml_data.get('description', ''),
            "columns": [],
            "tests": [],
            "sources": [],
            "refs": []
        }
        
        # Extract column definitions
        if 'columns' in yaml_data:
            for col_name, col_data in yaml_data['columns'].items():
                dbt_metadata['columns'].append({
                    "name": col_name,
                    "description": col_data.get('description', ''),
                    "tests": col_data.get('tests', [])
                })
        
        # Extract tests
        if 'tests' in yaml_data:
            dbt_metadata['tests'] = yaml_data['tests']
        
        # Extract refs and sources from the SQL content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find all ref() calls
            ref_matches = re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', content)
            dbt_metadata['refs'] = list(set(ref_matches))
            
            # Find all source() calls
            source_matches = re.findall(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', content)
            dbt_metadata['sources'] = [{"source": s[0], "table": s[1]} for s in source_matches]
        
        return dbt_metadata

    def _determine_dbt_model_type(self, file_path: str) -> str:
        """
        Determine the DBT model type based on file path.
        
        Args:
            file_path: Path to the DBT file
            
        Returns:
            Model type (model, seed, snapshot, etc.)
        """
        path_parts = file_path.split(os.sep)
        
        if 'models' in path_parts:
            return 'model'
        elif 'seeds' in path_parts:
            return 'seed'
        elif 'snapshots' in path_parts:
            return 'snapshot'
        elif 'analyses' in path_parts:
            return 'analysis'
        elif 'macros' in path_parts:
            return 'macro'
        elif 'tests' in path_parts:
            return 'test'
        else:
            return 'unknown'

    def _extract_sql_metadata(self, sql_content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract tables, columns, and relationships from SQL content.
        
        Args:
            sql_content: SQL content to analyze
            
        Returns:
            Tuple of (tables, relationships)
        """
        tables = []
        relationships = []
        
        # Try parsing with different dialects
        parsed = False
        for dialect in self.supported_dialects:
            try:
                # Split into statements
                statements = sqlglot.parse(sql_content, dialect=dialect)
                parsed = True
                
                # Process each statement
                for statement in statements:
                    self._process_sql_statement(statement, tables, relationships)
                
                break  # Stop if parsing succeeds
            except Exception as e:
                continue
        
        if not parsed:
            logger.error("Failed to parse SQL with any dialect")
            # Try a more basic approach to extract table names
            tables = self._extract_tables_basic(sql_content)
        
        return tables, relationships

    def _process_sql_statement(self, statement, tables, relationships):
        """Process a SQL statement to extract metadata."""
        try:
            # Handle CREATE TABLE statements
            if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Table):
                table_name = statement.this.name
                schema_name = statement.this.db or ''
                
                columns = []
                primary_keys = []
                foreign_keys = []
                
                # Extract columns
                if hasattr(statement, 'expressions') and statement.expressions:
                    for expr in statement.expressions:
                        if isinstance(expr, exp.ColumnDef):
                            column_name = expr.this.name
                            data_type = str(expr.kind) if expr.kind else 'UNKNOWN'
                            
                            # Check for constraints
                            constraints = []
                            for constraint in expr.args.get('constraints', []):
                                if isinstance(constraint, exp.PrimaryKey):
                                    primary_keys.append(column_name)
                                    constraints.append('PRIMARY KEY')
                                elif isinstance(constraint, exp.NotNull):
                                    constraints.append('NOT NULL')
                                elif isinstance(constraint, exp.Unique):
                                    constraints.append('UNIQUE')
                            
                            columns.append({
                                "name": column_name,
                                "data_type": data_type,
                                "constraints": constraints
                            })
                        
                        # Extract table-level constraints
                        elif isinstance(expr, exp.PrimaryKey):
                            for col in expr.expressions:
                                if hasattr(col, 'this') and hasattr(col.this, 'name'):
                                    primary_keys.append(col.this.name)
                        
                        elif isinstance(expr, exp.ForeignKey):
                            if hasattr(expr, 'columns') and hasattr(expr, 'reference'):
                                fk_columns = [col.name for col in expr.columns]
                                ref_table = expr.reference.name
                                ref_schema = expr.reference.db or schema_name
                                ref_columns = [col.name for col in expr.reference_columns]
                                
                                foreign_keys.append({
                                    "columns": fk_columns,
                                    "ref_table": ref_table,
                                    "ref_schema": ref_schema,
                                    "ref_columns": ref_columns
                                })
                                
                                # Add relationship
                                relationships.append({
                                    "source_table": table_name,
                                    "source_schema": schema_name,
                                    "source_columns": fk_columns,
                                    "target_table": ref_table,
                                    "target_schema": ref_schema,
                                    "target_columns": ref_columns,
                                    "relationship_type": "FOREIGN KEY"
                                })
                
                # Add table
                tables.append({
                    "name": table_name,
                    "schema": schema_name,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                })
            
            # Handle SELECT statements
            elif isinstance(statement, exp.Select):
                self._analyze_select(statement, relationships)
        
        except Exception as e:
            logger.warning(f"Error processing SQL statement: {str(e)}")

    def _analyze_select(self, select_stmt, relationships):
        """Analyze a SELECT statement to extract relationships."""
        try:
            # Extract source tables
            source_tables = self._extract_source_tables(select_stmt)
            
            # Extract target table (for CREATE TABLE AS or INSERT INTO)
            target_table = None
            if hasattr(select_stmt, 'parent') and select_stmt.parent:
                parent = select_stmt.parent
                if isinstance(parent, exp.Create) and isinstance(parent.this, exp.Table):
                    target_table = {
                        "name": parent.this.name,
                        "schema": parent.this.db or ''
                    }
                elif isinstance(parent, exp.Insert) and isinstance(parent.this, exp.Table):
                    target_table = {
                        "name": parent.this.name,
                        "schema": parent.this.db or ''
                    }
            
            # Extract join conditions
            joins = []
            if hasattr(select_stmt, 'joins') and select_stmt.joins:
                for join in select_stmt.joins:
                    if hasattr(join, 'on') and join.on:
                        joins.append(self._extract_join_condition(join))
            
            # Extract where conditions
            filters = []
            if hasattr(select_stmt, 'where') and select_stmt.where:
                filters = self._extract_filters(select_stmt.where)
            
            # Add relationships based on joins
            for join in joins:
                if join and 'left_table' in join and 'right_table' in join:
                    relationships.append({
                        "source_table": join['left_table'],
                        "source_schema": join.get('left_schema', ''),
                        "source_columns": join.get('left_columns', []),
                        "target_table": join['right_table'],
                        "target_schema": join.get('right_schema', ''),
                        "target_columns": join.get('right_columns', []),
                        "relationship_type": "JOIN",
                        "join_type": join.get('join_type', 'INNER')
                    })
            
            # Add lineage relationship if target table exists
            if target_table and source_tables:
                for source in source_tables:
                    relationships.append({
                        "source_table": source['name'],
                        "source_schema": source.get('schema', ''),
                        "target_table": target_table['name'],
                        "target_schema": target_table.get('schema', ''),
                        "relationship_type": "LINEAGE"
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing filters: {str(e)}")

    def _extract_source_tables(self, select_stmt):
        """Extract source tables from a SELECT statement."""
        source_tables = []
        
        try:
            # Extract from clause tables
            if hasattr(select_stmt, 'from') and select_stmt.from_:
                for from_item in select_stmt.from_:
                    if isinstance(from_item, exp.Table):
                        source_tables.append({
                            "name": from_item.name,
                            "schema": from_item.db or ''
                        })
                    elif isinstance(from_item, exp.Subquery):
                        # Recursively process subqueries
                        if hasattr(from_item, 'this') and isinstance(from_item.this, exp.Select):
                            sub_sources = self._extract_source_tables(from_item.this)
                            source_tables.extend(sub_sources)
            
            # Extract join tables
            if hasattr(select_stmt, 'joins') and select_stmt.joins:
                for join in select_stmt.joins:
                    if hasattr(join, 'this') and isinstance(join.this, exp.Table):
                        source_tables.append({
                            "name": join.this.name,
                            "schema": join.this.db or ''
                        })
                    elif hasattr(join, 'this') and isinstance(join.this, exp.Subquery):
                        # Recursively process subqueries
                        if hasattr(join.this, 'this') and isinstance(join.this.this, exp.Select):
                            sub_sources = self._extract_source_tables(join.this.this)
                            source_tables.extend(sub_sources)
        
        except Exception as e:
            logger.warning(f"Error extracting source tables: {str(e)}")
        
        return source_tables

    def _extract_join_condition(self, join):
        """Extract join condition details."""
        try:
            if not hasattr(join, 'on') or not join.on:
                return None
            
            join_type = join.kind.upper() if hasattr(join, 'kind') else 'INNER'
            
            # Extract tables and columns from the join condition
            left_table = None
            left_schema = None
            left_column = None
            right_table = None
            right_schema = None
            right_column = None
            
            # Handle basic equality join conditions
            if isinstance(join.on, exp.EQ):
                # Left side
                if isinstance(join.on.this, exp.Column):
                    left_column = join.on.this.name
                    if hasattr(join.on.this, 'table'):
                        left_table = join.on.this.table
                    if hasattr(join.on.this, 'db'):
                        left_schema = join.on.this.db
                
                # Right side
                if isinstance(join.on.expression, exp.Column):
                    right_column = join.on.expression.name
                    if hasattr(join.on.expression, 'table'):
                        right_table = join.on.expression.table
                    if hasattr(join.on.expression, 'db'):
                        right_schema = join.on.expression.db
            
            # If table names weren't in the join condition, try to get them from the join clause
            if not left_table and hasattr(join, 'this') and isinstance(join.this, exp.Table):
                left_table = join.this.name
                left_schema = join.this.db
            
            if not right_table and hasattr(join, 'from') and isinstance(join.from_, exp.Table):
                right_table = join.from_.name
                right_schema = join.from_.db
            
            return {
                "join_type": join_type,
                "left_table": left_table,
                "left_schema": left_schema,
                "left_columns": [left_column] if left_column else [],
                "right_table": right_table,
                "right_schema": right_schema,
                "right_columns": [right_column] if right_column else []
            }
        
        except Exception as e:
            logger.warning(f"Error extracting join condition: {str(e)}")
            return None

    def _extract_filters(self, where_clause):
        """Extract filter conditions from a WHERE clause."""
        filters = []
        
        try:
            if isinstance(where_clause, exp.And):
                # Recursively process AND conditions
                left_filters = self._extract_filters(where_clause.this)
                right_filters = self._extract_filters(where_clause.expression)
                filters.extend(left_filters)
                filters.extend(right_filters)
            
            elif isinstance(where_clause, exp.Or):
                # Recursively process OR conditions
                left_filters = self._extract_filters(where_clause.this)
                right_filters = self._extract_filters(where_clause.expression)
                filters.extend(left_filters)
                filters.extend(right_filters)
            
            elif isinstance(where_clause, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)):
                # Extract basic comparison
                filter_type = where_clause.__class__.__name__
                
                column = None
                if isinstance(where_clause.this, exp.Column):
                    column = {
                        "name": where_clause.this.name,
                        "table": where_clause.this.table if hasattr(where_clause.this, 'table') else None,
                        "schema": where_clause.this.db if hasattr(where_clause.this, 'db') else None
                    }
                
                value = None
                if hasattr(where_clause, 'expression'):
                    if isinstance(where_clause.expression, exp.Literal):
                        value = where_clause.expression.this
                    elif isinstance(where_clause.expression, exp.Column):
                        value = {
                            "name": where_clause.expression.name,
                            "table": where_clause.expression.table if hasattr(where_clause.expression, 'table') else None,
                            "schema": where_clause.expression.db if hasattr(where_clause.expression, 'db') else None
                        }
                
                filters.append({
                    "type": filter_type,
                    "column": column,
                    "value": value
                })
            
            elif isinstance(where_clause, exp.In):
                # Extract IN condition
                column = None
                if isinstance(where_clause.this, exp.Column):
                    column = {
                        "name": where_clause.this.name,
                        "table": where_clause.this.table if hasattr(where_clause.this, 'table') else None,
                        "schema": where_clause.this.db if hasattr(where_clause.this, 'db') else None
                    }
                
                values = []
                if hasattr(where_clause, 'expressions'):
                    for expr in where_clause.expressions:
                        if isinstance(expr, exp.Literal):
                            values.append(expr.this)
                
                filters.append({
                    "type": "IN",
                    "column": column,
                    "values": values
                })
            
            elif isinstance(where_clause, exp.Like):
                # Extract LIKE condition
                column = None
                if isinstance(where_clause.this, exp.Column):
                    column = {
                        "name": where_clause.this.name,
                        "table": where_clause.this.table if hasattr(where_clause.this, 'table') else None,
                        "schema": where_clause.this.db if hasattr(where_clause.this, 'db') else None
                    }
                
                pattern = None
                if hasattr(where_clause, 'expression') and isinstance(where_clause.expression, exp.Literal):
                    pattern = where_clause.expression.this
                
                filters.append({
                    "type": "LIKE",
                    "column": column,
                    "pattern": pattern
                })
        
        except Exception as e:
            logger.error(f"Error analyzing filters: {str(e)}")
        
        return filters

    def _extract_tables_basic(self, sql_content: str) -> List[Dict[str, Any]]:
        """
        Extract table names using regex as a fallback method.
        
        Args:
            sql_content: SQL content to analyze
            
        Returns:
            List of tables
        """
        tables = []
        
        # Find CREATE TABLE statements
        create_matches = re.finditer(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:([^\s.]+)\.)?([^\s.(]+)',
            sql_content,
            re.IGNORECASE
        )
        
        for match in create_matches:
            schema = match.group(1) or ''
            table = match.group(2)
            tables.append({
                "name": table,
                "schema": schema,
                "columns": []
            })
        
        # Find tables in FROM clauses
        from_matches = re.finditer(
            r'FROM\s+(?:([^\s.]+)\.)?([^\s,()]+)',
            sql_content,
            re.IGNORECASE
        )
        
        for match in from_matches:
            schema = match.group(1) or ''
            table = match.group(2)
            
            # Skip if already added or if it's a subquery alias
            if not any(t['name'] == table and t['schema'] == schema for t in tables):
                tables.append({
                    "name": table,
                    "schema": schema,
                    "columns": []
                })
        
        # Find tables in JOIN clauses
        join_matches = re.finditer(
            r'JOIN\s+(?:([^\s.]+)\.)?([^\s()]+)',
            sql_content,
            re.IGNORECASE
        )
        
        for match in join_matches:
            schema = match.group(1) or ''
            table = match.group(2)
            
            # Skip if already added
            if not any(t['name'] == table and t['schema'] == schema for t in tables):
                tables.append({
                    "name": table,
                    "schema": schema,
                    "columns": []
                })
        
        return tables

    def _extract_columns_from_sql(self, sql_content: str) -> List[Dict[str, str]]:
        """
        Extract column definitions from SQL using regex.
        
        Args:
            sql_content: SQL content to analyze
            
        Returns:
            List of column definitions
        """
        columns = []
        
        # Look for column definitions in CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE.*?\((.*?)\)'
        create_matches = re.search(create_table_pattern, sql_content, re.IGNORECASE | re.DOTALL)
        
        if create_matches:
            column_text = create_matches.group(1)
            # Split by commas, but not commas inside parentheses (for complex types)
            column_defs = re.findall(r'([^,]+(?:\([^)]*\)[^,]*)?),?', column_text)
            
            for col_def in column_defs:
                col_def = col_def.strip()
                # Skip if this is a constraint definition
                if col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'UNIQUE')):
                    continue
                    
                # Extract column name and type
                col_match = re.match(r'[\"`]?([a-zA-Z0-9_]+)[\"`]?\s+([a-zA-Z0-9_]+(?:\([^)]+\))?)', col_def)
                if col_match:
                    col_name = col_match.group(1)
                    col_type = col_match.group(2)
                    
                    column = {
                        "name": col_name,
                        "type": col_type
                    }
                    
                    # Check for NOT NULL
                    if 'NOT NULL' in col_def.upper():
                        column["nullable"] = False
                    else:
                        column["nullable"] = True
                        
                    columns.append(column)
        
        # If no columns found in CREATE TABLE, try to extract from SELECT statements
        if not columns:
            # Find columns in SELECT statements
            select_columns = re.findall(r'SELECT\s+(.*?)\s+FROM', sql_content, re.IGNORECASE | re.DOTALL)
            
            if select_columns:
                # Take the first SELECT statement
                col_list = select_columns[0]
                # Split by commas, but handle function calls and subqueries
                col_items = []
                
                # Simple split for basic cases
                col_items = [c.strip() for c in col_list.split(',')]
                
                for col_item in col_items:
                    # Skip * selections
                    if col_item == '*':
                        continue
                        
                    # Check for aliased columns
                    alias_match = re.search(r'(?:AS\s+|)[\"`]?([a-zA-Z0-9_]+)[\"`]?$', col_item, re.IGNORECASE)
                    if alias_match:
                        col_name = alias_match.group(1)
                    else:
                        # For non-aliased columns, use the expression or column name
                        col_parts = col_item.split('.')
                        col_name = col_parts[-1].strip('"`')
                    
                    columns.append({
                        "name": col_name,
                        "type": "unknown"  # Type information not available in SELECT
                    })
        
        return columns