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
        """Analyze a file based on its detected language"""
        try:
            # Determine language if not provided
            if not language:
                _, ext = os.path.splitext(file_path)
                language = self._determine_language(file_path, ext.lower().lstrip('.'))
            
            # Check if this is a dbt file
            if self._is_dbt_file(file_path):
                # Handle dbt-specific files
                if file_path.endswith('.yml') or file_path.endswith('.yaml'):
                    return self.analyze_dbt_schema(file_path)
                elif file_path.endswith('.sql'):
                    return self.analyze_dbt(file_path)
            
            # Proceed with regular language-specific analysis
            if language == 'python':
                return self.analyze_python(file_path)
            elif language == 'sql':
                return self.analyze_sql(file_path)
            # ... other languages
            
            return {"file_path": file_path, "language": language or "unknown"}
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {"file_path": file_path, "error": str(e), "language": language or "unknown"}
    
    def _is_dbt_file(self, file_path: str) -> bool:
        """Determine if a file is part of a dbt project"""
        # Check if path contains dbt project directories
        dbt_dirs = ['models', 'macros', 'snapshots', 'seeds', 'analyses', 'tests']
        file_parts = file_path.split(os.sep)
        
        for dbt_dir in dbt_dirs:
            if dbt_dir in file_parts:
                return True
        
        # Check for dbt_project.yml in parent directories
        dir_path = os.path.dirname(file_path)
        while dir_path and dir_path != os.path.dirname(dir_path):
            if os.path.exists(os.path.join(dir_path, 'dbt_project.yml')):
                return True
            dir_path = os.path.dirname(dir_path)
        
        # Check file content for dbt patterns if it's a SQL file
        if file_path.endswith('.sql'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read just the first 1000 chars
                    if '{{ ref(' in content or '{{ source(' in content:
                        return True
            except:
                pass
                
        return False
    
    def analyze_dbt_schema(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a dbt schema.yml file to extract metadata.
        
        Args:
            file_path: Path to the dbt schema file
            
        Returns:
            Dictionary with schema analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                yaml_data = yaml.safe_load(content)
            except Exception as e:
                logger.warning(f"Error parsing YAML file {file_path}: {str(e)}")
                return {
                    "file_path": file_path,
                    "file_type": "dbt_schema",
                    "error": f"Invalid YAML: {str(e)}",
                    "size": os.path.getsize(file_path)
                }
            
            # Initialize schema analysis
            schema_analysis = {
                "file_path": file_path,
                "file_type": "dbt_schema",
                "models": [],
                "sources": [],
                "size": os.path.getsize(file_path)
            }
            
            # Extract models
            if yaml_data and 'models' in yaml_data:
                for model in yaml_data['models']:
                    model_info = {
                        "name": model.get('name', ''),
                        "description": model.get('description', ''),
                        "columns": [],
                        "tests": model.get('tests', []),
                        "config": model.get('config', {})
                    }
                    
                    # Extract columns
                    if 'columns' in model:
                        for col_name, col_data in model['columns'].items():
                            column_info = {
                                "name": col_name,
                                "description": col_data.get('description', ''),
                                "tests": col_data.get('tests', []),
                                "meta": col_data.get('meta', {})
                            }
                            model_info["columns"].append(column_info)
                    
                    schema_analysis["models"].append(model_info)
            
            # Extract sources
            if yaml_data and 'sources' in yaml_data:
                for source in yaml_data['sources']:
                    source_info = {
                        "name": source.get('name', ''),
                        "description": source.get('description', ''),
                        "database": source.get('database', ''),
                        "schema": source.get('schema', ''),
                        "tables": []
                    }
                    
                    # Extract tables
                    if 'tables' in source:
                        for table in source['tables']:
                            table_info = {
                                "name": table.get('name', ''),
                                "description": table.get('description', ''),
                                "columns": [],
                                "tests": table.get('tests', []),
                                "meta": table.get('meta', {})
                            }
                            
                            # Extract columns
                            if 'columns' in table:
                                for col_name, col_data in table['columns'].items():
                                    column_info = {
                                        "name": col_name,
                                        "description": col_data.get('description', ''),
                                        "tests": col_data.get('tests', []),
                                        "meta": col_data.get('meta', {})
                                    }
                                    table_info["columns"].append(column_info)
                            
                            source_info["tables"].append(table_info)
                    
                    schema_analysis["sources"].append(source_info)
            
            # Extract exposures if present
            if yaml_data and 'exposures' in yaml_data:
                schema_analysis["exposures"] = yaml_data['exposures']
            
            # Extract metrics if present
            if yaml_data and 'metrics' in yaml_data:
                schema_analysis["metrics"] = yaml_data['metrics']
            
            return schema_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dbt schema file: {str(e)}")
            return {
                "file_path": file_path,
                "file_type": "dbt_schema",
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
            
            # Extract tests
            jinja_tests = []
            for line in content.split('\n'):
                if '{{' in line and 'test' in line.lower():
                    jinja_tests.append(line.strip())
            
            # Extract model name from file path or YAML
            model_name = os.path.basename(file_path).split('.')[0]
            if yaml_data and 'name' in yaml_data:
                model_name = yaml_data['name']
            
            # Determine model type from path
            model_type = self._determine_dbt_model_type(file_path)
            
            # Create an enhanced analysis result
            dbt_analysis = {
                "file_path": file_path,
                "file_type": f"dbt_{model_type}",
                "model_name": model_name,
                "model_type": model_type,
                "yaml_config": yaml_data,
                "jinja_references": jinja_refs,
                "jinja_sources": jinja_sources,
                "jinja_macros": jinja_macros,
                "jinja_tests": jinja_tests,
                "size": os.path.getsize(file_path),
                "documentation": yaml_data.get('description', '')
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
            
            # Extract materialization from SQL config if present
            materialization_match = re.search(r'{{\s*config\s*\(\s*materialized\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            if materialization_match:
                dbt_analysis["materialization"] = materialization_match.group(1)
            
            # Extract schema from SQL config if present
            schema_match = re.search(r'{{\s*config\s*\(\s*schema\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            if schema_match:
                dbt_analysis["schema"] = schema_match.group(1)
            
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
            
            # Extract dependencies between models
            dependencies = {
                "depends_on": {
                    "models": jinja_refs,
                    "sources": [f"{source[0]}.{source[1]}" for source in jinja_sources]
                },
                "supports": []  # Models that depend on this model (populated later)
            }
            dbt_analysis["dependencies"] = dependencies
            
            # Extract columns using regex from the simplified SQL
            columns = self._extract_columns_from_sql(simplified_sql)
            if columns:
                dbt_analysis["columns"] = columns
            
            # Extract SQL comments for additional documentation
            sql_comments = re.findall(r'--\s*(.*?)(?:\n|$)', content)
            if sql_comments:
                dbt_analysis["sql_comments"] = sql_comments
            
            # Extract doc blocks (multiline comments)
            doc_blocks = re.findall(r'/\*\*(.*?)\*/', content, re.DOTALL)
            if doc_blocks:
                dbt_analysis["doc_blocks"] = [block.strip() for block in doc_blocks]
            
            return dbt_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing DBT file: {str(e)}")
            return {
                "file_path": file_path,
                "file_type": "dbt",
                "error": str(e),
                "size": os.path.getsize(file_path)
            }
    
    def _extract_columns_from_sql(self, sql_content: str) -> List[str]:
        """Extract column names from SQL content using regex"""
        try:
            # Look for column definitions in SELECT statements
            selects = re.findall(r'select\s+(.*?)(?:from|where|group by|having|order by|limit|$)', 
                               sql_content.lower(), re.DOTALL)
            
            columns = []
            if selects:
                for select_clause in selects:
                    # Split the select clause by commas, handling subqueries and functions
                    select_parts = []
                    bracket_level = 0
                    current_part = ""
                    
                    for char in select_clause:
                        if char == ',' and bracket_level == 0:
                            select_parts.append(current_part.strip())
                            current_part = ""
                        else:
                            current_part += char
                            if char == '(':
                                bracket_level += 1
                            elif char == ')':
                                bracket_level -= 1
                    
                    if current_part.strip():
                        select_parts.append(current_part.strip())
                    
                    # Extract column names from each part
                    for part in select_parts:
                        # Handle aliased columns (using AS keyword)
                        as_match = re.search(r'(?:as\s+)?([a-zA-Z0-9_]+)$', part.strip(), re.IGNORECASE)
                        if as_match:
                            columns.append(as_match.group(1))
                        else:
                            # If no AS, use the last part of the expression
                            col_parts = part.split('.')
                            if col_parts and col_parts[-1].strip() != '*':
                                columns.append(col_parts[-1].strip())
            
            # Remove duplicates and return
            return list(set(columns))
            
        except Exception as e:
            logger.warning(f"Error extracting columns: {str(e)}")
            return []

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