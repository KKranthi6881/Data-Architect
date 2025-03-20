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
import subprocess
from datetime import datetime

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

@dataclass
class DbtLineage:
    """Class to represent DBT model lineage"""
    model_name: str
    upstream_models: List[str]  # Models this model depends on
    downstream_models: List[str]  # Models that depend on this model
    sources: List[Dict[str, str]]  # Source tables used
    intermediate_models: List[str]  # Intermediate transformations
    target_tables: List[str]  # Final output tables
    materialization: str  # How the model is materialized
    freshness: Optional[Dict[str, Any]]  # Data freshness rules
    tags: List[str]  # Model tags

@dataclass
class GitMetadata:
    """Class to represent Git metadata for a file"""
    last_modified: datetime
    last_author: str
    commit_hash: str
    commit_message: str
    file_history: List[Dict[str, Any]]
    branch_name: str
    total_commits: int
    contributors: List[str]

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
        self.dbt_lineage_cache = {}  # Cache for DBT lineage information
        self.git_metadata_cache = {}  # Cache for Git metadata
        self.dbt_model_types = {
            'models': 'model',
            'seeds': 'seed',
            'snapshots': 'snapshot',
            'analyses': 'analysis',
            'macros': 'macro',
            'tests': 'test'
        }
        
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
            
            # Initialize schema analysis with safe defaults
            schema_analysis = {
                "file_path": file_path,
                "file_type": "dbt_schema",
                "models": [],
                "sources": [],
                "size": os.path.getsize(file_path)
            }

            def safe_get(obj: Any, key: str, default: Any = None) -> Any:
                """Safely get a value from a dict or return default"""
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return default

            def process_column_data(col_data: Any, col_name: Optional[str] = None) -> Dict[str, Any]:
                """Process column data into a standardized format"""
                if isinstance(col_data, dict):
                    return {
                        "name": col_name or col_data.get('name', ''),
                        "description": col_data.get('description', ''),
                        "tests": col_data.get('tests', []) if isinstance(col_data.get('tests'), list) else [],
                        "meta": col_data.get('meta', {}) if isinstance(col_data.get('meta'), dict) else {}
                    }
                elif isinstance(col_data, str):
                    return {
                        "name": col_name or col_data,
                        "description": "",
                        "tests": [],
                        "meta": {}
                    }
                else:
                    return {
                        "name": col_name or str(col_data) if col_data is not None else '',
                        "description": "",
                        "tests": [],
                        "meta": {}
                    }

            def process_columns(columns: Any) -> List[Dict[str, Any]]:
                """Process columns into a standardized format"""
                processed_columns = []
                
                if isinstance(columns, dict):
                    for col_name, col_data in columns.items():
                        processed_columns.append(process_column_data(col_data, col_name))
                elif isinstance(columns, list):
                    for col_item in columns:
                        processed_columns.append(process_column_data(col_item))
                elif columns is not None:
                    # Handle case where columns is a single value
                    processed_columns.append(process_column_data(columns))
                
                return processed_columns

            # Process models
            if isinstance(yaml_data, dict) and 'models' in yaml_data:
                models_data = yaml_data['models']
                if isinstance(models_data, list):
                    for model in models_data:
                        if not isinstance(model, dict):
                            continue
                            
                        model_info = {
                            "name": safe_get(model, 'name', ''),
                            "description": safe_get(model, 'description', ''),
                            "columns": [],
                            "tests": safe_get(model, 'tests', []) if isinstance(safe_get(model, 'tests'), list) else [],
                            "config": safe_get(model, 'config', {}) if isinstance(safe_get(model, 'config'), dict) else {}
                        }
                        
                        # Process columns
                        if 'columns' in model:
                            model_info["columns"] = process_columns(model['columns'])
                        
                        schema_analysis["models"].append(model_info)

            # Process sources
            if isinstance(yaml_data, dict) and 'sources' in yaml_data:
                sources_data = yaml_data['sources']
                if isinstance(sources_data, list):
                    for source in sources_data:
                        if not isinstance(source, dict):
                            continue
                            
                        source_info = {
                            "name": safe_get(source, 'name', ''),
                            "description": safe_get(source, 'description', ''),
                            "database": safe_get(source, 'database', ''),
                            "schema": safe_get(source, 'schema', ''),
                            "tables": []
                        }
                        
                        # Process tables
                        tables = safe_get(source, 'tables', [])
                        if isinstance(tables, list):
                            for table in tables:
                                if not isinstance(table, dict):
                                    continue
                                    
                                table_info = {
                                    "name": safe_get(table, 'name', ''),
                                    "description": safe_get(table, 'description', ''),
                                    "columns": [],
                                    "tests": safe_get(table, 'tests', []) if isinstance(safe_get(table, 'tests'), list) else [],
                                    "meta": safe_get(table, 'meta', {}) if isinstance(safe_get(table, 'meta'), dict) else {}
                                }
                                
                                # Process columns
                                if 'columns' in table:
                                    table_info["columns"] = process_columns(table['columns'])
                                
                                source_info["tables"].append(table_info)
                        elif isinstance(tables, dict):
                            for table_name, table in tables.items():
                                table_info = {
                                    "name": table_name,
                                    "description": safe_get(table, 'description', '') if isinstance(table, dict) else str(table),
                                    "columns": [],
                                    "tests": safe_get(table, 'tests', []) if isinstance(table, dict) and isinstance(safe_get(table, 'tests'), list) else [],
                                    "meta": safe_get(table, 'meta', {}) if isinstance(table, dict) and isinstance(safe_get(table, 'meta'), dict) else {}
                                }
                                
                                # Process columns if table is a dict
                                if isinstance(table, dict) and 'columns' in table:
                                    table_info["columns"] = process_columns(table['columns'])
                                
                                source_info["tables"].append(table_info)
                        
                        schema_analysis["sources"].append(source_info)

            # Process exposures if present
            if isinstance(yaml_data, dict) and 'exposures' in yaml_data:
                exposures_data = yaml_data['exposures']
                if isinstance(exposures_data, (list, dict)):
                    schema_analysis["exposures"] = exposures_data

            # Process metrics if present
            if isinstance(yaml_data, dict) and 'metrics' in yaml_data:
                metrics_data = yaml_data['metrics']
                if isinstance(metrics_data, (list, dict)):
                    schema_analysis["metrics"] = metrics_data

            return schema_analysis

        except Exception as e:
            logger.error(f"Error analyzing dbt schema file: {str(e)}")
            return {
                "file_path": file_path,
                "file_type": "dbt_schema",
                "error": str(e),
                "size": os.path.getsize(file_path)
            }

    def _determine_dbt_model_type(self, file_path: str) -> str:
        """
        Determine the DBT model type based on file path and content.
        
        Args:
            file_path: Path to the DBT file
            
        Returns:
            Model type (model, seed, snapshot, etc.)
        """
        try:
            # First check path-based type
            path_parts = Path(file_path).parts
            for dir_name, model_type in self.dbt_model_types.items():
                if dir_name in path_parts:
                    return model_type

            # If not found in path, check file content
            if file_path.endswith('.sql'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # Read first 1000 chars
                        
                        # Check for snapshot indicators
                        if '{% snapshot' in content:
                            return 'snapshot'
                        
                        # Check for test indicators
                        if content.strip().startswith('test ') or '{% test' in content:
                            return 'test'
                        
                        # Check for analysis indicators
                        if '/analyses/' in file_path or '/analysis/' in file_path:
                            return 'analysis'
                        
                        # Check for macro indicators
                        if '{% macro' in content:
                            return 'macro'
                        
                        # Default to model if it has typical model patterns
                        if '{{ ref(' in content or '{{ source(' in content:
                            return 'model'
                except Exception as e:
                    logger.warning(f"Error reading file content for type determination: {str(e)}")

            # Check file extension for seeds
            if file_path.endswith('.csv'):
                return 'seed'

            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Error determining DBT model type: {str(e)}")
            return 'unknown'

    def _determine_language(self, file_path: str, extension: str) -> str:
        """
        Determine the programming language of a file.
        
        Args:
            file_path: Path to the file
            extension: File extension
            
        Returns:
            Language identifier
        """
        try:
            # Map extensions to languages
            extension_map = {
                'py': 'python',
                'sql': 'sql',
                'yml': 'yaml',
                'yaml': 'yaml',
                'json': 'json',
                'md': 'markdown',
                'ipynb': 'jupyter'
            }
            
            # First check extension
            if extension in extension_map:
                return extension_map[extension]
            
            # For files without extension or ambiguous ones, check content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    
                    # Check for Python indicators
                    if content.startswith('#!/usr/bin/env python') or \
                       'import ' in content or \
                       'from ' in content and ' import ' in content:
                        return 'python'
                    
                    # Check for SQL indicators
                    if content.strip().upper().startswith(('SELECT ', 'CREATE ', 'INSERT ', 'UPDATE ', 'DELETE ')):
                        return 'sql'
                    
                    # Check for YAML indicators
                    if content.startswith('---') or \
                       ': ' in content and '\n- ' in content:
                        return 'yaml'
                    
            except Exception as e:
                logger.warning(f"Error reading file content for language determination: {str(e)}")
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Error determining language: {str(e)}")
            return 'unknown'

    def analyze_dbt(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a DBT model file.
        
        Args:
            file_path: Path to the DBT model file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get git metadata
            git_metadata = self._get_git_metadata(file_path)
            
            # Basic file info
            basic_analysis = {
                "file_path": file_path,
                "file_type": "dbt",
                "size": os.path.getsize(file_path),
                "git_metadata": git_metadata,
                "last_modified": git_metadata.get("commit_date") if git_metadata else None,
                "last_author": git_metadata.get("author") if git_metadata else None
            }
            
            # Try to build DBT lineage
            try:
                lineage = self._build_dbt_lineage(file_path)
                if lineage:
                    basic_analysis["lineage"] = {
                        "model_name": lineage.model_name,
                        "upstream_models": lineage.upstream_models,
                        "downstream_models": lineage.downstream_models,
                        "sources": lineage.sources,
                        "materialization": lineage.materialization,
                        "tags": lineage.tags
                    }
                    if lineage.freshness:
                        basic_analysis["lineage"]["freshness"] = lineage.freshness
            except Exception as e:
                logger.warning(f"Error building DBT lineage: {str(e)}")
            
            # Try to extract SQL information
            try:
                # Remove Jinja templating for SQL analysis
                simplified_sql = re.sub(r'{%.*?%}', '', content, flags=re.DOTALL)
                simplified_sql = re.sub(r'{{.*?}}', '', simplified_sql, flags=re.DOTALL)
                
                # Extract tables and relationships
                tables, relationships = self._extract_sql_metadata(simplified_sql)
                if tables:
                    basic_analysis["tables"] = tables
                if relationships:
                    basic_analysis["relationships"] = relationships

            except Exception as e:
                logger.warning(f"Error extracting SQL information: {str(e)}")
            
            return basic_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing DBT file: {str(e)}")
            return {
                "file_path": file_path,
                "file_type": "dbt",
                "error": str(e),
                "size": os.path.getsize(file_path)
            }

    def _get_git_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get Git metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with Git metadata
        """
        try:
            # Get the repository root directory
            repo_dir = os.path.dirname(file_path)
            while repo_dir and not os.path.isdir(os.path.join(repo_dir, '.git')):
                parent = os.path.dirname(repo_dir)
                if parent == repo_dir:  # Reached root directory
                    repo_dir = None
                    break
                repo_dir = parent
            
            if not repo_dir:
                logger.warning(f"Could not find Git repository for {file_path}")
                return {}
            
            # Get the relative path within the repository
            rel_path = os.path.relpath(file_path, repo_dir)
            
            # Get the last commit information
            git_info = subprocess.check_output(
                ["git", "log", "-1", "--format=%H|%an|%at|%s", "--", rel_path],
                cwd=repo_dir,  # Use the repository root as the working directory
                text=True
            ).strip()
            
            if not git_info:
                return {}
                
            # Parse the git info
            parts = git_info.split('|')
            if len(parts) < 4:
                return {}
                
            commit_hash, author, timestamp, message = parts
            
            # Convert timestamp to datetime
            commit_date = datetime.fromtimestamp(int(timestamp))
            
            return {
                "commit_hash": commit_hash,
                "author": author,
                "commit_date": commit_date.isoformat(),
                "commit_message": message
            }
            
        except Exception as e:
            logger.warning(f"Error getting git metadata for {file_path}: {str(e)}")
            return {}

    def _build_dbt_lineage(self, model_path: str) -> Optional[DbtLineage]:
        """Build comprehensive DBT lineage for a model"""
        try:
            # Check cache first
            if model_path in self.dbt_lineage_cache:
                return self.dbt_lineage_cache[model_path]
            
            with open(model_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract model name from path
            model_name = os.path.basename(model_path).split('.')[0]
            
            # Find all refs (upstream models)
            upstream_models = list(set(
                re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', content)
            ))
            
            # Find all sources
            sources = []
            source_matches = re.findall(
                r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}',
                content
            )
            for source, table in source_matches:
                sources.append({
                    'source_name': source,
                    'table_name': table
                })
            
            # Extract materialization
            materialization = 'view'  # default
            materialization_match = re.search(
                r'{{\s*config\s*\(\s*materialized\s*=\s*[\'"]([^\'"]+)[\'"]',
                content
            )
            if materialization_match:
                materialization = materialization_match.group(1)
            
            # Extract tags from config
            tags = []
            tags_match = re.search(
                r'{{\s*config\s*\(\s*tags\s*=\s*\[([^\]]+)\]',
                content
            )
            if tags_match:
                tags = [t.strip(' \'"') for t in tags_match.group(1).split(',')]
            
            # Extract freshness rules from schema.yml
            freshness = None
            schema_path = os.path.join(os.path.dirname(model_path), 'schema.yml')
            if os.path.exists(schema_path):
                try:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_data = yaml.safe_load(f)
                        if schema_data and 'models' in schema_data:
                            for model in schema_data['models']:
                                if model.get('name') == model_name:
                                    freshness = model.get('freshness')
                                    break
                except Exception as e:
                    logger.warning(f"Error reading schema.yml for {model_path}: {str(e)}")
            
            # Find downstream models (models that reference this one)
            downstream_models = []
            models_dir = self._find_dbt_models_dir(model_path)
            if models_dir:
                for root, _, files in os.walk(models_dir):
                    for file in files:
                        if file.endswith('.sql'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                    if f'ref(\'{model_name}\')' in file_content or f'ref("{model_name}")' in file_content:
                                        downstream_models.append(os.path.basename(file_path).split('.')[0])
                            except Exception as e:
                                logger.warning(f"Error reading {file_path}: {str(e)}")
            
            # Identify intermediate and target tables
            intermediate_models = []
            target_tables = []
            
            for downstream in downstream_models:
                downstream_path = os.path.join(models_dir, f"{downstream}.sql")
                if os.path.exists(downstream_path):
                    with open(downstream_path, 'r', encoding='utf-8') as f:
                        downstream_content = f.read()
                        # If this downstream model is referenced by others, it's intermediate
                        if re.search(r'{{\s*ref\([\'"]' + downstream + r'[\'"]\)\s*}}', downstream_content):
                            intermediate_models.append(downstream)
                        else:
                            target_tables.append(downstream)
            
            lineage = DbtLineage(
                model_name=model_name,
                upstream_models=upstream_models,
                downstream_models=downstream_models,
                sources=sources,
                intermediate_models=intermediate_models,
                target_tables=target_tables,
                materialization=materialization,
                freshness=freshness,
                tags=tags
            )
            
            # Cache the result
            self.dbt_lineage_cache[model_path] = lineage
            return lineage
            
        except Exception as e:
            logger.error(f"Error building DBT lineage for {model_path}: {str(e)}")
            return None

    def _find_dbt_models_dir(self, file_path: str) -> Optional[str]:
        """Find the DBT models directory from a file path"""
        current_dir = os.path.dirname(file_path)
        while current_dir and current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, 'dbt_project.yml')):
                models_dir = os.path.join(current_dir, 'models')
                if os.path.exists(models_dir):
                    return models_dir
            current_dir = os.path.dirname(current_dir)
        return None

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
        try:
            # First, handle DBT-specific macros
            content = self._handle_dbt_macros(content)
            
            # Replace Jinja control structures with empty strings
            content = re.sub(r'{%.*?%}', ' ', content, flags=re.DOTALL)
            
            # Replace Jinja expressions with placeholder values
            content = re.sub(r'{{.*?}}', 'NULL', content, flags=re.DOTALL)
            
            # Replace Jinja comments with SQL comments
            content = re.sub(r'{#.*?#}', '/* Jinja comment */', content, flags=re.DOTALL)
            
            # Clean up any remaining Jinja artifacts
            content = re.sub(r'\{\{|\}\}|\{%|%\}|\{#|#\}', ' ', content)
            
            # Remove multiple spaces
            content = re.sub(r'\s+', ' ', content)
            
            return content
            
        except Exception as e:
            logger.warning(f"Error preprocessing Jinja SQL: {str(e)}")
            return content

    def _handle_dbt_macros(self, content: str) -> str:
        """Handle DBT-specific macros and convert them to valid SQL"""
        try:
            # Handle ref() macro
            content = re.sub(
                r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}',
                r'\1',  # Replace with table name
                content
            )
            
            # Handle source() macro
            content = re.sub(
                r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}',
                r'\1_\2',  # Replace with source_table
                content
            )
            
            # Handle config() macro
            content = re.sub(
                r'{{\s*config\([^\)]+\)\s*}}',
                '',  # Remove config blocks
                content
            )
            
            # Handle var() macro
            content = re.sub(
                r'{{\s*var\([\'"]([^\'"]+)[\'"]\)\s*}}',
                'NULL',  # Replace with NULL
                content
            )
            
            # Handle macro calls
            content = re.sub(
                r'{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\([^\)]*\)\s*}}',
                'NULL',  # Replace with NULL
                content
            )
            
            return content
            
        except Exception as e:
            logger.warning(f"Error handling DBT macros: {str(e)}")
            return content

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
        
        try:
            # First try parsing with sqlglot
            parsed = False
            for dialect in self.supported_dialects:
                try:
                    statements = sqlglot.parse(sql_content, dialect=dialect)
                    parsed = True
                    
                    # Process each statement
                    for statement in statements:
                        self._process_sql_statement(statement, tables, relationships)
                    
                    break  # Stop if parsing succeeds
                except Exception as e:
                    continue
            
            if not parsed:
                # Fall back to regex-based parsing
                logger.info("Falling back to regex-based SQL parsing")
                tables = self._extract_tables_basic(sql_content)
                
                # Try to extract relationships from JOIN conditions
                join_relationships = self._extract_joins_basic(sql_content)
                relationships.extend(join_relationships)
            
            return tables, relationships
            
        except Exception as e:
            logger.warning(f"Error extracting SQL metadata: {str(e)}")
            return [], []

    def _extract_joins_basic(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract join relationships using regex"""
        relationships = []
        
        try:
            # Find JOIN clauses
            join_pattern = re.compile(
                r'(\w+\s+)?JOIN\s+(?:([^\s.]+)\.)?([^\s]+)\s+(?:AS\s+)?(\w+)?\s+ON\s+(.+?)(?=(?:\s+(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)\s+JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$))',
                re.IGNORECASE | re.DOTALL
            )
            
            matches = join_pattern.finditer(sql_content)
            
            for match in matches:
                join_type = (match.group(1) or 'INNER').strip().upper()
                schema = match.group(2)
                table = match.group(3)
                alias = match.group(4)
                condition = match.group(5)
                
                # Extract tables and columns from the join condition
                condition_pattern = re.compile(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)')
                condition_matches = condition_pattern.finditer(condition)
                
                for cond_match in condition_matches:
                    left_table, left_col, right_table, right_col = cond_match.groups()
                    
                    relationships.append({
                        "source_table": left_table,
                        "source_schema": schema if left_table == table else None,
                        "source_columns": [left_col],
                        "target_table": right_table,
                        "target_schema": schema if right_table == table else None,
                        "target_columns": [right_col],
                        "relationship_type": "JOIN",
                        "join_type": join_type
                    })
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Error extracting joins: {str(e)}")
            return []

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

    def analyze_yaml(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the YAML
            try:
                yaml_data = yaml.safe_load(content)
            except Exception as e:
                logger.warning(f"Error parsing YAML file {file_path}: {str(e)}")
                yaml_data = None
            
            # Get git metadata if available
            git_metadata = self._get_git_metadata(file_path)
            
            return {
                "file_path": file_path,
                "file_type": "yaml",
                "yaml_structure": yaml_data if yaml_data else {},
                "git_metadata": git_metadata
            }
        except Exception as e:
            logger.error(f"Error analyzing YAML file {file_path}: {str(e)}")
            return {"file_path": file_path, "error": str(e), "file_type": "yaml"}

    def analyze_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Markdown file.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract headers
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            structured_headers = []
            
            for h in headers:
                level = len(h[0])
                text = h[1].strip()
                structured_headers.append({"level": level, "text": text})
            
            # Get git metadata if available
            git_metadata = self._get_git_metadata(file_path)
            
            return {
                "file_path": file_path,
                "file_type": "markdown",
                "headers": structured_headers,
                "content_length": len(content),
                "git_metadata": git_metadata
            }
        except Exception as e:
            logger.error(f"Error analyzing Markdown file {file_path}: {str(e)}")
            return {"file_path": file_path, "error": str(e), "file_type": "markdown"}