from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import os
import tempfile
import subprocess
from pathlib import Path
import yaml
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
import shutil
from urllib.parse import urlparse
import sqlite3
import difflib
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Singleton to track repo state across agent calls
class RepoSingleton:
    _instance = None
    _repos = {}  # Store multiple repos by URL
    _lock = __import__('threading').Lock()
    
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._instance._load_repos()
        return cls._instance
    
    def __init__(self):
        self._cache_file = os.path.join(os.path.expanduser("~"), ".dbt_data_architect", "repo_cache.json")
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
    
    def _load_repos(self):
        """Load repository mapping from cache file"""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Validate each repo path exists before adding it
                    for repo_url, repo_path in cache_data.items():
                        if os.path.exists(repo_path) and os.path.exists(os.path.join(repo_path, ".git")):
                            self._repos[repo_url] = repo_path
                        else:
                            logger.warning(f"Ignoring cached repo path that doesn't exist: {repo_path}")
                logger.info(f"Loaded {len(self._repos)} repository mappings from cache")
        except Exception as e:
            logger.warning(f"Error loading repository cache: {str(e)}")
    
    def _save_repos(self):
        """Save repository mapping to cache file"""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._repos, f)
            logger.info(f"Saved {len(self._repos)} repository mappings to cache")
        except Exception as e:
            logger.warning(f"Error saving repository cache: {str(e)}")
    
    def get_repo_path(self, repo_url: str) -> Optional[str]:
        """Get repository path if already cloned"""
        return self._repos.get(repo_url)
        
    def set_repo_path(self, repo_url: str, repo_path: str):
        """Set repository path after cloning"""
        self._repos[repo_url] = repo_path
        self._save_repos()
        
    def cleanup_all(self):
        """Clean up all temporary directories"""
        # Only clean up directories that are in temp location, not our persistent repos
        temp_dirs = [path for url, path in self._repos.items() if '/tmp/' in path or path.startswith('/var/folders')]
        for temp_dir in temp_dirs:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                # Remove from our mapping
                for url, path in list(self._repos.items()):
                    if path == temp_dir:
                        del self._repos[url]
        # Save the updated repo mapping
        self._save_repos()


# Simple data structures
@dataclass
class DbtModel:
    """Represents a DBT model"""
    name: str
    file_path: str
    content: str = ""
    description: str = ""
    model_type: str = "view"  # or "table", "incremental", etc.
    columns: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    yaml_path: str = ""
    yaml_content: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result"""
    model_name: str = ""
    file_path: str = ""
    content: str = ""
    match_type: str = ""  # "model", "column", "source", etc.
    match_context: str = ""  # The matched context
    description: str = ""
    yaml_content: Dict[str, Any] = field(default_factory=dict)
    related_files: List[str] = field(default_factory=list)
    schema_info: Dict[str, Any] = field(default_factory=dict)
    # Additional fields for column search
    column_name: str = ""
    calculation: str = ""
    # Additional fields for content search
    search_text: str = ""
    match_contexts: List[str] = field(default_factory=list)
    # Additional fields for file path search
    file_name: str = ""
    yaml_path: str = ""


class GitHubRepoManager:
    """Manages GitHub repository cloning and access with singleton pattern"""
    
    def __init__(self):
        self.singleton = RepoSingleton.get_instance()
        # Create a permanent directory for storing repositories
        self.repos_dir = os.path.join(os.path.expanduser("~"), ".dbt_data_architect", "repos")
        os.makedirs(self.repos_dir, exist_ok=True)
    
    def _get_auth_url(self, repo_url: str, username: str = "", token: str = "") -> str:
        """Add authentication to GitHub URL if credentials are provided"""
        if username and token:
            try:
                parsed = urlparse(repo_url)
                
                # Ensure we have a proper scheme
                if not parsed.scheme:
                    logger.warning(f"URL missing scheme, adding https://: {repo_url}")
                    repo_url = f"https://{repo_url}"
                    parsed = urlparse(repo_url)
                
                # For enterprise GitHub, we need to handle the auth differently
                # Format should be: https://username:token@hostname/path
                netloc = parsed.netloc.split('@')[-1]  # Remove any existing auth
                path = parsed.path.rstrip('/')  # Remove trailing slash
                
                # Add .git extension if not present
                if not path.endswith('.git'):
                    path = f"{path}.git"
                
                # Construct the authenticated URL with proper escaping of username and token
                safe_username = quote(username)
                safe_token = quote(token)
                auth_url = f"{parsed.scheme}://{safe_username}:{safe_token}@{netloc}{path}"
                logger.info(f"Created authenticated URL for repository (token hidden)")
                return auth_url
            except Exception as e:
                logger.error(f"Error formatting authentication URL: {str(e)}")
                return repo_url
        return repo_url
    
    def _get_repo_hash(self, repo_url: str) -> str:
        """Generate a unique hash for the repository URL"""
        import hashlib
        # Use the last part of the URL as a readable name prefix
        repo_name = repo_url.rstrip('/').split('/')[-1]
        # Add a hash suffix for uniqueness
        url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        return f"{repo_name}_{url_hash}"
    
    def _should_update_repo(self, repo_path: str) -> bool:
        """Check if the repository should be updated based on remote changes"""
        try:
            # Fetch latest changes without merging
            subprocess.run(["git", "-C", repo_path, "fetch"], check=True, capture_output=True)
            
            # Check if there are differences between local and remote
            result = subprocess.run(
                ["git", "-C", repo_path, "rev-list", "HEAD..origin/HEAD", "--count"],
                check=True, capture_output=True, text=True
            )
            
            # If count is > 0, there are changes to pull
            count = int(result.stdout.strip() or "0")
            return count > 0
        except Exception as e:
            logger.warning(f"Error checking if repo needs update: {str(e)}")
            # Default to True if we can't determine
            return True
    
    def clone_repository(self, repo_url: str, username: str = "", token: str = "") -> str:
        """Clone a GitHub repository only if not already cloned"""
        # Check if already cloned
        repo_path = self.singleton.get_repo_path(repo_url)
        if repo_path and os.path.exists(repo_path):
            logger.info(f"Using existing repository at {repo_path}")
            
            # Check if we need to update
            if self._should_update_repo(repo_path):
                # Pull latest changes
                try:
                    logger.info("Pulling latest changes...")
                    subprocess.run(["git", "-C", repo_path, "pull"], check=True, capture_output=True)
                    logger.info("Successfully pulled latest changes")
                except Exception as e:
                    logger.warning(f"Warning: Could not pull latest changes: {str(e)}")
            else:
                logger.info("Repository is already up to date, skipping pull")
                
            return repo_path
            
        # Clone new repository
        try:
            # Create a persistent directory for this repo based on its URL
            repo_hash = self._get_repo_hash(repo_url)
            persistent_dir = os.path.join(self.repos_dir, repo_hash)
            
            # Clean up if directory exists but is not a git repo
            if os.path.exists(persistent_dir):
                if not os.path.exists(os.path.join(persistent_dir, ".git")):
                    logger.info(f"Removing existing non-git directory: {persistent_dir}")
                    shutil.rmtree(persistent_dir)
                    os.makedirs(persistent_dir)
            else:
                os.makedirs(persistent_dir, exist_ok=True)
                
            logger.info(f"Cloning repository to {persistent_dir}")
            
            auth_url = self._get_auth_url(repo_url, username, token)
            
            # Clone with depth 1 for faster cloning
            subprocess.run(
                ["git", "clone", "--depth", "1", auth_url, persistent_dir], 
                check=True, 
                capture_output=True
            )
            
            logger.info(f"Successfully cloned repository to {persistent_dir}")
            self.singleton.set_repo_path(repo_url, persistent_dir)
            return persistent_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git process error: {e.stderr.decode() if e.stderr else str(e)}")
            if os.path.exists(persistent_dir) and not os.path.exists(os.path.join(persistent_dir, ".git")):
                shutil.rmtree(persistent_dir)
            raise RuntimeError(f"Failed to clone repository: {e.stderr.decode() if e.stderr else str(e)}")
            
        except Exception as e:
            logger.error(f"Error cloning repository: {str(e)}")
            if os.path.exists(persistent_dir) and not os.path.exists(os.path.join(persistent_dir, ".git")):
                shutil.rmtree(persistent_dir)
            raise


class DbtFileScanner:
    """Scans and indexes DBT project files"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.models_dir = os.path.join(repo_path, "models")
        self.model_files = {}  # Map of model name to file path
        self.yaml_files = {}   # Map of YAML file path to content
        self.schema_files = {} # Map of model name to schema file
        self.indexed = False
        
    def index_project(self) -> Dict[str, Any]:
        """Index all files in the DBT project"""
        if self.indexed:
            return {
                "model_files": self.model_files,
                "yaml_files": self.yaml_files,
                "schema_files": self.schema_files
            }
            
        logger.info(f"Indexing DBT project at {self.repo_path}")
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found at {self.models_dir}")
            return {}
            
        # Index SQL files (models)
        for root, _, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith('.sql'):
                    model_name = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    self.model_files[model_name] = rel_path
                    
                elif file.endswith(('.yml', '.yaml')):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # Load YAML content
                    try:
                        with open(file_path, 'r') as f:
                            yaml_content = yaml.safe_load(f)
                            self.yaml_files[rel_path] = yaml_content
                            
                            # Map models to their schema files
                            if yaml_content and 'models' in yaml_content:
                                for model in yaml_content['models']:
                                    if 'name' in model:
                                        self.schema_files[model['name']] = rel_path
                    except Exception as e:
                        logger.warning(f"Error loading YAML file {file_path}: {str(e)}")
        
        self.indexed = True
        logger.info(f"Indexed {len(self.model_files)} models and {len(self.yaml_files)} YAML files")
        
        return {
            "model_files": self.model_files,
            "yaml_files": self.yaml_files,
            "schema_files": self.schema_files
        }
    
    def get_model_file_path(self, model_name: str) -> Optional[str]:
        """Get file path for a model by name"""
        if not self.indexed:
            self.index_project()
            
        return self.model_files.get(model_name)
    
    def get_schema_for_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get schema information for a model"""
        if not self.indexed:
            self.index_project()
            
        schema_file = self.schema_files.get(model_name)
        if not schema_file:
            return None
            
        yaml_content = self.yaml_files.get(schema_file)
        if not yaml_content or 'models' not in yaml_content:
            return None
            
        # Find the model in the schema file
        for model in yaml_content['models']:
            if model.get('name') == model_name:
                return model
                
        return None


class DbtModelParser:
    """Parses DBT models from SQL files"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.file_scanner = DbtFileScanner(repo_path)
        
    def parse_model(self, model_name: str) -> Optional[DbtModel]:
        """Parse a model by name"""
        # Get file path
        file_path = self.file_scanner.get_model_file_path(model_name)
        if not file_path:
            logger.warning(f"Model file not found for {model_name}")
            return None
            
        full_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(full_path):
            logger.warning(f"Model file does not exist at {full_path}")
            return None
            
        # Read file content
        try:
            with open(full_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading model file: {str(e)}")
            return None
            
        # Parse the model
        model = DbtModel(
            name=model_name,
            file_path=file_path,
            content=content
        )
        
        # Extract model type from content
        model_type_match = re.search(r'{{[\s]*config\s*\(\s*materialized\s*=\s*[\'"](\w+)[\'"]', content)
        if model_type_match:
            model.model_type = model_type_match.group(1)
        
        # Extract references
        refs = re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', content)
        model.references = list(set(refs))  # Remove duplicates
        
        # Extract sources
        sources = re.findall(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', content)
        model.sources = [f"{source[0]}.{source[1]}" for source in sources]
        
        # Extract description from comments
        desc_match = re.search(r'--\s*@description:?\s*(.+?)(?:\n|$)', content)
        if desc_match:
            model.description = desc_match.group(1).strip()
            
        # Get schema information
        schema_info = self.file_scanner.get_schema_for_model(model_name)
        if schema_info:
            if 'description' in schema_info and schema_info['description']:
                model.description = schema_info['description']
                
            if 'columns' in schema_info:
                model.columns = schema_info['columns']
                
            model.yaml_path = self.file_scanner.schema_files.get(model_name, "")
            model.yaml_content = schema_info
            
        # Extract dependencies (combined refs and sources)
        model.dependencies = model.references + model.sources
            
        return model
    
    def extract_columns_from_sql(self, content: str) -> List[Dict[str, str]]:
        """Extract column definitions from SQL content"""
        columns = []
        
        # Look for select statements
        select_match = re.search(r'select\s+(.+?)(?:from|$)', content, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return columns
            
        # Extract column definitions
        select_clause = select_match.group(1).strip()
        col_defs = re.split(r',\s*(?=(?:[^\']*\'[^\']*\')*[^\']*$)', select_clause)
        
        for col_def in col_defs:
            col_def = col_def.strip()
            if not col_def:
                continue
                
            # Check for alias (as keyword)
            alias_match = re.search(r'(?:^|\s)as\s+([^\s,]+)(?:\s*(?:--|$))?', col_def, re.IGNORECASE)
            if alias_match:
                name = alias_match.group(1).strip('"\'')
                expr = col_def[:alias_match.start()].strip()
                
                # Clean up name if it has quotes
                name = re.sub(r'^["\']|["\']$', '', name)
                
                columns.append({
                    "name": name,
                    "expression": expr
                })
                
        return columns


class DbtSearcher:
    """Search tools for DBT models"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.file_scanner = DbtFileScanner(repo_path)
        self.model_parser = DbtModelParser(repo_path)
        
    def search_by_model_name(self, model_name: str) -> List[SearchResult]:
        """Search for models by name (exact or partial match)"""
        results = []
        self.file_scanner.index_project()
        
        # Try exact match first
        if model_name in self.file_scanner.model_files:
            model = self.model_parser.parse_model(model_name)
            if model:
                result = SearchResult(
                    model_name=model.name,
                    file_path=model.file_path,
                    content=model.content,
                    match_type="model",
                    match_context="exact match",
                    description=model.description,
                    yaml_content=model.yaml_content,
                    schema_info={
                        "columns": model.columns,
                        "sources": model.sources,
                        "references": model.references
                    }
                )
                results.append(result)
                return results
        
        # Try partial matches
        partial_matches = []
        for name in self.file_scanner.model_files:
            if model_name.lower() in name.lower():
                partial_matches.append(name)
        
        # Sort by closest match (most similar names first)
        partial_matches.sort(key=lambda x: difflib.SequenceMatcher(None, x.lower(), model_name.lower()).ratio(), reverse=True)
        
        # Get top 5 partial matches
        for name in partial_matches[:5]:
            model = self.model_parser.parse_model(name)
            if model:
                result = SearchResult(
                    model_name=model.name,
                    file_path=model.file_path,
                    content=model.content,
                    match_type="model",
                    match_context=f"partial match ({model_name})",
                    description=model.description,
                    yaml_content=model.yaml_content,
                    schema_info={
                        "columns": model.columns,
                        "sources": model.sources,
                        "references": model.references
                    }
                )
                results.append(result)
        
        return results
    
    def search_by_column_name(self, column_name: str) -> List[SearchResult]:
        """Search for models containing a specific column"""
        results = []
        self.file_scanner.index_project()
        
        # Check if the column name includes a table/model prefix (e.g., "order_items.amount")
        if "." in column_name:
            parts = column_name.split(".")
            if len(parts) == 2:
                model_name, col_name = parts
                # Look for this specific column in this specific model
                file_path = self.file_scanner.get_model_file_path(model_name)
                if file_path:
                    self._add_column_search_result(results, model_name, col_name, file_path)
                return results
        
        # Look through all available models
        for model_name, file_path in self.file_scanner.model_files.items():
            self._add_column_search_result(results, model_name, column_name, file_path)
            
        # Also search in schema YAML files for column definitions
        for yaml_path in self.file_scanner.yaml_files:
            full_yaml_path = os.path.join(self.repo_path, yaml_path)
            if not os.path.exists(full_yaml_path):
                continue
                
            with open(full_yaml_path, 'r') as f:
                try:
                    yaml_content = yaml.safe_load(f)
                    if not yaml_content or 'models' not in yaml_content:
                        continue
                        
                    for model in yaml_content['models']:
                        if 'columns' not in model:
                            continue
                            
                        for column in model.get('columns', []):
                            if column.get('name', '').lower() == column_name.lower():
                                model_name = model.get('name', '')
                                file_path = self.file_scanner.get_model_file_path(model_name)
                                
                                if not file_path:
                                    continue
                                    
                                result = SearchResult(
                                    model_name=model_name,
                                    file_path=file_path,
                                    match_type="column",
                                    match_context=f"column definition in schema",
                                    column_name=column_name,
                                    description=column.get('description', ''),
                                    yaml_path=yaml_path,
                                    yaml_content={
                                        "column": column,
                                        "model": model
                                    }
                                )
                                results.append(result)
                except Exception as e:
                    logger.error(f"Error parsing YAML file {yaml_path}: {str(e)}")
                    
        return results
    
    def _add_column_search_result(self, results: List[SearchResult], model_name: str, column_name: str, file_path: str):
        """Helper method to search for a column in a model and add results if found"""
        full_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(full_path):
            return
            
        # Read file content
        try:
            with open(full_path, 'r') as f:
                content = f.read()
                
            # Check for column name in content (considering typical SQL patterns)
            # This handles various cases like "select col_name", "col_name as", "as col_name", etc.
            # Case-insensitive search
            patterns = [
                # Column in SELECT statements
                rf'\b{re.escape(column_name)}\b',  
                # Column in AS statements
                rf'as\s+[\'"]?{re.escape(column_name)}[\'"]?', 
                # Column assignments
                rf'[\'"]?{re.escape(column_name)}[\'"]?\s*=',
                rf'=\s*[\'"]?{re.escape(column_name)}[\'"]?'
            ]
            
            found = False
            calculation_context = None
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                
                if matches:
                    found = True
                    # Find the calculation context for this column (if it exists)
                    for match in matches:
                        # Extract a reasonable context window around the match
                        start_pos = match.start()
                        # Try to find the start of the logical expression
                        # Go backward to find SELECT, CASE, WITH, or comma
                        context_start = max(0, start_pos - 200)
                        context_end = min(len(content), match.end() + 200)
                        
                        # Try to extract the full SQL calculation
                        if match.re.pattern.startswith(r'as\s+'):
                            # This is an "as column_name" pattern, look for the calculation before it
                            calc_text = self._extract_calculation_before_alias(content, start_pos)
                            if calc_text:
                                calculation_context = calc_text
                                break
                        elif "=" in match.re.pattern:
                            # This is a column assignment, extract the full expression
                            calc_text = self._extract_calculation_from_assignment(content, start_pos, match.end())
                            if calc_text:
                                calculation_context = calc_text
                                break
                        else:
                            # For other matches, get surrounding context
                            context_text = content[context_start:context_end]
                            if "select" in context_text.lower() or "as" in context_text.lower():
                                calculation_context = context_text
                                break
            
            if found:
                result = SearchResult(
                    model_name=model_name,
                    file_path=file_path,
                    content=content,
                    match_type="column",
                    match_context=f"column usage in model",
                    column_name=column_name,
                    calculation=calculation_context
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error searching for column {column_name} in file {file_path}: {str(e)}")
    
    def _extract_calculation_before_alias(self, content: str, alias_pos: int) -> Optional[str]:
        """Extract the calculation expression that comes before 'as column_name'"""
        # Find the relevant portion of the content before this position
        preceding_content = content[:alias_pos].strip()
        
        # Try to find the start of this expression by looking for commas, SELECT, or other clauses
        last_comma = preceding_content.rfind(',')
        last_select = preceding_content.lower().rfind('select')
        last_from = preceding_content.lower().rfind('from')
        last_join = preceding_content.lower().rfind('join')
        last_where = preceding_content.lower().rfind('where')
        
        # Find the most recent relevant SQL keyword or separator
        start_points = [p for p in [last_comma, last_select, last_from, last_join, last_where] if p >= 0]
        if not start_points:
            return None
            
        start_pos = max(start_points)
        
        # Extract the calculation part (from the last clause start to the alias position)
        calculation = preceding_content[start_pos:].strip()
        
        # Remove trailing comma if any
        if calculation.startswith(','):
            calculation = calculation[1:].strip()
            
        return calculation
    
    def _extract_calculation_from_assignment(self, content: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Extract a calculation from an assignment expression like 'column_name = expression'"""
        # Find the most reasonable context around this assignment
        preceding_content = content[:start_pos].strip()
        following_content = content[end_pos:].strip()
        
        # Find the end of the assignment (typically a comma or line end)
        next_comma = following_content.find(',')
        next_newline = following_content.find('\n')
        
        end_points = [p for p in [next_comma, next_newline] if p >= 0]
        if not end_points:
            end_of_expr = len(following_content)
        else:
            end_of_expr = min(end_points)
            
        # Get the right-hand side of the expression (after the = sign)
        expression = following_content[:end_of_expr].strip()
        
        # Include the column name and equals sign for context
        column_part = content[start_pos:end_pos].strip()
        
        return f"{column_part} {expression}"
    
    def search_by_content(self, search_text: str, file_ext: str = ".sql") -> List[SearchResult]:
        """Search for text in model content"""
        results = []
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(file_ext):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        if search_text.lower() in content.lower():
                            # Find all occurrences of the search text
                            content_lower = content.lower()
                            search_lower = search_text.lower()
                            match_indices = []
                            start_idx = 0
                            
                            while True:
                                idx = content_lower.find(search_lower, start_idx)
                                if idx == -1:
                                    break
                                match_indices.append(idx)
                                start_idx = idx + len(search_lower)
                            
                            # For each match, extract context
                            contexts = []
                            for idx in match_indices:
                                # Get context before and after
                                context_start = max(0, idx - 100)
                                context_end = min(len(content), idx + len(search_text) + 100)
                                context = content[context_start:context_end]
                                contexts.append(context)
                            
                            # Get model name from file path
                            model_name = os.path.basename(file).replace('.sql', '')
                            
                            # Check if this is a model file
                            is_model = False
                            if 'models/' in rel_path:
                                is_model = True
                            
                            result = SearchResult(
                                model_name=model_name if is_model else "",
                                file_path=rel_path,
                                content=content,
                                match_type="content",
                                match_context=f"search term found in {'model' if is_model else 'file'}",
                                search_text=search_text,
                                match_contexts=contexts
                            )
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error searching content in file {file_path}: {str(e)}")
        
        return results
    
    def search_by_file_path(self, path_pattern: str) -> List[SearchResult]:
        """Search for files by path pattern"""
        results = []
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                # Check if the path contains the pattern
                if path_pattern.lower() in rel_path.lower():
                    try:
                        # Only read content for certain file types
                        content = ""
                        if file.endswith(('.sql', '.yml', '.yaml', '.md')):
                            with open(file_path, 'r') as f:
                                content = f.read()
                        
                        # Get model name if this is a model file
                        model_name = ""
                        if file.endswith('.sql') and 'models/' in rel_path:
                            model_name = os.path.basename(file).replace('.sql', '')
                        
                        result = SearchResult(
                            model_name=model_name,
                            file_path=rel_path,
                            content=content,
                            match_type="file_path",
                            match_context=f"path contains '{path_pattern}'",
                            file_name=file
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
        
        return results
    
    def find_related_models(self, model_name: str) -> Dict[str, List[str]]:
        """Find upstream and downstream dependencies for a model"""
        model = self.model_parser.parse_model(model_name)
        if not model:
            return {"upstream": [], "downstream": []}
            
        # Upstream dependencies are direct references from this model
        upstream = model.references
        
        # Downstream dependencies are models that reference this model
        downstream = []
        
        # Scan all models to find those that reference this one
        for other_model_name in self.file_scanner.model_files:
            if other_model_name == model_name:
                continue
                
            other_model = self.model_parser.parse_model(other_model_name)
            if other_model and model_name in other_model.references:
                downstream.append(other_model_name)
        
        return {
            "upstream": upstream,
            "downstream": downstream
        }
    
    def extract_column_calculations(self, model_name: str) -> Dict[str, str]:
        """Extract calculations for all columns in a model"""
        model = self.model_parser.parse_model(model_name)
        if not model:
            return {}
            
        # Extract column calculations from the SQL content
        content = model.content
        
        # Find all column definitions in the format "calculation as column_name"
        as_pattern = re.compile(r'(.*?)\s+as\s+[\'"]?([a-zA-Z0-9_]+)[\'"]?', re.IGNORECASE)
        matches = as_pattern.finditer(content)
        
        calculations = {}
        
        for match in matches:
            calculation = match.group(1).strip()
            column_name = match.group(2).strip()
            calculations[column_name] = calculation
        
        return calculations


class DbtToolsFactory:
    """Factory class for DBT tools with repo caching"""
    
    @staticmethod
    def get_tools_from_db() -> Tuple[str, str, str]:
        """Get repository info from settings database"""
        try:
            # Connect to the database
            conn = sqlite3.connect('knowledge.db')
            cursor = conn.cursor()
            
            # Get the latest GitHub configuration
            cursor.execute("""
                SELECT value FROM settings 
                WHERE key = 'github_connectors' 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            conn.close()
            
            if result:
                configs = json.loads(result[0])
                if configs:
                    # Use the most recent configuration
                    latest_config = configs[-1]
                    repo_url = latest_config.get('repoUrl', '')
                    username = latest_config.get('username', '')
                    token = latest_config.get('token', '')
                    return repo_url, username, token
            
            return "", "", ""
            
        except Exception as e:
            logger.error(f"Error getting GitHub configuration from database: {str(e)}")
            return "", "", ""
    
    @staticmethod
    def create_dbt_tools(repo_url: str = "", username: str = "", token: str = "") -> "DbtTools":
        """Create DBT tools instance with repository"""
        # If no repo_url provided, try to get from database
        if not repo_url:
            repo_url, username, token = DbtToolsFactory.get_tools_from_db()
            
        if not repo_url:
            logger.warning("No repository URL provided or found in database")
            return None
            
        return DbtTools(repo_url, username, token)
        

class DbtTools:
    """Main class for DBT tools"""
    
    def __init__(self, repo_url: str, username: str = "", token: str = ""):
        self.repo_url = repo_url
        self.username = username
        self.token = token
        self.repo_manager = GitHubRepoManager()
        self.repo_path = None
        self.searcher = None
        self.parser = None
        self.file_scanner = None
        
    def initialize(self):
        """Initialize tools by cloning repository"""
        if self.repo_path:
            return self.repo_path
            
        try:
            self.repo_path = self.repo_manager.clone_repository(
                self.repo_url, 
                self.username, 
                self.token
            )
            
            self.searcher = DbtSearcher(self.repo_path)
            self.parser = DbtModelParser(self.repo_path)
            self.file_scanner = DbtFileScanner(self.repo_path)
            
            # Index the project
            self.file_scanner.index_project()
            
            return self.repo_path
            
        except Exception as e:
            logger.error(f"Error initializing DBT tools: {str(e)}")
            raise
    
    def search_model(self, model_name: str) -> List[SearchResult]:
        """Search for a model by name"""
        self.initialize()
        return self.searcher.search_by_model_name(model_name)
    
    def search_column(self, column_name: str) -> List[SearchResult]:
        """Search for a column by name"""
        self.initialize()
        return self.searcher.search_by_column_name(column_name)
    
    def search_content(self, search_text: str, file_ext: str = ".sql") -> List[SearchResult]:
        """Search for text in model content"""
        self.initialize()
        return self.searcher.search_by_content(search_text, file_ext)
    
    def search_file_path(self, path_pattern: str) -> List[SearchResult]:
        """Search for files by path pattern"""
        self.initialize()
        return self.searcher.search_by_file_path(path_pattern)
    
    def find_related_models(self, model_name: str) -> Dict[str, List[str]]:
        """Find upstream and downstream dependencies for a model"""
        self.initialize()
        return self.searcher.find_related_models(model_name)
    
    def extract_column_calculations(self, model_name: str) -> Dict[str, str]:
        """Extract calculations for all columns in a model"""
        self.initialize()
        return self.searcher.extract_column_calculations(model_name)
    
    def extract_specific_calculation(self, model_name: str, column_name: str) -> Optional[str]:
        """Extract the calculation for a specific column in a model"""
        calculations = self.extract_column_calculations(model_name)
        return calculations.get(column_name)
    
    def get_all_models(self) -> List[str]:
        """Get a list of all model names in the repository"""
        self.initialize()
        return list(self.file_scanner.model_files.keys())
    
    def get_all_columns(self, model_name: str = "") -> List[Dict[str, str]]:
        """Get all columns in a model or across all models"""
        self.initialize()
        
        if model_name:
            # Get columns for a specific model
            model = self.parser.parse_model(model_name)
            if not model:
                return []
            return model.columns
        else:
            # Get columns across all models
            all_columns = []
            for model_name in self.file_scanner.model_files:
                model = self.parser.parse_model(model_name)
                if model and model.columns:
                    for column in model.columns:
                        column_info = column.copy()
                        column_info["model"] = model_name
                        all_columns.append(column_info)
            return all_columns 