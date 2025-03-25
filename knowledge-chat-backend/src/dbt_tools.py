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

    def _safe_load_yaml(self, file_path: str) -> Optional[Dict]:
        """Safely load a YAML file with error handling"""
        try:
            with open(file_path, 'r') as f:
                # First, read the content and clean it
                content = f.read()
                # Remove any invalid characters that might cause parsing issues
                content = content.replace('\t', '    ')  # Replace tabs with spaces
                content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII chars
                # Parse the cleaned content
                return yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning(f"YAML parsing error in {file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Error reading YAML file {file_path}: {str(e)}")
            return None


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
    """Scans and indexes DBT project files with enhanced enterprise capabilities"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.models_dir = os.path.join(repo_path, "models")
        self.model_files = {}  # Map of model name to file path
        self.yaml_files = {}   # Map of YAML file path to content
        self.schema_files = {} # Map of model name to schema file
        self.schema_by_path = {} # Map of model file path to schema file
        self.model_relationships = {} # Map of model name to related models
        self.source_mapping = {} # Map of sources to models
        self.model_by_path = {} # Map of file path to model name
        self.project_structure = {} # Tree structure of the project
        self.indexed = False
        self._yaml_cache = {}  # Cache for parsed YAML files
        self._model_content_cache = {} # Cache for model content
        self._source_cache = {} # Cache for source definitions
        self._common_path_prefixes = [] # Store common path prefixes for better searching
        
    def _clean_yaml_content(self, content: str) -> str:
        """Clean YAML content before parsing"""
        # Replace tabs with spaces
        content = content.replace('\t', '    ')
        # Remove BOM if present
        content = content.replace('\ufeff', '')
        # Remove any other non-ASCII characters
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        # Remove trailing spaces
        content = '\n'.join(line.rstrip() for line in content.split('\n'))
        return content
        
    def _safe_load_yaml(self, file_path: str) -> Optional[Dict]:
        """Safely load a YAML file with error handling and caching"""
        if file_path in self._yaml_cache:
            return self._yaml_cache[file_path]
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # First, read and clean the content
                content = f.read()
                content = self._clean_yaml_content(content)
                
                # Try to parse the YAML
                yaml_content = yaml.safe_load(content)
                
                # Cache the result
                self._yaml_cache[file_path] = yaml_content
                return yaml_content
                
        except yaml.YAMLError as e:
            logger.warning(f"YAML parsing error in {file_path}: {str(e)}")
            # Cache the failure to avoid retrying
            self._yaml_cache[file_path] = None
            return None
        except Exception as e:
            logger.warning(f"Error reading YAML file {file_path}: {str(e)}")
            self._yaml_cache[file_path] = None
            return None
            
    def _extract_project_structure(self):
        """Build a hierarchical representation of the project structure"""
        structure = {}
        
        # Start from models directory
        if os.path.exists(self.models_dir):
            for root, dirs, files in os.walk(self.models_dir):
                rel_path = os.path.relpath(root, self.repo_path)
                current_node = structure
                
                # Create the path hierarchy
                path_parts = rel_path.split(os.sep)
                for part in path_parts:
                    if part not in current_node:
                        current_node[part] = {"files": [], "dirs": {}}
                    current_node = current_node[part]["dirs"]
                
                # Add files
                current_node["files"] = [f for f in files if f.endswith(('.sql', '.yml', '.yaml'))]
                
        # Store the structure
        self.project_structure = structure
        
        # Extract common path prefixes for better searching
        self._extract_common_prefixes()
    
    def _extract_common_prefixes(self):
        """Extract common path prefixes to aid in file path resolution"""
        prefixes = set()
        
        for path in self.model_files.values():
            parts = path.split('/')
            # Add all parent directories
            for i in range(1, len(parts)):
                prefixes.add('/'.join(parts[:i]))
        
        # Only keep prefixes that appear multiple times
        prefix_counts = {}
        for path in self.model_files.values():
            for prefix in prefixes:
                if path.startswith(prefix):
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        # Sort prefixes by frequency, most common first
        self._common_path_prefixes = sorted(
            [p for p, c in prefix_counts.items() if c > 1],
            key=lambda p: prefix_counts[p],
            reverse=True
        )
        
        logger.info(f"Extracted {len(self._common_path_prefixes)} common path prefixes")
    
    def index_project(self) -> Dict[str, Any]:
        """Index all files in the DBT project with enhanced relationship tracking"""
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
            # Try to find models directory in case it's not directly at the top level
            for root, dirs, _ in os.walk(self.repo_path):
                if "models" in dirs:
                    self.models_dir = os.path.join(root, "models")
                    logger.info(f"Found models directory at {self.models_dir}")
                    break
            
            if not os.path.exists(self.models_dir):
                return {}
            
        # First pass: collect all SQL and YAML files
        for root, _, files in os.walk(self.models_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                if file.endswith('.sql'):
                    model_name = os.path.splitext(file)[0]
                    self.model_files[model_name] = rel_path
                    self.model_by_path[rel_path] = model_name
                    
                elif file.endswith(('.yml', '.yaml')):
                    # Load and validate YAML content
                    yaml_content = self._safe_load_yaml(file_path)
                    if yaml_content:
                        self.yaml_files[rel_path] = yaml_content
        
        # Second pass: process YAML files to map schemas to models
        for yaml_path, yaml_content in self.yaml_files.items():
            if not yaml_content:
                continue
                
            # Process model definitions
            if isinstance(yaml_content, dict) and 'models' in yaml_content:
                for model in yaml_content['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_name = model['name']
                        self.schema_files[model_name] = yaml_path
                        
                        # Map schema to model file path
                        if model_name in self.model_files:
                            model_path = self.model_files[model_name]
                            self.schema_by_path[model_path] = yaml_path
            
            # Process source definitions
            if isinstance(yaml_content, dict) and 'sources' in yaml_content:
                for source in yaml_content['sources']:
                    if isinstance(source, dict) and 'name' in source and 'tables' in source:
                        source_name = source['name']
                        
                        for table in source['tables']:
                            if isinstance(table, dict) and 'name' in table:
                                table_name = table['name']
                                source_ref = f"{source_name}.{table_name}"
                                
                                # Store source definition
                                self._source_cache[source_ref] = {
                                    'source': source_name,
                                    'table': table_name,
                                    'definition': table,
                                    'yaml_path': yaml_path
                                }
        
        # Third pass: Scan model SQL files to extract relationships
        for model_name, model_path in self.model_files.items():
            full_path = os.path.join(self.repo_path, model_path)
            
            # Read file content with caching
            if full_path not in self._model_content_cache:
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        self._model_content_cache[full_path] = content
                except Exception as e:
                    logger.error(f"Error reading model file {model_path}: {str(e)}")
                    self._model_content_cache[full_path] = ""
            
            content = self._model_content_cache[full_path]
            
            # Extract references (models this model depends on)
            refs = re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', content)
            unique_refs = list(set(refs))
            
            # Extract sources this model uses
            sources = re.findall(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', content)
            unique_sources = [f"{source[0]}.{source[1]}" for source in sources]
            
            # Store relationships
            self.model_relationships[model_name] = {
                "refs": unique_refs,
                "sources": unique_sources
            }
            
            # Update source mapping (which models use which sources)
            for source_ref in unique_sources:
                if source_ref not in self.source_mapping:
                    self.source_mapping[source_ref] = []
                if model_name not in self.source_mapping[source_ref]:
                    self.source_mapping[source_ref].append(model_name)
        
        # Build project structure for better navigation
        self._extract_project_structure()
        
        self.indexed = True
        logger.info(f"Indexed {len(self.model_files)} models and {len(self.yaml_files)} YAML files")
        logger.info(f"Mapped {len(self.schema_files)} schema definitions and {len(self.source_mapping)} sources")
        
        return {
            "model_files": self.model_files,
            "yaml_files": self.yaml_files,
            "schema_files": self.schema_files
        }
    
    def get_model_file_path(self, model_name: str) -> Optional[str]:
        """Get the file path for a model, handling both exact and partial matches"""
        if not self.indexed:
            self.index_project()
            
        # Try exact match first
        if model_name in self.model_files:
            return self.model_files[model_name]
            
        # Try case-insensitive match
        model_name_lower = model_name.lower()
        for name, path in self.model_files.items():
            if name.lower() == model_name_lower:
                return path
                
        # Try matching against full path
        for path in self.model_files.values():
            if model_name in path or model_name_lower in path.lower():
                return path
        
        # Try smarter path matching using common prefixes
        if '/' in model_name:
            model_path = model_name
            # If it ends with .sql, remove it
            if model_path.endswith('.sql'):
                model_path = model_path[:-4]
                
            # Try each common prefix as a potential root
            for prefix in self._common_path_prefixes:
                # If the model path doesn't already start with the prefix
                if not model_path.startswith(prefix):
                    test_path = f"{prefix}/{model_path}"
                    # Check if this constructed path exists
                    for path in self.model_files.values():
                        if test_path in path:
                            return path
        
        return None
    
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
        
    def get_schema_for_model_path(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Get schema information for a model by its file path"""
        if not self.indexed:
            self.index_project()
            
        # If we have a direct mapping, use it
        if model_path in self.schema_by_path:
            schema_file = self.schema_by_path[model_path]
            yaml_content = self.yaml_files.get(schema_file)
            
            if not yaml_content or 'models' not in yaml_content:
                return None
                
            # Get the model name from the path
            model_name = self.model_by_path.get(model_path)
            if not model_name:
                return None
                
            # Find the model in the schema file
            for model in yaml_content['models']:
                if model.get('name') == model_name:
                    return model
        
        # If no direct mapping, try to get the model name and use that
        model_name = self.model_by_path.get(model_path)
        if model_name:
            return self.get_schema_for_model(model_name)
            
        return None
        
    def get_source_definition(self, source_name: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get source definition by name and table"""
        if not self.indexed:
            self.index_project()
            
        source_ref = f"{source_name}.{table_name}"
        return self._source_cache.get(source_ref)
        
    def get_model_relationships(self) -> Dict[str, Dict[str, List[str]]]:
        """Get all model relationships in the project"""
        if not self.indexed:
            self.index_project()
            
        return self.model_relationships
        
    def get_upstream_models(self, model_name: str) -> List[str]:
        """Get models that this model directly depends on"""
        if not self.indexed:
            self.index_project()
            
        if model_name not in self.model_relationships:
            return []
            
        return self.model_relationships[model_name]["refs"]
        
    def get_downstream_models(self, model_name: str) -> List[str]:
        """Get models that directly depend on this model"""
        if not self.indexed:
            self.index_project()
            
        downstream = []
        for other_model, relationships in self.model_relationships.items():
            if model_name in relationships["refs"]:
                downstream.append(other_model)
                
        return downstream
        
    def get_model_sources(self, model_name: str) -> List[str]:
        """Get sources that this model directly depends on"""
        if not self.indexed:
            self.index_project()
            
        if model_name not in self.model_relationships:
            return []
            
        return self.model_relationships[model_name]["sources"]
        
    def get_model_lineage(self, model_name: str, depth: int = 3) -> Dict[str, Any]:
        """Get the full lineage for a model (upstream and downstream) up to a certain depth"""
        if not self.indexed:
            self.index_project()
            
        def get_upstream_recursive(model: str, current_depth: int, visited: Set[str]) -> Dict[str, Any]:
            if current_depth <= 0 or model in visited:
                return {"model": model, "upstream": []}
                
            visited.add(model)
            upstream_models = self.get_upstream_models(model)
            upstream_sources = self.get_model_sources(model)
            
            upstream = []
            for upstream_model in upstream_models:
                upstream.append(get_upstream_recursive(upstream_model, current_depth - 1, visited.copy()))
                
            return {
                "model": model,
                "upstream": upstream,
                "sources": upstream_sources
            }
            
        def get_downstream_recursive(model: str, current_depth: int, visited: Set[str]) -> Dict[str, Any]:
            if current_depth <= 0 or model in visited:
                return {"model": model, "downstream": []}
                
            visited.add(model)
            downstream_models = self.get_downstream_models(model)
            
            downstream = []
            for downstream_model in downstream_models:
                downstream.append(get_downstream_recursive(downstream_model, current_depth - 1, visited.copy()))
                
            return {
                "model": model,
                "downstream": downstream
            }
            
        upstream_lineage = get_upstream_recursive(model_name, depth, set())
        downstream_lineage = get_downstream_recursive(model_name, depth, set())
        
        return {
            "model": model_name,
            "upstream": upstream_lineage["upstream"],
            "sources": upstream_lineage.get("sources", []),
            "downstream": downstream_lineage["downstream"]
        }
        
    def search_models_by_path_pattern(self, path_pattern: str) -> List[Tuple[str, str]]:
        """Search for models by a path pattern, returning [(model_name, path)]"""
        if not self.indexed:
            self.index_project()
            
        results = []
        pattern_lower = path_pattern.lower()
        
        for model_name, path in self.model_files.items():
            if pattern_lower in path.lower():
                results.append((model_name, path))
                
        # Sort by relevance - exact matches first, then path similarity
        results.sort(
            key=lambda x: (
                # Exact path match gets highest score
                1 if x[1] == path_pattern else 0,
                # Then check for substring matches
                1 if path_pattern in x[1] else 0,
                # Then use sequence matcher for similarity
                difflib.SequenceMatcher(None, x[1], path_pattern).ratio()
            ),
            reverse=True
        )
        
        return results


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
        self._column_cache = {}  # Cache for column search results
        
    def search_by_model_name(self, model_name: str, search_mode: str = "parse") -> List[SearchResult]:
        """
        Search for models by name or path (exact or partial match)
        
        Args:
            model_name: The model name or path to search for
            search_mode: 'parse' for broader matching (question parsing) or 'output' for precise matching (final response)
        """
        results = []
        self.file_scanner.index_project()
        
        # Handle full path search
        if '/' in model_name:
            # Clean up the path
            clean_path = model_name.strip('/')
            if clean_path.endswith('.sql'):
                clean_path = clean_path[:-4]
            
            # Try exact path match first
            exact_path_matches = []
            for name, path in self.file_scanner.model_files.items():
                if clean_path in path:
                    model = self.model_parser.parse_model(name)
                    if model:
                        result = SearchResult(
                            model_name=model.name,
                            file_path=model.file_path,
                            content=model.content,
                            match_type="model",
                            match_context=f"path match: {clean_path}",
                            description=model.description,
                            yaml_content=model.yaml_content,
                            schema_info={
                                "columns": model.columns,
                                "sources": model.sources,
                                "references": model.references
                            }
                        )
                        exact_path_matches.append(result)
                        logger.info(f"Found path match: {path}")
            
            # Sort exact path matches by path similarity for better precision
            if exact_path_matches:
                # Sort by similarity score
                exact_path_matches.sort(
                    key=lambda x: difflib.SequenceMatcher(None, x.file_path, clean_path).ratio(),
                    reverse=True
                )
                
                # For 'output' mode, just return the best match
                if search_mode == "output" and exact_path_matches:
                    return [exact_path_matches[0]]
                    
                # For 'parse' mode or if output mode still wants more results
                results.extend(exact_path_matches)
                if results:
                    return results
            
            # If no exact path match, try more aggressive file path searching
            if not results:
                file_path_results = self.search_by_file_path(model_name)
                if file_path_results:
                    logger.info(f"Found {len(file_path_results)} files matching path pattern: {model_name}")
                    
                    # Convert file search results to model search results
                    for file_result in file_path_results:
                        if file_result.file_path.endswith('.sql'):
                            # Extract model name from path
                            model_name = os.path.splitext(os.path.basename(file_result.file_path))[0]
                            
                            # Try to parse as a model
                            model = self.model_parser.parse_model(model_name)
                            if model:
                                result = SearchResult(
                                    model_name=model.name,
                                    file_path=model.file_path,
                                    content=model.content,
                                    match_type="model",
                                    match_context=f"path match via file search: {file_result.file_path}",
                                    description=model.description,
                                    yaml_content=model.yaml_content,
                                    schema_info={
                                        "columns": model.columns,
                                        "sources": model.sources,
                                        "references": model.references
                                    }
                                )
                                results.append(result)
                            else:
                                # If model parsing fails, use the file result directly
                                file_result.match_type = "model"
                                file_result.model_name = model_name
                                results.append(file_result)
                    
                    # For output mode, limit to top result
                    if search_mode == "output" and results:
                        return [results[0]]
                    
                    # For parse mode, return all matches
                    if results:
                        return results
            
            # If no file path matches, try matching parts of the path
            path_parts = clean_path.split('/')
            model_name = path_parts[-1]  # Use the last part as model name
        
        # Try exact model name match
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
        for name, path in self.file_scanner.model_files.items():
            # Check both name and path for matches
            if (model_name.lower() in name.lower() or 
                model_name.lower() in path.lower()):
                partial_matches.append((name, path))
        
        # Sort by closest match using path similarity
        partial_matches.sort(
            key=lambda x: (
                difflib.SequenceMatcher(None, x[1].lower(), model_name.lower()).ratio(),
                difflib.SequenceMatcher(None, x[0].lower(), model_name.lower()).ratio()
            ),
            reverse=True
        )
        
        # Limit results based on search mode
        limit = 1 if search_mode == "output" else 5
        
        # Get top matches
        for name, _ in partial_matches[:limit]:
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
        
        # If still no results and we're in output mode, try file search as a last resort
        if not results and search_mode == "output":
            logger.info(f"No model results found, trying file search for {model_name}")
            file_path_results = self.search_by_file_path(model_name)
            if file_path_results:
                for file_result in file_path_results:
                    if file_result.file_path.endswith('.sql'):
                        file_result.match_type = "file"
                        if not file_result.model_name:
                            file_result.model_name = os.path.splitext(os.path.basename(file_result.file_path))[0]
                        results.append(file_result)
            
            # For output mode, limit to top result
            if results:
                return [results[0]]
        
        return results
    
    def search_by_column_name(self, column_name: str) -> List[SearchResult]:
        """Search for models containing a specific column with caching"""
        # Check cache first
        if column_name in self._column_cache:
            return self._column_cache[column_name]
            
        results = []
        self.file_scanner.index_project()
        
        # Normalize column name for comparison
        normalized_col = column_name.lower().strip()
        
        # Check if the column name includes a table/model prefix
        if "." in normalized_col:
            parts = normalized_col.split(".")
            if len(parts) == 2:
                model_name, col_name = parts
                # Look for this specific column in this specific model
                file_path = self.file_scanner.get_model_file_path(model_name)
                if file_path:
                    self._add_column_search_result(results, model_name, col_name, file_path)
                self._column_cache[column_name] = results
                return results
        
        # Look through all available models
        for model_name, file_path in self.file_scanner.model_files.items():
            self._add_column_search_result(results, model_name, normalized_col, file_path)
            
        # Also search in schema YAML files for column definitions
        for yaml_path, yaml_content in self.file_scanner.yaml_files.items():
            if not yaml_content or not isinstance(yaml_content, dict) or 'models' not in yaml_content:
                continue
                
            for model in yaml_content['models']:
                if not isinstance(model, dict) or 'columns' not in model:
                    continue
                    
                for column in model.get('columns', []):
                    if not isinstance(column, dict):
                        continue
                        
                    col_name = column.get('name', '').lower().strip()
                    if col_name == normalized_col:
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
        
        # Cache the results
        self._column_cache[column_name] = results
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
        """Search for text in model content using efficient chunked reading"""
        results = []
        chunk_size = 8192  # 8KB chunks
        
        def search_file(file_path: str, rel_path: str) -> Optional[SearchResult]:
            try:
                matches = []
                with open(file_path, 'r') as f:
                    # Read file in chunks for memory efficiency
                    content = ""
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        content += chunk
                        
                        # If we've accumulated enough content or reached EOF
                        if len(content) >= chunk_size * 2 or not chunk:
                            if search_text.lower() in content.lower():
                                # Find all occurrences
                                content_lower = content.lower()
                                search_lower = search_text.lower()
                                start_idx = 0
                                
                                while True:
                                    idx = content_lower.find(search_lower, start_idx)
                                    if idx == -1:
                                        break
                                        
                                    # Get context around match
                                    context_start = max(0, idx - 100)
                                    context_end = min(len(content), idx + len(search_text) + 100)
                                    context = content[context_start:context_end]
                                    matches.append(context)
                                    
                                    start_idx = idx + len(search_lower)
                            
                            # Keep the last chunk for overlap
                            content = content[-chunk_size:] if chunk else ""
                
                if matches:
                    # Get model name from file path
                    model_name = os.path.basename(file_path).replace('.sql', '')
                    
                    # Check if this is a model file
                    is_model = 'models/' in rel_path
                    
                    return SearchResult(
                        model_name=model_name if is_model else "",
                        file_path=rel_path,
                        content=content,
                        match_type="content",
                        match_context=f"search term found in {'model' if is_model else 'file'}",
                        search_text=search_text,
                        match_contexts=matches
                    )
                
            except Exception as e:
                logger.error(f"Error searching content in file {file_path}: {str(e)}")
            
            return None
        
        # First, search in model files for efficiency
        for model_name, rel_path in self.file_scanner.model_files.items():
            if not file_ext or rel_path.endswith(file_ext):
                file_path = os.path.join(self.repo_path, rel_path)
                result = search_file(file_path, rel_path)
                if result:
                    results.append(result)
        
        # Then search other files in models directory
        for root, _, files in os.walk(self.models_dir):
            for file in files:
                if not file_ext or file.endswith(file_ext):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # Skip if already processed as a model
                    if rel_path in self.file_scanner.model_files.values():
                        continue
                    
                    result = search_file(file_path, rel_path)
                    if result:
                        results.append(result)
        
        return results
    
    def search_by_file_path(self, path_pattern: str) -> List[SearchResult]:
        """Search for files by path pattern with intelligent path normalization"""
        results = []
        
        # Normalize the path pattern to handle common variations
        normalized_patterns = self._generate_path_variations(path_pattern)
        
        # Track files we've already found to avoid duplicates
        found_files = set()
        
        for pattern in normalized_patterns:
            for root, _, files in os.walk(self.repo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # Skip if we've already processed this file
                    if rel_path in found_files:
                        continue
                    
                    # Check if the path contains the pattern (case insensitive)
                    pattern_lower = pattern.lower()
                    rel_path_lower = rel_path.lower()
                    
                    if pattern_lower in rel_path_lower:
                        try:
                            # Only read content for certain file types
                            content = ""
                            if file.endswith(('.sql', '.yml', '.yaml', '.md')):
                                with open(file_path, 'r') as f:
                                    content = f.read()
                        
                            # Get model name if this is a model file
                            model_name = ""
                            if file.endswith('.sql') and 'models/' in rel_path.lower():
                                model_name = os.path.splitext(file)[0]
                        
                            result = SearchResult(
                                model_name=model_name,
                                file_path=rel_path,
                                content=content,
                                match_type="file_path",
                                match_context=f"path contains '{pattern}'",
                                file_name=file
                            )
                            results.append(result)
                            found_files.add(rel_path)
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {str(e)}")
        
        return results
    
    def _generate_path_variations(self, path_pattern: str) -> List[str]:
        """
        Generate variations of a path pattern for more robust searching
        
        This handles common path variations like:
        - singular/plural directory names (model/models)
        - different file extensions (.sql, no extension)
        - partial paths (just filename vs directory/filename)
        - common directory patterns (marts/core, models/marts/core, etc.)
        
        Args:
            path_pattern: The original path pattern to generate variations for
            
        Returns:
            List of path pattern variations to try
        """
        variations = [path_pattern]  # Always include the original
        
        # Remove .sql extension if present
        if path_pattern.endswith('.sql'):
            variations.append(path_pattern[:-4])
        else:
            variations.append(f"{path_pattern}.sql")
        
        # Normalize slashes
        normalized = path_pattern.replace('\\', '/')
        if normalized not in variations:
            variations.append(normalized)
        
        # Extract filename and try that
        if '/' in normalized:
            filename = normalized.split('/')[-1]
            variations.append(filename)
            
            # Add filename without extension
            if filename.endswith('.sql'):
                variations.append(filename[:-4])
        
        # Try with and without leading/trailing slashes
        if normalized.startswith('/'):
            variations.append(normalized[1:])
        else:
            variations.append('/' + normalized)
            
        if normalized.endswith('/'):
            variations.append(normalized[:-1])
        
        # Handle singular/plural variations of common directories
        common_dirs = [
            ('model/', 'models/'),
            ('mart/', 'marts/'),
            ('source/', 'sources/'),
            ('staging/', 'stage/'),
            ('transform/', 'transforms/'),
            ('core/', 'cores/')
        ]
        
        for singular, plural in common_dirs:
            if singular in normalized:
                variations.append(normalized.replace(singular, plural))
            if plural in normalized:
                variations.append(normalized.replace(plural, singular))
        
        # Try common directory prefixes
        if not normalized.startswith('models/') and not normalized.startswith('/models/'):
            variations.append(f"models/{normalized}")
            
        if not any(v.startswith('marts/') or v.startswith('/marts/') for v in variations):
            # Check if this looks like a mart model
            is_mart = 'dim_' in normalized or 'fact_' in normalized or 'mart' in normalized
            if is_mart:
                variations.append(f"marts/{normalized}")
                variations.append(f"models/marts/{normalized}")
        
        # Remove duplicates while preserving order
        unique_variations = []
        for v in variations:
            if v not in unique_variations:
                unique_variations.append(v)
        
        logger.info(f"Generated {len(unique_variations)} path variations for '{path_pattern}'")
        return unique_variations
    
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
        self._search_cache = {}  # Cache for search results
        
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
    
    def search_model(self, model_name: str, search_mode: str = "parse") -> List[SearchResult]:
        """
        Search for a model by name with specified search mode
        
        Args:
            model_name: The model name or path to search for
            search_mode: 'parse' for broader matching or 'output' for precise matching
        """
        self.initialize()
        
        # Cache key based on both model name and search mode
        cache_key = f"{model_name}:{search_mode}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        # Determine if this is a file path or a model name
        is_file_path = '/' in model_name or '\\' in model_name or model_name.endswith('.sql')
        
        results = []
        
        # If this appears to be a file path, try file path resolution first
        if is_file_path:
            logger.info(f"Input appears to be a file path: {model_name}")
            results = self.resolve_model_from_path(model_name, search_mode)
            
        # If no results from path resolution or it's not a path, try normal model search
        if not results:
            results = self.searcher.search_by_model_name(model_name, search_mode)
            
        # If still no results and it's a path, try more aggressive pattern matching
        if not results and is_file_path:
            path_pattern = model_name
            # Clean up path
            if path_pattern.endswith('.sql'):
                path_pattern = path_pattern[:-4]
            path_pattern = path_pattern.strip('/')
            
            # Try different path variations
            path_variations = self._generate_path_variations(path_pattern)
            for variation in path_variations:
                results = self.searcher.search_by_file_path(variation)
                if results:
                    logger.info(f"Found results using path variation: {variation}")
                    break
        
        # Cache and return results
        self._search_cache[cache_key] = results
        return results
    
    def _generate_path_variations(self, path_pattern: str) -> List[str]:
        """
        Generate variations of a path pattern for more robust searching
        
        This handles common path variations like:
        - singular/plural directory names (model/models)
        - different file extensions (.sql, no extension)
        - partial paths (just filename vs directory/filename)
        - common directory patterns (marts/core, models/marts/core, etc.)
        
        Args:
            path_pattern: The original path pattern to generate variations for
            
        Returns:
            List of path pattern variations to try
        """
        variations = [path_pattern]  # Always include the original
        
        # Remove .sql extension if present
        if path_pattern.endswith('.sql'):
            variations.append(path_pattern[:-4])
        else:
            variations.append(f"{path_pattern}.sql")
        
        # Normalize slashes
        normalized = path_pattern.replace('\\', '/')
        if normalized not in variations:
            variations.append(normalized)
        
        # Extract filename and try that
        if '/' in normalized:
            filename = normalized.split('/')[-1]
            variations.append(filename)
            
            # Add filename without extension
            if filename.endswith('.sql'):
                variations.append(filename[:-4])
        
        # Try with and without leading/trailing slashes
        if normalized.startswith('/'):
            variations.append(normalized[1:])
        else:
            variations.append('/' + normalized)
            
        if normalized.endswith('/'):
            variations.append(normalized[:-1])
        
        # Handle singular/plural variations of common directories
        common_dirs = [
            ('model/', 'models/'),
            ('mart/', 'marts/'),
            ('source/', 'sources/'),
            ('staging/', 'stage/'),
            ('transform/', 'transforms/'),
            ('core/', 'cores/')
        ]
        
        for singular, plural in common_dirs:
            if singular in normalized:
                variations.append(normalized.replace(singular, plural))
            if plural in normalized:
                variations.append(normalized.replace(plural, singular))
        
        # Try common directory prefixes
        if not normalized.startswith('models/') and not normalized.startswith('/models/'):
            variations.append(f"models/{normalized}")
            
        if not any(v.startswith('marts/') or v.startswith('/marts/') for v in variations):
            # Check if this looks like a mart model
            is_mart = 'dim_' in normalized or 'fact_' in normalized or 'mart' in normalized
            if is_mart:
                variations.append(f"marts/{normalized}")
                variations.append(f"models/marts/{normalized}")
        
        # Remove duplicates while preserving order
        unique_variations = []
        for v in variations:
            if v not in unique_variations:
                unique_variations.append(v)
        
        logger.info(f"Generated {len(unique_variations)} path variations for '{path_pattern}'")
        return unique_variations
    
    def resolve_model_from_path(self, file_path: str, search_mode: str = "output") -> List[SearchResult]:
        """
        Advanced model resolution from a file path
        
        This method tries multiple strategies to resolve a model from a file path:
        1. Exact path match
        2. Path normalization and matching
        3. Path component matching
        4. Smart file search
        """
        self.initialize()
        
        logger.info(f"Resolving model from path: {file_path}")
        results = []
        
        # Normalize path
        normalized_path = file_path.replace('\\', '/')
        if normalized_path.endswith('.sql'):
            path_without_ext = normalized_path[:-4]
        else:
            path_without_ext = normalized_path
            normalized_path = f"{normalized_path}.sql"
        
        # Step 1: Try exact path match
        if os.path.exists(os.path.join(self.repo_path, normalized_path)):
            # Get model name from path
            model_name = os.path.basename(path_without_ext)
            model = self.parser.parse_model(model_name)
            if model:
                result = SearchResult(
                    model_name=model.name,
                    file_path=normalized_path,
                    content=model.content,
                    match_type="model",
                    match_context="exact path match",
                    description=model.description,
                    yaml_content=model.yaml_content
                )
                results.append(result)
                
                # For output mode, return just this result
                if search_mode == "output":
                    return results
        
        # Step 2: Try model lookup directly using path components
        path_parts = path_without_ext.strip('/').split('/')
        model_name = path_parts[-1]  # Last component is likely the model name
        
        # Look for exact model name match
        if model_name in self.file_scanner.model_files:
            model = self.parser.parse_model(model_name)
            if model:
                result = SearchResult(
                    model_name=model.name,
                    file_path=model.file_path,
                    content=model.content,
                    match_type="model",
                    match_context="model name match from path",
                    description=model.description,
                    yaml_content=model.yaml_content
                )
                results.append(result)
        
        # Step 3: Use the file scanner's path pattern search
        if not results or search_mode != "output":
            path_results = self.file_scanner.search_models_by_path_pattern(normalized_path)
            
            for model_name, path in path_results:
                model = self.parser.parse_model(model_name)
                if model:
                    similarity = difflib.SequenceMatcher(None, path, normalized_path).ratio()
                    result = SearchResult(
                        model_name=model.name,
                        file_path=model.file_path,
                        content=model.content,
                        match_type="model",
                        match_context=f"path pattern match (similarity: {similarity:.2f})",
                        description=model.description,
                        yaml_content=model.yaml_content
                    )
                    results.append(result)
            
            # Sort results by path similarity
            if results:
                results.sort(
                    key=lambda r: difflib.SequenceMatcher(None, r.file_path, normalized_path).ratio(),
                    reverse=True
                )
                
                # For output mode, just return the best match
                if search_mode == "output" and results:
                    return [results[0]]
        
        # Step 4: If still no results, do an aggressive file search
        if not results:
            file_results = self.searcher.search_by_file_path(normalized_path)
            
            # Convert file results to model results where possible
            for file_result in file_results:
                if file_result.file_path.endswith('.sql'):
                    model_name = os.path.splitext(os.path.basename(file_result.file_path))[0]
                    model = self.parser.parse_model(model_name)
                    if model:
                        result = SearchResult(
                            model_name=model.name,
                            file_path=model.file_path,
                            content=model.content,
                            match_type="model",
                            match_context="file path search match",
                            description=model.description,
                            yaml_content=model.yaml_content
                        )
                        results.append(result)
                    else:
                        # Use the file result directly
                        file_result.model_name = model_name
                        file_result.match_type = "file"
                        results.append(file_result)
        
        return results
    
    def search_column(self, column_name: str) -> List[SearchResult]:
        """Search for a column by name"""
        self.initialize()
        
        # Cache lookup
        cache_key = f"column:{column_name}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
            
        results = self.searcher.search_by_column_name(column_name)
        
        # Cache results
        self._search_cache[cache_key] = results
        return results
    
    def search_content(self, search_text: str, file_ext: str = ".sql") -> List[SearchResult]:
        """Search for text in model content using efficient chunked reading"""
        self.initialize()
        
        # Cache lookup
        cache_key = f"content:{search_text}:{file_ext}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
            
        results = self.searcher.search_by_content(search_text, file_ext)
        
        # Cache results
        self._search_cache[cache_key] = results
        return results
    
    def search_file_path(self, path_pattern: str) -> List[SearchResult]:
        """Search for files by path pattern with intelligent path normalization"""
        self.initialize()
        
        # Cache lookup
        cache_key = f"path:{path_pattern}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
            
        results = self.searcher.search_by_file_path(path_pattern)
        
        # Cache results
        self._search_cache[cache_key] = results
        return results
    
    def find_related_models(self, model_name: str) -> Dict[str, List[str]]:
        """Find upstream and downstream dependencies for a model"""
        self.initialize()
        
        # Try to get from file scanner's precomputed relationships first
        upstream = self.file_scanner.get_upstream_models(model_name)
        downstream = self.file_scanner.get_downstream_models(model_name)
        
        if upstream or downstream:
            return {
                "upstream": upstream,
                "downstream": downstream
            }
        
        # Fall back to searcher method if not found
        return self.searcher.find_related_models(model_name)
    
    def get_model_lineage(self, model_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get model lineage with configurable depth"""
        self.initialize()
        return self.file_scanner.get_model_lineage(model_name, depth)
    
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
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get the content of a file by path
        
        Args:
            file_path: Path to the file, relative to the repo root
        
        Returns:
            File content as string, or None if file not found
        """
        self.initialize()
        
        # Handle absolute path by making it relative to repo
        if os.path.isabs(file_path):
            if file_path.startswith(self.repo_path):
                file_path = os.path.relpath(file_path, self.repo_path)
            else:
                # Not in our repo, can't read
                return None
        
        full_path = os.path.join(self.repo_path, file_path)
        
        # Try direct path first
        if os.path.exists(full_path) and os.path.isfile(full_path):
            try:
                with open(full_path, 'r') as f:
                    logger.info(f"Found file content via direct path: {file_path}")
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return None
        
        # If not found, try all common path variations
        path_variations = self._generate_path_variations(file_path)
        for variation in path_variations:
            full_variation_path = os.path.join(self.repo_path, variation)
            if os.path.exists(full_variation_path) and os.path.isfile(full_variation_path):
                try:
                    with open(full_variation_path, 'r') as f:
                        logger.info(f"Found file content via path variation: {variation}")
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading file variation {variation}: {str(e)}")
                    continue
        
        # If still not found, try file path search
        results = self.search_file_path(file_path)
        if results and len(results) > 0:
            result = results[0]  # Use first match
            
            # Check for content in result
            content = None
            if hasattr(result, 'content') and result.content:
                logger.info(f"Found file content via search_file_path content attribute: {file_path}")
                content = result.content
            elif isinstance(result, dict) and 'content' in result and result['content']:
                logger.info(f"Found file content via search_file_path dictionary content: {file_path}")
                content = result['content']
                
            if content:
                return content
            
            # If content wasn't loaded, try to load it from file_path
            result_path = None
            if hasattr(result, 'file_path') and result.file_path:
                result_path = result.file_path
            elif isinstance(result, dict) and 'file_path' in result:
                result_path = result['file_path']
                
            if result_path:
                try:
                    full_path = os.path.join(self.repo_path, result_path)
                    with open(full_path, 'r') as f:
                        logger.info(f"Found file content via search_file_path result path: {result_path}")
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading file {result_path}: {str(e)}")
        
        # Last attempt: Try model search with the basename
        model_name = os.path.basename(file_path)
        if model_name.endswith('.sql'):
            model_name = model_name[:-4]
            
        model_results = self.search_model(model_name, search_mode="output")
        if model_results and len(model_results) > 0:
            result = model_results[0]
            
            # Check for content in result
            content = None
            if hasattr(result, 'content') and result.content:
                logger.info(f"Found file content via model search: {model_name}")
                content = result.content
            elif isinstance(result, dict) and 'content' in result and result['content']:
                logger.info(f"Found file content via model search (dict): {model_name}")
                content = result['content']
                
            if content:
                return content
            
            # If content wasn't loaded, try to load it from file_path
            result_path = None
            if hasattr(result, 'file_path') and result.file_path:
                result_path = result.file_path
            elif isinstance(result, dict) and 'file_path' in result:
                result_path = result['file_path']
                
            if result_path:
                try:
                    full_path = os.path.join(self.repo_path, result_path)
                    with open(full_path, 'r') as f:
                        logger.info(f"Found file content via model search result path: {result_path}")
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading file {result_path}: {str(e)}")
        
        logger.warning(f"Could not find file content for: {file_path}")
        return None
    
    def get_model_schema_and_lineage(self, model_name_or_path: str) -> Dict[str, Any]:
        """
        Comprehensive model information gathering function that combines schema information and lineage
        
        Args:
            model_name_or_path: Either a model name or a file path
            
        Returns:
            Dictionary with model schema and lineage information
        """
        self.initialize()
        
        # Step 1: Resolve the model
        search_results = self.search_model(model_name_or_path, search_mode="output")
        if not search_results:
            logger.warning(f"No model found for {model_name_or_path}")
            return {}
        
        result = search_results[0]
        model_name = result.model_name
        file_path = result.file_path
        
        # Step 2: Get schema information
        schema_info = self.file_scanner.get_schema_for_model(model_name)
        if not schema_info:
            # Try getting by path
            schema_info = self.file_scanner.get_schema_for_model_path(file_path)
        
        # Step 3: Get lineage information
        lineage = self.get_model_lineage(model_name)
        
        # Step 4: Get column calculations
        column_calcs = self.extract_column_calculations(model_name)
        
        # Combine everything
        return {
            "model_name": model_name,
            "file_path": file_path,
            "schema": schema_info,
            "lineage": lineage,
            "column_calculations": column_calcs,
            "content": result.content
        }
    
    def search_multi_model(self, query: str, search_mode: str = "parse") -> Dict[str, List[SearchResult]]:
        """
        Intelligent search for models based on a query
        
        This method combines multiple search strategies:
        1. Search by model name/path
        2. Search by content
        3. Search by column name
        
        Args:
            query: The search query (could be model name, path, or keyword)
            search_mode: 'parse' for broader matching or 'output' for precise matching
            
        Returns:
            Dictionary with search results by category
        """
        self.initialize()
        
        # Cache key
        cache_key = f"multi:{query}:{search_mode}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        results = {
            "models": [],
            "columns": [],
            "content": []
        }
        
        # 1. Search by model name/path
        model_results = self.search_model(query, search_mode)
        if model_results:
            results["models"] = model_results
        
        # 2. If no model results or we want comprehensive results, search by content
        if not model_results or search_mode == "parse":
            content_results = self.search_content(query)
            results["content"] = content_results
        
        # 3. Try column search if the query might be a column name
        if '_' in query or '.' in query:
            column_results = self.search_column(query)
            results["columns"] = column_results
        
        # Cache results
        self._search_cache[cache_key] = results
        return results
    
    def clear_cache(self):
        """Clear the search cache"""
        self._search_cache = {} 
    
    def _generate_path_variations(self, file_path: str) -> List[str]:
        """
        Generate common variations of a file path
        
        Args:
            file_path: Original file path
            
        Returns:
            List of possible path variations
        """
        variations = []
        
        # Normalize path
        normalized_path = file_path.replace('\\', '/')
        
        # Add original normalized path if different
        if normalized_path != file_path:
            variations.append(normalized_path)
        
        # Handle SQL extension
        base_path = normalized_path
        if normalized_path.endswith('.sql'):
            base_path = normalized_path[:-4]
            variations.append(base_path)  # Without extension
        else:
            variations.append(f"{normalized_path}.sql")  # With extension
        
        # Handle models/ prefix
        if not normalized_path.startswith('models/'):
            variations.append(f"models/{normalized_path}")
            
            # Also with SQL extension if needed
            if not normalized_path.endswith('.sql'):
                variations.append(f"models/{normalized_path}.sql")
        
        # For dimension and fact tables, try common patterns
        file_name = os.path.basename(normalized_path)
        if file_name.startswith('dim_') or file_name.startswith('fact_'):
            base_name = file_name
            if base_name.endswith('.sql'):
                base_name = base_name[:-4]
                
            variations.append(f"models/marts/core/{base_name}.sql")
            variations.append(f"models/marts/{base_name}.sql")
            variations.append(f"models/core/{base_name}.sql")
            variations.append(f"models/marts/core/{base_name}")
            variations.append(f"models/core/{base_name}")
            variations.append(f"marts/core/{base_name}.sql")
            variations.append(f"marts/{base_name}.sql")
        
        # Incremental models pattern
        if 'incremental' in normalized_path:
            base_name = os.path.basename(normalized_path)
            if base_name.endswith('.sql'):
                base_name = base_name[:-4]
                
            variations.append(f"models/incremental/{base_name}.sql")
            variations.append(f"models/incremental/{base_name}")
        
        # Handle snake_case vs CamelCase
        if '_' in file_path:
            # Try camel case version
            parts = file_path.split('_')
            camel_case = parts[0] + ''.join(p.capitalize() for p in parts[1:])
            if camel_case != file_path:
                variations.append(camel_case)
                if not camel_case.endswith('.sql'):
                    variations.append(f"{camel_case}.sql")
        
        return variations