from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import tempfile
import subprocess
from pathlib import Path
import yaml
import json
from dataclasses import dataclass
from datetime import datetime
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class DbtModel:
    name: str
    file_path: str
    description: Optional[str] = None
    materialization: str = "view"
    columns: List[Dict[str, Any]] = None
    tests: List[Dict[str, Any]] = None
    dependencies: Dict[str, List[str]] = None
    sources: List[Dict[str, Any]] = None
    refs: List[str] = None
    tags: List[str] = None
    meta: Dict[str, Any] = None

class GitHubRepoFetcher:
    def __init__(self, username: str = "", token: str = ""):
        """Initialize GitHub repository fetcher with optional credentials."""
        self.username = username
        self.token = token
        self.temp_dir = None

    def _get_auth_url(self, repo_url: str) -> str:
        """Get authenticated GitHub URL if credentials are provided."""
        if self.username and self.token:
            parsed = urlparse(repo_url)
            return f"{parsed.scheme}://{self.username}:{self.token}@{parsed.netloc}{parsed.path}"
        return repo_url

    def clone_repository(self, repo_url: str) -> str:
        """Clone a GitHub repository to a temporary directory."""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="dbt_repo_")
            logger.info(f"Created temporary directory: {self.temp_dir}")

            # Get authenticated URL if credentials provided
            auth_url = self._get_auth_url(repo_url)

            # Clone repository
            subprocess.run(["git", "clone", auth_url, self.temp_dir], check=True)
            logger.info(f"Successfully cloned repository to {self.temp_dir}")

            return self.temp_dir

        except Exception as e:
            logger.error(f"Error cloning repository: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory")

class DbtManifestParser:
    def __init__(self, repo_path: str):
        """Initialize DBT manifest parser with repository path."""
        self.repo_path = repo_path
        self.manifest_path = os.path.join(repo_path, "target", "manifest.json")
        self.catalog_path = os.path.join(repo_path, "target", "catalog.json")
        self.manifest = None
        self.catalog = None

    def load_manifest(self) -> Dict[str, Any]:
        """Load and parse DBT manifest file."""
        try:
            if not os.path.exists(self.manifest_path):
                logger.warning(f"Manifest file not found at {self.manifest_path}")
                # Create a basic manifest
                return self._create_basic_manifest()

            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
                logger.info("Successfully loaded DBT manifest")
                return self.manifest

        except Exception as e:
            logger.error(f"Error loading manifest: {str(e)}")
            # Create a basic manifest as fallback
            return self._create_basic_manifest()

    def _create_basic_manifest(self) -> Dict[str, Any]:
        """Create a basic manifest with minimal information."""
        try:
            models_dir = os.path.join(self.repo_path, "models")
            if not os.path.exists(models_dir):
                logger.warning("Models directory not found")
                return {}

            basic_manifest = {
                "nodes": {},
                "sources": {},
                "parent_map": {},
                "child_map": {}
            }

            # Walk through models directory
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith('.sql'):
                        # Get relative path
                        rel_path = os.path.relpath(os.path.join(root, file), self.repo_path)
                        model_name = os.path.splitext(file)[0]

                        # Create basic model info
                        model_key = f"model.{model_name}"
                        basic_manifest["nodes"][model_key] = {
                            "name": model_name,
                            "path": rel_path,
                            "description": "",
                            "config": {"materialized": "view"},
                            "columns": [],
                            "tests": [],
                            "depends_on": {"nodes": []},
                            "child_map": []
                        }

            # Save the basic manifest
            os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
            with open(self.manifest_path, 'w') as f:
                json.dump(basic_manifest, f, indent=2)

            logger.info("Created basic manifest file")
            return basic_manifest

        except Exception as e:
            logger.error(f"Error creating basic manifest: {str(e)}")
            return {}

    def load_catalog(self) -> Dict[str, Any]:
        """Load and parse DBT catalog file."""
        try:
            if not os.path.exists(self.catalog_path):
                logger.warning(f"Catalog file not found at {self.catalog_path}")
                return {}

            with open(self.catalog_path, 'r') as f:
                self.catalog = json.load(f)
                logger.info("Successfully loaded DBT catalog")
                return self.catalog

        except Exception as e:
            logger.error(f"Error loading catalog: {str(e)}")
            return {}

    def get_model_info(self, model_name: str) -> Optional[DbtModel]:
        """Get detailed information about a specific model."""
        if not self.manifest:
            self.load_manifest()

        try:
            # Find model in manifest
            model_data = self.manifest.get('nodes', {}).get(f"model.{model_name}")
            if not model_data:
                logger.warning(f"Model {model_name} not found in manifest")
                return None

            # Create DbtModel instance
            model = DbtModel(
                name=model_name,
                file_path=model_data.get('path', ''),
                description=model_data.get('description', ''),
                materialization=model_data.get('config', {}).get('materialized', 'view'),
                columns=model_data.get('columns', []),
                tests=model_data.get('tests', []),
                dependencies={
                    'parents': model_data.get('depends_on', {}).get('nodes', []),
                    'children': model_data.get('child_map', [])
                },
                sources=model_data.get('sources', []),
                refs=model_data.get('refs', []),
                tags=model_data.get('tags', []),
                meta=model_data.get('meta', {})
            )

            return model

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None

class DbtModelEnhancer:
    def __init__(self, repo_path: str):
        """Initialize DBT model enhancer with repository path."""
        self.repo_path = repo_path
        self.manifest_parser = DbtManifestParser(repo_path)

    def get_model_sql(self, model_path: str) -> str:
        """Get SQL content of a model file."""
        try:
            full_path = os.path.join(self.repo_path, model_path)
            with open(full_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading model SQL: {str(e)}")
            return ""

    def get_model_schema(self, model_name: str) -> Dict[str, Any]:
        """Get schema information for a model."""
        try:
            schema_path = os.path.join(self.repo_path, "models", "schema.yml")
            if not os.path.exists(schema_path):
                return {}

            with open(schema_path, 'r') as f:
                schema_data = yaml.safe_load(f)

            # Find model in schema
            for model in schema_data.get('models', []):
                if model.get('name') == model_name:
                    return model

            return {}

        except Exception as e:
            logger.error(f"Error getting model schema: {str(e)}")
            return {}

    def enhance_model(self, model_name: str) -> Dict[str, Any]:
        """Enhance a model with additional context and information."""
        try:
            # Get basic model info
            model = self.manifest_parser.get_model_info(model_name)
            if not model:
                return {}

            # Get SQL content
            sql_content = self.get_model_sql(model.file_path)

            # Get schema information
            schema_info = self.get_model_schema(model_name)

            # Get catalog information
            catalog = self.manifest_parser.load_catalog()
            catalog_info = catalog.get('nodes', {}).get(f"model.{model_name}", {})

            # Build enhanced model information
            enhanced_model = {
                'name': model.name,
                'file_path': model.file_path,
                'description': model.description,
                'materialization': model.materialization,
                'columns': model.columns,
                'tests': model.tests,
                'dependencies': model.dependencies,
                'sources': model.sources,
                'refs': model.refs,
                'tags': model.tags,
                'meta': model.meta,
                'sql_content': sql_content,
                'schema_info': schema_info,
                'catalog_info': catalog_info,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(self.repo_path, model.file_path))).isoformat()
            }

            return enhanced_model

        except Exception as e:
            logger.error(f"Error enhancing model: {str(e)}")
            return {}

class DbtDependencyTracer:
    def __init__(self, repo_path: str):
        """Initialize DBT dependency tracer with repository path."""
        self.repo_path = repo_path
        self.manifest_parser = DbtManifestParser(repo_path)

    def get_upstream_models(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all upstream models for a given model."""
        try:
            model = self.manifest_parser.get_model_info(model_name)
            if not model:
                return []

            upstream_models = []
            for parent in model.dependencies.get('parents', []):
                if parent.startswith('model.'):
                    parent_name = parent.split('.')[-1]
                    parent_model = self.manifest_parser.get_model_info(parent_name)
                    if parent_model:
                        upstream_models.append({
                            'name': parent_name,
                            'file_path': parent_model.file_path,
                            'description': parent_model.description
                        })

            return upstream_models

        except Exception as e:
            logger.error(f"Error getting upstream models: {str(e)}")
            return []

    def get_downstream_models(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all downstream models for a given model."""
        try:
            model = self.manifest_parser.get_model_info(model_name)
            if not model:
                return []

            downstream_models = []
            for child in model.dependencies.get('children', []):
                if child.startswith('model.'):
                    child_name = child.split('.')[-1]
                    child_model = self.manifest_parser.get_model_info(child_name)
                    if child_model:
                        downstream_models.append({
                            'name': child_name,
                            'file_path': child_model.file_path,
                            'description': child_model.description
                        })

            return downstream_models

        except Exception as e:
            logger.error(f"Error getting downstream models: {str(e)}")
            return []

    def get_model_lineage(self, model_name: str) -> Dict[str, Any]:
        """Get complete lineage information for a model."""
        try:
            upstream = self.get_upstream_models(model_name)
            downstream = self.get_downstream_models(model_name)

            return {
                'model': model_name,
                'upstream': upstream,
                'downstream': downstream,
                'total_upstream': len(upstream),
                'total_downstream': len(downstream)
            }

        except Exception as e:
            logger.error(f"Error getting model lineage: {str(e)}")
            return {}

class DbtSearchTools:
    def __init__(self, repo_url: str, username: str = "", token: str = ""):
        """Initialize DBT search tools with repository URL and optional credentials."""
        self.repo_url = repo_url
        self.repo_fetcher = GitHubRepoFetcher(username, token)
        self.repo_path = None
        self.model_enhancer = None
        self.dependency_tracer = None

    def initialize(self):
        """Initialize the search tools by cloning the repository and generating manifest."""
        try:
            self.repo_path = self.repo_fetcher.clone_repository(self.repo_url)
            
            # Generate manifest if it doesn't exist
            self._generate_manifest()
            
            self.model_enhancer = DbtModelEnhancer(self.repo_path)
            self.dependency_tracer = DbtDependencyTracer(self.repo_path)
            logger.info("Successfully initialized DBT search tools")
        except Exception as e:
            logger.error(f"Error initializing DBT search tools: {str(e)}")
            raise

    def _generate_manifest(self):
        """Generate DBT manifest file if it doesn't exist."""
        try:
            manifest_path = os.path.join(self.repo_path, "target", "manifest.json")
            
            if not os.path.exists(manifest_path):
                logger.info("Manifest file not found, generating it...")
                
                # Create target directory if it doesn't exist
                target_dir = os.path.join(self.repo_path, "target")
                os.makedirs(target_dir, exist_ok=True)
                
                # Run dbt compile to generate manifest
                try:
                    subprocess.run(
                        ["dbt", "compile"],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True
                    )
                    logger.info("Successfully generated DBT manifest")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running dbt compile: {e.stderr.decode()}")
                    # Create a basic manifest if dbt compile fails
                    self._create_basic_manifest()
            else:
                logger.info("Manifest file already exists")
                
        except Exception as e:
            logger.error(f"Error generating manifest: {str(e)}")
            # Create a basic manifest as fallback
            self._create_basic_manifest()

    def _create_basic_manifest(self):
        """Create a basic manifest file with minimal information."""
        try:
            manifest_path = os.path.join(self.repo_path, "target", "manifest.json")
            
            # Find all SQL files in the models directory
            models_dir = os.path.join(self.repo_path, "models")
            if not os.path.exists(models_dir):
                logger.warning("Models directory not found")
                return
                
            basic_manifest = {
                "nodes": {},
                "sources": {},
                "parent_map": {},
                "child_map": {}
            }
            
            # Walk through models directory
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith('.sql'):
                        # Get relative path
                        rel_path = os.path.relpath(os.path.join(root, file), self.repo_path)
                        model_name = os.path.splitext(file)[0]
                        
                        # Create basic model info
                        model_key = f"model.{model_name}"
                        basic_manifest["nodes"][model_key] = {
                            "name": model_name,
                            "path": rel_path,
                            "description": "",
                            "config": {"materialized": "view"},
                            "columns": [],
                            "tests": [],
                            "depends_on": {"nodes": []},
                            "child_map": []
                        }
            
            # Save the basic manifest
            with open(manifest_path, 'w') as f:
                json.dump(basic_manifest, f, indent=2)
                
            logger.info("Created basic manifest file")
            
        except Exception as e:
            logger.error(f"Error creating basic manifest: {str(e)}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.repo_fetcher:
            self.repo_fetcher.cleanup()

    def search_model(self, model_name: str) -> Dict[str, Any]:
        """Search for a specific DBT model and get enhanced information."""
        try:
            if not self.repo_path:
                self.initialize()

            # Get enhanced model information
            model_info = self.model_enhancer.enhance_model(model_name)
            if not model_info:
                return {
                    "status": "error",
                    "message": f"Model {model_name} not found"
                }

            # Get lineage information
            lineage = self.dependency_tracer.get_model_lineage(model_name)

            return {
                "status": "success",
                "model_info": model_info,
                "lineage": lineage
            }

        except Exception as e:
            logger.error(f"Error searching model: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def search_by_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for DBT models matching a pattern."""
        try:
            if not self.repo_path:
                self.initialize()

            # Get all model files
            model_files = []
            for root, _, files in os.walk(os.path.join(self.repo_path, "models")):
                for file in files:
                    if file.endswith('.sql'):
                        model_files.append(os.path.join(root, file))

            # Filter models matching pattern
            matching_models = []
            for file_path in model_files:
                model_name = os.path.splitext(os.path.basename(file_path))[0]
                if re.search(pattern, model_name, re.IGNORECASE):
                    model_info = self.model_enhancer.enhance_model(model_name)
                    if model_info:
                        matching_models.append(model_info)

            return matching_models

        except Exception as e:
            logger.error(f"Error searching by pattern: {str(e)}")
            return []

    def get_model_dependencies(self, model_name: str) -> Dict[str, Any]:
        """Get complete dependency information for a model."""
        try:
            if not self.repo_path:
                self.initialize()

            # Get upstream and downstream models
            upstream = self.dependency_tracer.get_upstream_models(model_name)
            downstream = self.dependency_tracer.get_downstream_models(model_name)

            # Get model info
            model_info = self.model_enhancer.enhance_model(model_name)

            return {
                "status": "success",
                "model": model_name,
                "model_info": model_info,
                "upstream": upstream,
                "downstream": downstream
            }

        except Exception as e:
            logger.error(f"Error getting model dependencies: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 