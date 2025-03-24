import unittest
import os
import tempfile
import shutil
import sys
import logging
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.dbt_tools import DbtTools, DbtToolsFactory, RepoSingleton

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDbtTools(unittest.TestCase):
    """Tests for the DBT tools implementation."""
    
    def setUp(self):
        """Set up a mock repository for testing."""
        self.test_dir = tempfile.mkdtemp(prefix="test_dbt_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create a few mock DBT model files
        self.create_mock_model("customer", "SELECT id, name, email FROM source_customers")
        self.create_mock_model("orders", "SELECT o.id, o.customer_id, o.amount FROM source_orders o")
        self.create_mock_model("order_items", "SELECT oi.order_id, oi.product_id, oi.quantity FROM source_order_items oi")
        self.create_mock_model("customer_orders", "SELECT c.id, c.name, o.id as order_id, o.amount FROM {{ ref('customer') }} c JOIN {{ ref('orders') }} o ON c.id = o.customer_id")
        
        # Create a schema YAML file
        self.create_mock_schema()
        
        # Initialize the singleton for testing
        self.repo_singleton = RepoSingleton.get_instance()
        self.repo_singleton.set_repo_path("test_repo", self.test_dir)
        
        # Create DBT tools instance
        self.dbt_tools = DbtTools("test_repo")
        self.dbt_tools.repo_path = self.test_dir
        self.dbt_tools.initialize()
        
    def tearDown(self):
        """Clean up the test directory."""
        shutil.rmtree(self.test_dir)
    
    def create_mock_model(self, name, sql_content):
        """Create a mock DBT model file."""
        model_path = os.path.join(self.models_dir, f"{name}.sql")
        with open(model_path, 'w') as f:
            f.write(f"-- @description: This is the {name} model\n")
            f.write(f"{sql_content}")
    
    def create_mock_schema(self):
        """Create a mock schema YAML file."""
        schema_path = os.path.join(self.models_dir, "schema.yml")
        schema_content = """
version: 2

models:
  - name: customer
    description: Customer information
    columns:
      - name: id
        description: Primary key
        tests:
          - unique
          - not_null
      - name: name
        description: Customer name
      - name: email
        description: Customer email address
        
  - name: orders
    description: Order details
    columns:
      - name: id
        description: Order ID
        tests:
          - unique
          - not_null
      - name: customer_id
        description: Foreign key to customer
        tests:
          - not_null
      - name: amount
        description: Order amount
        
  - name: customer_orders
    description: Customers with their orders
    columns:
      - name: id
        description: Customer ID
      - name: name
        description: Customer name
      - name: order_id
        description: Order ID
      - name: amount
        description: Order amount
"""
        with open(schema_path, 'w') as f:
            f.write(schema_content)
    
    def test_file_scanner(self):
        """Test that the file scanner correctly indexes the project."""
        index_result = self.dbt_tools.file_scanner.index_project()
        
        # Verify model files are indexed
        self.assertIn("customer", index_result["model_files"])
        self.assertIn("orders", index_result["model_files"])
        self.assertIn("order_items", index_result["model_files"])
        self.assertIn("customer_orders", index_result["model_files"])
        
        # Verify YAML files are indexed
        yaml_paths = list(index_result["yaml_files"].keys())
        self.assertTrue(any("schema.yml" in path for path in yaml_paths))
        
        # Verify schema mappings
        self.assertIn("customer", index_result["schema_files"])
        self.assertIn("orders", index_result["schema_files"])
        self.assertIn("customer_orders", index_result["schema_files"])
    
    def test_search_model(self):
        """Test searching for a model by name."""
        result = self.dbt_tools.search_model("customer")
        
        # Verify the result is successful
        self.assertEqual(result["status"], "success")
        
        # Verify the right model was found
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["model_name"], "customer")
        
        # Verify the content was retrieved
        self.assertIn("SELECT id, name, email FROM source_customers", result["results"][0]["content"])
    
    def test_search_column(self):
        """Test searching for a column."""
        result = self.dbt_tools.search_column("amount")
        
        # Verify the result is successful
        self.assertEqual(result["status"], "success")
        
        # Verify the column was found in the right models
        column_models = [r["model_name"] for r in result["results"]]
        self.assertIn("orders", column_models)
        self.assertIn("customer_orders", column_models)
    
    def test_find_related_models(self):
        """Test finding related models."""
        # Initialize searcher
        searcher = self.dbt_tools.searcher
        
        # Find models related to customer
        related = searcher.find_related_models("customer")
        
        # Verify customer_orders references customer
        self.assertIn("customer_orders", related["downstream"])
        
        # Find models related to customer_orders
        related = searcher.find_related_models("customer_orders")
        
        # Verify customer_orders depends on customer
        self.assertIn("customer", related["upstream"])

if __name__ == "__main__":
    unittest.main() 