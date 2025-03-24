import sys
import os
import tempfile
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.dbt_tools import DbtTools, DbtSearcher, DbtFileScanner, DbtModelParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_repo():
    """Create a temporary test repository with some sample models."""
    test_dir = tempfile.mkdtemp(prefix="test_dbt_")
    models_dir = os.path.join(test_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create sample model files
    create_model(models_dir, "orders.sql", """
    SELECT 
        order_id,
        customer_id,
        order_date,
        gross_amount,
        discount_amount,
        tax_amount,
        (gross_amount + discount_amount + tax_amount) as net_amount
    FROM raw_orders
    """)
    
    create_model(models_dir, "customers.sql", """
    SELECT 
        customer_id,
        first_name,
        last_name,
        email,
        created_at
    FROM raw_customers
    """)
    
    create_model(models_dir, "order_items.sql", """
    SELECT
        order_id,
        item_id,
        product_id,
        quantity,
        price as item_price,
        (quantity * price) as gross_item_sales_amount,
        (-1 * quantity * price * discount_percentage) as item_discount_amount,
        ((quantity * price) + (-1 * quantity * price * discount_percentage)) * tax_rate as item_tax_amount,
        (quantity * price) + 
        (-1 * quantity * price * discount_percentage) + 
        (((quantity * price) + (-1 * quantity * price * discount_percentage)) * tax_rate) as net_item_sales_amount
    FROM raw_order_items
    """)
    
    # Create a schema YAML file
    create_schema(models_dir, "schema.yml", """
    version: 2
    
    models:
      - name: orders
        description: "Order information"
        columns:
          - name: order_id
            description: "Primary key for orders"
          - name: customer_id
            description: "Foreign key to customers"
          - name: gross_amount
            description: "Gross order amount before discounts and taxes"
          - name: net_amount
            description: "Net order amount after discounts and taxes"
            
      - name: customers
        description: "Customer information"
        columns:
          - name: customer_id
            description: "Primary key for customers"
          - name: email
            description: "Customer email address"
            
      - name: order_items
        description: "Order line items"
        columns:
          - name: order_id
            description: "Foreign key to orders"
          - name: item_id
            description: "Primary key for order items"
          - name: gross_item_sales_amount
            description: "Gross sales amount for the line item"
          - name: item_discount_amount
            description: "Discount amount for the line item"
          - name: item_tax_amount
            description: "Tax amount for the line item"
          - name: net_item_sales_amount
            description: "Net sales amount for the line item after discounts and taxes"
    """)
    
    return test_dir

def create_model(models_dir, name, content):
    """Create a model file with the given name and content."""
    file_path = os.path.join(models_dir, name)
    with open(file_path, 'w') as f:
        f.write(content)
    logger.info(f"Created model {name}")

def create_schema(models_dir, name, content):
    """Create a schema file with the given name and content."""
    file_path = os.path.join(models_dir, name)
    with open(file_path, 'w') as f:
        f.write(content)
    logger.info(f"Created schema {name}")

def test_search_functionality():
    """Test the search functionality."""
    # Create a test repository
    test_dir = create_test_repo()
    logger.info(f"Created test repository at {test_dir}")
    
    try:
        # Initialize the file scanner
        file_scanner = DbtFileScanner(test_dir)
        file_scanner.index_project()
        
        # Initialize the model parser
        model_parser = DbtModelParser(test_dir)
        
        # Initialize the searcher
        searcher = DbtSearcher(test_dir)
        
        # Test column search
        logger.info("\nTesting column search...")
        column_results = searcher.search_by_column_name("net_item_sales_amount")
        
        if column_results:
            logger.info(f"Found {len(column_results)} results for column 'net_item_sales_amount'")
            for result in column_results:
                logger.info(f"Model: {result.model_name}")
                logger.info(f"File Path: {result.file_path}")
                logger.info(f"Match Type: {result.match_type}")
                
                if hasattr(result, 'calculation') and result.calculation:
                    logger.info(f"Calculation: {result.calculation}")
        else:
            logger.warning("No results found for column 'net_item_sales_amount'")
        
        # Test content search
        logger.info("\nTesting content search...")
        content_results = searcher.search_by_content("discount")
        
        if content_results:
            logger.info(f"Found {len(content_results)} results for content 'discount'")
            for result in content_results:
                logger.info(f"File Path: {result.file_path}")
                if result.model_name:
                    logger.info(f"Model: {result.model_name}")
                
                if hasattr(result, 'match_contexts') and result.match_contexts:
                    logger.info(f"Match Contexts: {len(result.match_contexts)}")
                    logger.info(f"First Context: {result.match_contexts[0][:100]}...")
        else:
            logger.warning("No results found for content 'discount'")
        
        # Test calculation extraction
        logger.info("\nTesting calculation extraction...")
        calculations = searcher.extract_column_calculations("order_items")
        
        if calculations:
            logger.info(f"Found {len(calculations)} calculations in model 'order_items'")
            for column, calculation in calculations.items():
                logger.info(f"Column: {column}")
                logger.info(f"Calculation: {calculation}")
        else:
            logger.warning("No calculations found in model 'order_items'")
        
        # Test model dependencies
        logger.info("\nTesting model dependencies...")
        
        # First add some dependencies to the models
        order_items_path = os.path.join(test_dir, "models", "order_items.sql")
        with open(order_items_path, 'r') as f:
            content = f.read()
        
        content = content.replace("FROM raw_order_items", "FROM {{ ref('orders') }}")
        
        with open(order_items_path, 'w') as f:
            f.write(content)
        
        # Re-index the project
        file_scanner.index_project()
        
        # Test finding related models
        related = searcher.find_related_models("orders")
        logger.info(f"Related models for 'orders': {related}")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up test repository at {test_dir}")

if __name__ == "__main__":
    test_search_functionality() 