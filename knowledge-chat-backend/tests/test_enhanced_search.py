import sys
import os
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.dbt_tools import DbtTools, RepoSingleton

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_column_search():
    """Test column search functionality in DbtTools."""
    # Get the test repository path
    repo_singleton = RepoSingleton.get_instance()
    test_repo_path = repo_singleton.get_repo_path("test_repo")
    
    if not test_repo_path:
        logger.error("No test repository found. Please run test_dbt_tools.py first to set up a test repository.")
        return
    
    # Create a DbtTools instance with the test repository
    dbt_tools = DbtTools("test_repo")
    dbt_tools.repo_path = test_repo_path
    
    # Test column search
    logger.info("Testing column search...")
    
    # Search for a column
    column_name = "amount"
    results = dbt_tools.search_column(column_name)
    
    logger.info(f"Found {len(results)} results for column '{column_name}'")
    
    # Display the results
    for result in results:
        logger.info(f"Model: {result.model_name}")
        logger.info(f"File Path: {result.file_path}")
        logger.info(f"Match Type: {result.match_type}")
        logger.info(f"Match Context: {result.match_context}")
        
        if hasattr(result, 'calculation') and result.calculation:
            logger.info(f"Calculation: {result.calculation}")
    
    # Test content search
    logger.info("\nTesting content search...")
    
    # Search for content
    search_term = "customer"
    content_results = dbt_tools.search_content(search_term)
    
    logger.info(f"Found {len(content_results)} results for content '{search_term}'")
    
    # Display the content results
    for result in content_results:
        logger.info(f"File Path: {result.file_path}")
        if result.model_name:
            logger.info(f"Model: {result.model_name}")
        
        if hasattr(result, 'match_contexts') and result.match_contexts:
            logger.info(f"Match Contexts: {len(result.match_contexts)}")
            logger.info(f"First Context: {result.match_contexts[0][:100]}...")
    
    # Test calculation extraction
    logger.info("\nTesting calculation extraction...")
    
    # Get calculations for a model
    model_name = "order_items"
    calculations = dbt_tools.extract_column_calculations(model_name)
    
    logger.info(f"Found {len(calculations)} calculations in model '{model_name}'")
    
    # Display the calculations
    for column, calculation in calculations.items():
        logger.info(f"Column: {column}")
        logger.info(f"Calculation: {calculation}")
    
    logger.info("\nTesting complete!")

if __name__ == "__main__":
    test_column_search() 