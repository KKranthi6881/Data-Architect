import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_script")

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge-chat-backend"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge-chat-backend/src"))

try:
    # Import DbtTools
    from knowledge_chat_backend.src.dbt_tools import DbtTools
    logger.info("Successfully imported DbtTools")
except ImportError as e:
    logger.error(f"Failed to import DbtTools: {str(e)}")
    try:
        # Try direct import
        from dbt_tools import DbtTools
        logger.info("Successfully imported DbtTools directly")
    except ImportError as e:
        logger.error(f"Failed to import DbtTools directly: {str(e)}")
        sys.exit(1)

def search_keyword(keyword):
    """Test the keyword search functionality"""
    logger.info(f"Testing keyword search for: {keyword}")
    
    # Initialize DbtTools
    tools = DbtTools()
    
    try:
        # Try content search
        logger.info("Trying content search...")
        content_results = tools.search_content(keyword)
        logger.info(f"Found {len(content_results) if content_results else 0} content results")
        
        # Check results structure
        if content_results and len(content_results) > 0:
            result = content_results[0]
            logger.info(f"First result type: {type(result)}")
            if hasattr(result, 'file_path'):
                logger.info(f"File path: {result.file_path}")
            else:
                logger.info("Result has no file_path attribute")
                
            if hasattr(result, 'content'):
                logger.info(f"Content available? {bool(result.content)}")
            else:
                logger.info("Result has no content attribute")
    except Exception as e:
        logger.error(f"Error in content search: {str(e)}")
    
    try:
        # Try file path search
        logger.info("Trying file path search...")
        path_pattern = f"*{keyword}*"
        path_results = tools.search_file_path(path_pattern)
        logger.info(f"Found {len(path_results) if path_results else 0} path results")
    except Exception as e:
        logger.error(f"Error in file path search: {str(e)}")
    
    # Try to get file content
    if content_results and len(content_results) > 0:
        result = content_results[0]
        if hasattr(result, 'file_path'):
            try:
                file_content = tools.get_file_content(result.file_path)
                logger.info(f"Retrieved file content successfully? {bool(file_content)}")
            except Exception as e:
                logger.error(f"Error getting file content: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    else:
        keyword = "order_item_summary"
    
    search_keyword(keyword)
    logger.info("Test completed") 