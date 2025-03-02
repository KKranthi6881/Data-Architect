from src.agents.code_research import SearchTools,SimpleAnalysisSystem
from src.utils import ChromaDBManager
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize ChromaDB manager
db_manager = ChromaDBManager(persist_directory=str(BASE_DIR / "chroma_db"))
tools = SearchTools(db_manager)

graph = SimpleAnalysisSystem(tools)