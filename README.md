# Data Architect - Knowledge Chat Backend

A knowledge-based chat system that integrates with various data sources including GitHub repositories, SQL scripts, and PDF documents.

## Features

- **Vector Database Integration**: Uses ChromaDB for efficient similarity search
- **Multiple Data Source Support**: 
  - GitHub repositories
  - SQL scripts with schema analysis
  - PDF documents
- **Code Analysis**: Analyzes code structure and relationships
- **Chat Interface**: Query your knowledge base using natural language

## Setup

### Prerequisites

- Python 3.9+
- Ollama for embeddings
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/KKranthi6881/Data-Architect.git
   cd Data-Architect
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```

5. Start the frontend (in a separate terminal):
   ```
   cd knowledge-chat-app
   npm install
   npm run dev
   ```

## Usage

### Adding Knowledge Sources

1. **GitHub Repositories**: Connect to GitHub repositories to analyze code
2. **SQL Scripts**: Upload SQL scripts to analyze schema and relationships
3. **PDF Documents**: Upload PDF documents for knowledge extraction

### Querying Knowledge

Use the chat interface to ask questions about your connected knowledge sources.

## Architecture

- **Backend**: FastAPI with ChromaDB vector store
- **Frontend**: React with Chakra UI
- **Embeddings**: Ollama for text embeddings
- **Analysis**: Custom code analyzer for SQL and programming languages

## License

MIT 