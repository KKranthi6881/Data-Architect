import os
from utils import ChromaDBManager

def main():
    # Initialize ChromaDB manager
    db_manager = ChromaDBManager(persist_directory="./chroma_db")

    # Example usage for PDF files
    pdf_collection = "pdf_documents"
    pdf_path = "path/to/your/document.pdf"
    if os.path.exists(pdf_path):
        pdf_docs = db_manager.process_pdf(pdf_path)
        db_manager.add_documents(
            pdf_collection, 
            pdf_docs,
            metadata={"source": "pdf", "file_path": pdf_path}
        )

    # Example usage for SQL files
    sql_collection = "sql_documents"
    sql_path = "path/to/your/query.sql"
    if os.path.exists(sql_path):
        sql_docs = db_manager.process_sql(sql_path)
        db_manager.add_documents(
            sql_collection,
            sql_docs,
            metadata={"source": "sql", "file_path": sql_path}
        )

    # Example usage for Python files
    python_collection = "python_documents"
    python_path = "path/to/your/script.py"
    if os.path.exists(python_path):
        python_docs = db_manager.process_python(python_path)
        db_manager.add_documents(
            python_collection,
            python_docs,
            metadata={"source": "python", "file_path": python_path}
        )

    # Example query
    query_text = "What is the main function doing?"
    results = db_manager.query_collection("python_documents", query_text)
    print(f"Query results: {results}")

if __name__ == "__main__":
    main()