"""
Utils package initialization file.
This file makes the utils directory a proper Python package.
"""
# This file intentionally left empty to avoid circular imports.
# SQL analyzer functions are directly in sql_analyzer.py 

# Import and export ChromaDBManager and OllamaEmbeddingFunction
from .chromadb_manager import ChromaDBManager, OllamaEmbeddingFunction

# Import and export SQL analyzer functions
from .sql_analyzer import (
    analyze_sql_model,
    generate_enhancement_modifications,
    generate_suggested_approach,
    generate_failure_feedback,
    find_best_modification_target
)

# List of all exported names
__all__ = [
    'ChromaDBManager',
    'OllamaEmbeddingFunction',
    'analyze_sql_model',
    'generate_enhancement_modifications',
    'generate_suggested_approach',
    'generate_failure_feedback',
    'find_best_modification_target'
] 