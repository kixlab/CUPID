"""
Cupid utilities package.

This package provides common utility functions for:
- File operations (JSON loading/saving, directory management)
- Logging operations (error logging, file logging setup)
- Data validation (context factors, sessions validation)
- LLM generation (API clients, generation functionality)
- Results processing (aggregation, compilation)
- Data parsing (JSON/YAML parsing utilities)
"""

from .files import load_json, save_json, ensure_directory
from .validation import validate_context_factors, validate_sessions
from .generation import generate, generate_chat, Generator, GeneratorChat
from .parsing import parse_json, parse_yaml, json_to_yaml_str
from .formatting import format_interaction_log
from .logging import setup_main_logging, setup_worker_logging

__all__ = [
    # File utilities
    'load_json', 'save_json', 'ensure_directory',
    
    # Validation utilities
    'validate_context_factors', 'validate_sessions',
    
    # Generation utilities
    'generate', 'generate_chat', 'Generator', 'GeneratorChat',
    
    # Parsing utilities
    'parse_json', 'parse_yaml', 'json_to_yaml_str',
    
    # Formatting utilities
    'format_interaction_log',
    # Logging utilities
    'setup_main_logging', 'setup_worker_logging',
]
