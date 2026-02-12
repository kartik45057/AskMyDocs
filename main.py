"""
AskMyDocs - Retrieval Augmented Generation (RAG) System
--------------------------------------------------------
Entry point for the application.

Author: Kartik Singh
Description:
CLI interface for querying documents using a RAG workflow
powered by LangGraph, ChromaDB, and LLM.
"""

import os
import sys
import logging
from pathlib import Path
from Rag.Workflow import Execute_Workflow
from Rag.models import FileInfo


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def validate_file(path: str) -> None:
    """Validate file existence and type."""
    file_path = Path(path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    if file_path.suffix.lower() != ".pdf":
        logger.error("Currently only PDF files are supported.")
        sys.exit(1)


def print_banner() -> None:
    """Display application banner."""
    print("\n" + "=" * 60)
    print("📚  AskMyDocs - Intelligent Document Q&A System")
    print("=" * 60)
    print("Type 'exit' anytime to quit.\n")


# -----------------------------------------------------------------------------
# Main Application Entry
# -----------------------------------------------------------------------------
def main() -> None:
    """Main application runner."""
    try:
        print_banner()

        file_path = input("Enter path to your PDF file: ").strip()
        validate_file(file_path)

        logger.info("Initializing document workflow...")

        file_info = FileInfo(
            file_path=file_path,
            file_type="pdf"
        )

        Execute_Workflow(file_info)

    except KeyboardInterrupt:
        print("\n\nExiting gracefully... 👋")
        sys.exit(0)

    except Exception as e:
        logger.exception("Unexpected error occurred")
        sys.exit(1)


# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()




