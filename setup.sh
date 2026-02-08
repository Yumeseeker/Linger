#!/bin/bash
# Writing Copilot — Quick Start
# Run this script to set up the entire environment and index the sample files.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, to index YOUR writing:
#   source venv/bin/activate
#   python index_writing.py /path/to/your/obsidian/vault

set -e

echo "========================================="
echo "  Writing Copilot — Setup"
echo "========================================="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3.10+ is required but not found."
    echo "Install it from https://python.org or via your package manager."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes on first run)..."
echo "  - sentence-transformers (embedding model)"
echo "  - chromadb (vector database)"
echo "  - spacy (sentence splitting)"
echo "  - rich (terminal formatting)"
pip install -r requirements.txt --quiet

# Download spaCy model
echo ""
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm --quiet

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "To index the sample files (test that everything works):"
echo "  source venv/bin/activate"
echo "  python index_writing.py sample_markdown/"
echo ""
echo "To query your indexed writing:"
echo "  python query_writing.py \"your sentence here\""
echo "  python query_writing.py --word \"said\""
echo "  python query_writing.py --interactive"
echo ""
echo "To index YOUR actual writing:"
echo "  python index_writing.py /path/to/your/markdown/folder"
echo ""
echo "For example, if you use Obsidian:"
echo "  python index_writing.py ~/Documents/MyVault"
echo ""
