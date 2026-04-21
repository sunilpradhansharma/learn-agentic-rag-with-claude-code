"""
solution/compare_eval_methods.py — Reference implementation for Lesson 8.

Identical to lessons/08-ragas/compare_eval_methods.py.
"""

import json
import os
import sys

_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", "..", "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "src", "rag")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

# Re-use the lesson script directly.
sys.path.insert(0, os.path.join(_LESSON_DIR, ".."))

from compare_eval_methods import main  # noqa: E402

if __name__ == "__main__":
    main()
