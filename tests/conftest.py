"""
Test configuration.

Heavy ML packages (torch, transformers, pypdf) are stubbed at import time
so that test_rag.py can import rag.ingest without requiring the full ML stack.
The actual embedding model is replaced per-test via FakeEmbeddingModel.
"""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub heavy packages before rag.ingest is imported.
# EmbeddingModel is never instantiated in tests (get_model is patched),
# so stubs only need to satisfy the module-level imports.
_HEAVY_PACKAGES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "transformers",
    "pypdf",
    "faiss",
]

for _pkg in _HEAVY_PACKAGES:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = MagicMock()
