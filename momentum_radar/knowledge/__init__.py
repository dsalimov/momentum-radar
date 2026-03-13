"""
knowledge – PDF-based trading knowledge base.

This sub-package provides tools for loading and querying a rule-set derived
from trading books and guides stored as PDF files.  The extracted rules are
used at signal-evaluation time to apply an additional qualitative filter
before a signal is forwarded to users.

Modules
-------
* :mod:`~momentum_radar.knowledge.pdf_loader` – extracts text from PDF files,
  classifies rules into thematic categories, and exposes a
  :class:`~momentum_radar.knowledge.pdf_loader.KnowledgeBase` for real-time
  rule queries.
"""

from momentum_radar.knowledge.pdf_loader import KnowledgeBase, load_knowledge_base

__all__ = ["KnowledgeBase", "load_knowledge_base"]
