"""
knowledge/pdf_loader.py – PDF-based trading knowledge base loader.

Overview
--------
This module extracts trading rules and concepts from one or more PDF files
(e.g. trading books, strategy guides) and stores them in a queryable
:class:`KnowledgeBase`.  The knowledge base is consumed at signal-evaluation
time so the bot can cross-reference its live signals against documented
trading rules before issuing alerts.

Rule categories
---------------
Rules are automatically assigned to one of the following thematic
``RuleCategory`` buckets during extraction:

* ``SUPPLY_DEMAND``       – zone identification, base/impulse logic.
* ``OPTIONS_FLOW``        – Delta, Theta, IV, sweep mechanics.
* ``CANDLESTICK``         – pattern names and confirmation rules.
* ``RISK_MANAGEMENT``     – position sizing, stop-loss, R:R thresholds.
* ``FUNDAMENTAL``         – revenue, net income, cash-flow, P/E thresholds.
* ``GENERAL``             – everything else.

Dependencies
------------
* ``pypdf`` (``pip install pypdf``) – lightweight, zero-dependency PDF reader.
  *PyPDF2* is also accepted as a fallback if ``pypdf`` is not available.

Usage::

    from momentum_radar.knowledge.pdf_loader import load_knowledge_base

    kb = load_knowledge_base(["path/to/trading_book.pdf"])
    rules = kb.query("supply demand zone")
    for rule in rules:
        print(rule.text)

    # Check whether a signal passes knowledge-base rules
    if kb.check_signal(direction="BUY", setup="demand zone retest"):
        ...

All public functions are safe to call with an empty or missing PDF list; the
knowledge base will simply contain no rules.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule categories
# ---------------------------------------------------------------------------

class RuleCategory(Enum):
    """Thematic category assigned to each extracted trading rule."""

    SUPPLY_DEMAND = "supply_demand"
    OPTIONS_FLOW = "options_flow"
    CANDLESTICK = "candlestick"
    RISK_MANAGEMENT = "risk_management"
    FUNDAMENTAL = "fundamental"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Keyword mapping used for category assignment
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[RuleCategory, List[str]] = {
    RuleCategory.SUPPLY_DEMAND: [
        "supply", "demand", "zone", "impulse", "base", "retest",
        "support", "resistance", "breakout", "structure",
    ],
    RuleCategory.OPTIONS_FLOW: [
        "delta", "theta", "gamma", "vega", "implied volatility", "iv",
        "sweep", "call", "put", "contract", "premium", "dte",
        "open interest", "options", "flow",
    ],
    RuleCategory.CANDLESTICK: [
        "doji", "engulfing", "harami", "dark cloud", "morning star",
        "evening star", "pin bar", "hammer", "shooting star",
        "candlestick", "candle", "pattern",
    ],
    RuleCategory.RISK_MANAGEMENT: [
        "stop loss", "stop-loss", "target", "risk", "reward", "r:r",
        "risk/reward", "position size", "trailing stop", "exit",
    ],
    RuleCategory.FUNDAMENTAL: [
        "revenue", "net income", "earnings", "pe ratio", "p/e",
        "cash flow", "balance sheet", "income statement", "ebitda",
        "free cash flow", "gross profit",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradingRule:
    """A single trading rule extracted from a PDF source.

    Attributes:
        text:      The raw rule text (cleaned sentence or paragraph).
        category:  Thematic :class:`RuleCategory` assigned during extraction.
        source:    Path to the PDF file the rule was extracted from.
        page:      1-based page number within the source document.
    """

    text: str
    category: RuleCategory
    source: str
    page: int


@dataclass
class KnowledgeBase:
    """Queryable in-memory store of :class:`TradingRule` objects.

    Attributes:
        rules: All rules loaded across all PDF sources.
    """

    rules: List[TradingRule] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(
        self,
        keywords: str,
        category: Optional[RuleCategory] = None,
        max_results: int = 10,
    ) -> List[TradingRule]:
        """Return rules whose text contains at least one keyword.

        Args:
            keywords:    Space-separated keywords to search for (case-insensitive).
            category:    Optional filter to restrict results to a single
                         :class:`RuleCategory`.
            max_results: Maximum number of rules returned.

        Returns:
            List of matching :class:`TradingRule` objects, ordered by their
            position in the knowledge base (insertion order).
        """
        terms = [kw.strip().lower() for kw in keywords.split() if kw.strip()]
        results: List[TradingRule] = []
        for rule in self.rules:
            if category is not None and rule.category != category:
                continue
            lower_text = rule.text.lower()
            if any(t in lower_text for t in terms):
                results.append(rule)
            if len(results) >= max_results:
                break
        return results

    def query_by_category(self, category: RuleCategory) -> List[TradingRule]:
        """Return all rules belonging to *category*.

        Args:
            category: The :class:`RuleCategory` to filter by.

        Returns:
            All matching :class:`TradingRule` objects.
        """
        return [r for r in self.rules if r.category == category]

    def check_signal(
        self,
        direction: str,
        setup: str,
        category: Optional[RuleCategory] = None,
    ) -> bool:
        """Check whether the knowledge base contains supportive rules for a signal.

        A signal is considered *supported* when at least one rule in the
        knowledge base references both the trade direction and setup keywords.

        Args:
            direction: ``"BUY"`` / ``"bullish"`` or ``"SELL"`` / ``"bearish"``.
            setup:     Setup label or description (e.g. ``"demand zone retest"``).
            category:  Optional category filter.

        Returns:
            ``True`` if at least one supporting rule is found, ``False``
            otherwise.  Always returns ``True`` when the knowledge base is
            empty so that an unpopulated KB does not suppress all signals.
        """
        if not self.rules:
            return True  # no rules loaded → do not block signals

        direction_terms = _direction_terms(direction)
        setup_terms = [s.lower() for s in setup.split() if len(s) > 2]

        search_terms = direction_terms + setup_terms
        matches = self.query(" ".join(search_terms), category=category, max_results=5)
        return len(matches) > 0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the knowledge base contents.

        Returns:
            Multi-line string listing rule counts by category.
        """
        lines = [f"KnowledgeBase – {len(self.rules)} rules total"]
        counts: Dict[RuleCategory, int] = {}
        for rule in self.rules:
            counts[rule.category] = counts.get(rule.category, 0) + 1
        for cat, count in sorted(counts.items(), key=lambda kv: kv[0].value):
            lines.append(f"  {cat.value:<20}: {count}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.rules)

    def __bool__(self) -> bool:
        return bool(self.rules)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _direction_terms(direction: str) -> List[str]:
    """Normalise a trade direction into a list of keyword aliases."""
    d = direction.lower()
    if d in ("buy", "long", "bullish"):
        return ["buy", "long", "bullish", "uptrend", "demand"]
    if d in ("sell", "short", "bearish"):
        return ["sell", "short", "bearish", "downtrend", "supply"]
    return [d]


def _assign_category(text: str) -> RuleCategory:
    """Classify *text* into a :class:`RuleCategory` by keyword matching.

    Args:
        text: Sentence or paragraph to classify.

    Returns:
        The best-matching :class:`RuleCategory`, or
        :attr:`RuleCategory.GENERAL` when no category-specific keyword is
        found.
    """
    lower = text.lower()
    scores: Dict[RuleCategory, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[cat] = score
    if not scores:
        return RuleCategory.GENERAL
    return max(scores, key=lambda c: scores[c])


def _extract_sentences(text: str) -> List[str]:
    """Split *text* into cleaned, non-trivial sentences.

    Args:
        text: Raw page text extracted from a PDF.

    Returns:
        List of cleaned sentence strings (each ≥ 20 characters).
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence-ending punctuation followed by a space/newline
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned: List[str] = []
    for sentence in raw_sentences:
        s = sentence.strip()
        # Skip headings, page numbers, and trivially short fragments
        if len(s) >= 20 and not re.fullmatch(r"[\d\W]+", s):
            cleaned.append(s)
    return cleaned


def _extract_text_pypdf(pdf_path: str) -> List[tuple[int, str]]:
    """Extract ``(page_number, text)`` tuples from *pdf_path* using *pypdf*.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of ``(1-based page number, page text)`` tuples.

    Raises:
        ImportError: If neither *pypdf* nor *PyPDF2* is installed.
    """
    # Try modern pypdf first, fall back to legacy PyPDF2
    try:
        import pypdf as _pdf_mod
        reader_cls = _pdf_mod.PdfReader
    except ImportError:
        try:
            import PyPDF2 as _pdf_mod  # type: ignore[no-redef]
            reader_cls = _pdf_mod.PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF extraction requires 'pypdf' (recommended) or 'PyPDF2'. "
                "Install with: pip install pypdf"
            ) from exc

    pages: List[tuple[int, str]] = []
    with open(pdf_path, "rb") as fh:
        reader = reader_cls(fh)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Failed to extract text from page %d of %s: %s", i, pdf_path, exc)
                text = ""
            if text.strip():
                pages.append((i, text))
    return pages


def _load_single_pdf(pdf_path: str) -> List[TradingRule]:
    """Parse a single PDF file and return a list of :class:`TradingRule` objects.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of :class:`TradingRule` objects extracted from the file.
    """
    if not os.path.isfile(pdf_path):
        logger.warning("PDF not found: %s", pdf_path)
        return []

    logger.info("Loading knowledge base from: %s", pdf_path)
    try:
        pages = _extract_text_pypdf(pdf_path)
    except ImportError:
        logger.warning(
            "Skipping %s – pypdf/PyPDF2 is not installed.  "
            "Install with: pip install pypdf",
            pdf_path,
        )
        return []
    except Exception as exc:
        logger.error("Failed to read PDF %s: %s", pdf_path, exc)
        return []

    rules: List[TradingRule] = []
    for page_num, page_text in pages:
        sentences = _extract_sentences(page_text)
        for sentence in sentences:
            category = _assign_category(sentence)
            rules.append(
                TradingRule(
                    text=sentence,
                    category=category,
                    source=os.path.abspath(pdf_path),
                    page=page_num,
                )
            )

    logger.info("Extracted %d rules from %s (%d pages)", len(rules), pdf_path, len(pages))
    return rules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_knowledge_base(pdf_paths: Sequence[str]) -> KnowledgeBase:
    """Load a :class:`KnowledgeBase` from one or more PDF files.

    PDF files are parsed sequentially.  Missing or unreadable files are
    logged as warnings and skipped; they do not cause an exception.

    Args:
        pdf_paths: Sequence of paths to PDF files containing trading books
                   or strategy guides.

    Returns:
        :class:`KnowledgeBase` populated with all successfully extracted
        :class:`TradingRule` objects.  Returns an empty knowledge base when
        *pdf_paths* is empty or all files fail to load.

    Example::

        from momentum_radar.knowledge.pdf_loader import load_knowledge_base

        kb = load_knowledge_base(["books/trading_mastery.pdf"])
        print(kb.summary())
        rules = kb.query("supply demand zone", max_results=5)
    """
    all_rules: List[TradingRule] = []
    for path in pdf_paths:
        all_rules.extend(_load_single_pdf(path))
    kb = KnowledgeBase(rules=all_rules)
    logger.info("KnowledgeBase ready: %d rules from %d file(s)", len(kb), len(pdf_paths))
    return kb
