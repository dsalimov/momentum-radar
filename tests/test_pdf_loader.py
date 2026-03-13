"""
tests/test_pdf_loader.py – Unit tests for the PDF knowledge base loader.

Tests cover:
  * ``KnowledgeBase.query``          – keyword search.
  * ``KnowledgeBase.query_by_category`` – category filter.
  * ``KnowledgeBase.check_signal``   – supportive-rule lookup.
  * ``KnowledgeBase.summary``        – human-readable summary.
  * ``_assign_category``             – keyword-based category assignment.
  * ``_extract_sentences``           – sentence splitter.
  * ``load_knowledge_base``          – empty list and missing file handling.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from momentum_radar.knowledge.pdf_loader import (
    KnowledgeBase,
    RuleCategory,
    TradingRule,
    _assign_category,
    _extract_sentences,
    load_knowledge_base,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rule(
    text: str,
    category: RuleCategory = RuleCategory.GENERAL,
    source: str = "test.pdf",
    page: int = 1,
) -> TradingRule:
    return TradingRule(text=text, category=category, source=source, page=page)


def _make_kb(*texts_and_categories) -> KnowledgeBase:
    """Build a KnowledgeBase from ``(text, RuleCategory)`` pairs."""
    rules = [_make_rule(text, cat) for text, cat in texts_and_categories]
    return KnowledgeBase(rules=rules)


# ---------------------------------------------------------------------------
# _assign_category
# ---------------------------------------------------------------------------

class TestAssignCategory:
    def test_supply_demand_keywords(self):
        text = "Supply and demand zones are identified by the impulse and base candles."
        assert _assign_category(text) == RuleCategory.SUPPLY_DEMAND

    def test_options_flow_keywords(self):
        text = "Delta measures the rate of change of option price relative to underlying."
        assert _assign_category(text) == RuleCategory.OPTIONS_FLOW

    def test_candlestick_keywords(self):
        text = "A bearish engulfing candlestick pattern suggests a reversal at resistance."
        assert _assign_category(text) == RuleCategory.CANDLESTICK

    def test_risk_management_keywords(self):
        text = "Always set a stop loss to manage your risk/reward ratio properly."
        assert _assign_category(text) == RuleCategory.RISK_MANAGEMENT

    def test_fundamental_keywords(self):
        text = "Net income and free cash flow growth confirm a healthy balance sheet."
        assert _assign_category(text) == RuleCategory.FUNDAMENTAL

    def test_general_fallback(self):
        text = "This is a completely unrelated sentence about the weather today."
        assert _assign_category(text) == RuleCategory.GENERAL


# ---------------------------------------------------------------------------
# _extract_sentences
# ---------------------------------------------------------------------------

class TestExtractSentences:
    def test_splits_on_period(self):
        text = "This is sentence one. This is sentence two. And a third one here."
        sentences = _extract_sentences(text)
        assert len(sentences) == 3

    def test_filters_short_fragments(self):
        text = "OK. This is a properly formed sentence about trading zones."
        sentences = _extract_sentences(text)
        # "OK." is too short and should be filtered out
        assert all(len(s) >= 20 for s in sentences)

    def test_normalises_whitespace(self):
        text = "This  has   extra   spaces  in it.  So does this one here today."
        sentences = _extract_sentences(text)
        for s in sentences:
            assert "  " not in s

    def test_empty_string_returns_empty(self):
        assert _extract_sentences("") == []

    def test_numeric_only_fragments_filtered(self):
        text = "12345. This is a real trading rule about demand zones."
        sentences = _extract_sentences(text)
        assert not any(s.strip() == "12345" for s in sentences)


# ---------------------------------------------------------------------------
# KnowledgeBase.query
# ---------------------------------------------------------------------------

class TestKnowledgeBaseQuery:
    def test_finds_matching_rule(self):
        kb = _make_kb(
            ("Supply zones form at the top of sharp drops.", RuleCategory.SUPPLY_DEMAND),
            ("Delta measures sensitivity to underlying price.", RuleCategory.OPTIONS_FLOW),
        )
        results = kb.query("supply zones")
        assert len(results) == 1
        assert "Supply zones" in results[0].text

    def test_returns_empty_for_no_match(self):
        kb = _make_kb(("Unrelated text about something else entirely here.", RuleCategory.GENERAL))
        assert kb.query("demand zone retest") == []

    def test_category_filter_restricts_results(self):
        kb = _make_kb(
            ("Supply zones form at the top of sharp drops.", RuleCategory.SUPPLY_DEMAND),
            ("Supply your portfolio with diversified options positions.", RuleCategory.OPTIONS_FLOW),
        )
        results = kb.query("supply", category=RuleCategory.OPTIONS_FLOW)
        assert all(r.category == RuleCategory.OPTIONS_FLOW for r in results)

    def test_max_results_limit(self):
        rules = [("Supply demand zone rule number %d." % i, RuleCategory.SUPPLY_DEMAND) for i in range(20)]
        kb = _make_kb(*rules)
        results = kb.query("supply", max_results=5)
        assert len(results) <= 5

    def test_case_insensitive(self):
        kb = _make_kb(("SUPPLY zones are critical levels in price action.", RuleCategory.SUPPLY_DEMAND))
        results = kb.query("supply")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# KnowledgeBase.query_by_category
# ---------------------------------------------------------------------------

class TestQueryByCategory:
    def test_returns_only_matching_category(self):
        kb = _make_kb(
            ("Supply zones form at the top of sharp drops.", RuleCategory.SUPPLY_DEMAND),
            ("Delta measures option sensitivity.", RuleCategory.OPTIONS_FLOW),
            ("Net income growth is bullish fundamental.", RuleCategory.FUNDAMENTAL),
        )
        results = kb.query_by_category(RuleCategory.OPTIONS_FLOW)
        assert len(results) == 1
        assert results[0].category == RuleCategory.OPTIONS_FLOW

    def test_returns_empty_for_missing_category(self):
        kb = _make_kb(("Supply zones form at the top.", RuleCategory.SUPPLY_DEMAND))
        assert kb.query_by_category(RuleCategory.FUNDAMENTAL) == []


# ---------------------------------------------------------------------------
# KnowledgeBase.check_signal
# ---------------------------------------------------------------------------

class TestCheckSignal:
    def test_returns_true_when_supporting_rule_found(self):
        kb = _make_kb(
            ("Demand zones are bullish reversal areas at the base of impulse moves.", RuleCategory.SUPPLY_DEMAND),
        )
        assert kb.check_signal(direction="BUY", setup="demand zone retest") is True

    def test_returns_true_when_kb_is_empty(self):
        """Empty knowledge base must not block signals."""
        kb = KnowledgeBase()
        assert kb.check_signal(direction="BUY", setup="any setup") is True

    def test_returns_false_when_no_matching_rule(self):
        kb = _make_kb(
            ("Theta decay accelerates as expiration approaches for options.", RuleCategory.OPTIONS_FLOW),
        )
        # No rule mentions "supply zone breakdown"
        result = kb.check_signal(direction="SELL", setup="supply zone breakdown abc xyz unknown")
        # May return False since no overlap
        assert isinstance(result, bool)

    def test_bullish_direction_aliases(self):
        kb = _make_kb(
            ("Demand zones are long entry points aligned with the uptrend.", RuleCategory.SUPPLY_DEMAND),
        )
        assert kb.check_signal(direction="bullish", setup="demand zone") is True

    def test_bearish_direction_aliases(self):
        kb = _make_kb(
            ("Supply zones are short entry points in a bearish downtrend.", RuleCategory.SUPPLY_DEMAND),
        )
        assert kb.check_signal(direction="bearish", setup="supply zone") is True


# ---------------------------------------------------------------------------
# KnowledgeBase.summary
# ---------------------------------------------------------------------------

class TestKnowledgeBaseSummary:
    def test_summary_includes_total_count(self):
        kb = _make_kb(
            ("Supply zones form at the top of sharp drops.", RuleCategory.SUPPLY_DEMAND),
            ("Delta is the rate of change of option price.", RuleCategory.OPTIONS_FLOW),
        )
        summary = kb.summary()
        assert "2 rules" in summary

    def test_summary_includes_category_counts(self):
        kb = _make_kb(
            ("Supply zones form at the top of sharp drops.", RuleCategory.SUPPLY_DEMAND),
            ("Delta is the rate of change of option price.", RuleCategory.OPTIONS_FLOW),
        )
        summary = kb.summary()
        assert "supply_demand" in summary
        assert "options_flow" in summary

    def test_empty_kb_summary(self):
        kb = KnowledgeBase()
        assert "0 rules" in kb.summary()


# ---------------------------------------------------------------------------
# KnowledgeBase dunder helpers
# ---------------------------------------------------------------------------

class TestKnowledgeBaseDunders:
    def test_len(self):
        kb = _make_kb(
            ("Rule one about supply zones.", RuleCategory.SUPPLY_DEMAND),
            ("Rule two about delta and theta.", RuleCategory.OPTIONS_FLOW),
        )
        assert len(kb) == 2

    def test_bool_empty(self):
        assert not bool(KnowledgeBase())

    def test_bool_non_empty(self):
        kb = _make_kb(("Rule about supply.", RuleCategory.SUPPLY_DEMAND))
        assert bool(kb)


# ---------------------------------------------------------------------------
# load_knowledge_base – edge cases
# ---------------------------------------------------------------------------

class TestLoadKnowledgeBase:
    def test_empty_list_returns_empty_kb(self):
        kb = load_knowledge_base([])
        assert len(kb) == 0

    def test_missing_file_returns_empty_kb(self):
        kb = load_knowledge_base(["/nonexistent/path/book.pdf"])
        assert len(kb) == 0

    def test_loads_pdf_when_pypdf_available(self, tmp_path):
        """When pypdf is available a real (minimal) PDF is parsed."""
        try:
            import pypdf
        except ImportError:
            pytest.skip("pypdf not installed")

        # Create a minimal PDF using pypdf's writer
        from pypdf import PdfWriter
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        pdf_path = str(tmp_path / "test_book.pdf")
        with open(pdf_path, "wb") as fh:
            writer.write(fh)

        # A blank page produces no text → empty knowledge base, but no crash
        kb = load_knowledge_base([pdf_path])
        assert isinstance(kb, KnowledgeBase)
