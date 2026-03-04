"""
tests/test_ui_formatter.py – Unit tests for the UI embed formatter.
"""

import pytest


def _make_valid_signal():
    from momentum_radar.strategies.base import StrategySignal

    s = StrategySignal(
        ticker="SPY",
        strategy="intraday",
        direction="BUY",
        timeframe="5m",
        score=84,
        grade="A",
        confirmations=[
            "HTF Alignment",
            "Break of Structure",
            "Volume Expansion",
            "Demand Zone Retest",
        ],
        entry=512.40,
        stop=510.80,
        target=516.80,
        rr=2.75,
        regime="Trending",
        htf_bias="Bullish",
        session="morning",
        fake_breakout_passed=True,
        valid=True,
    )
    return s


class TestFormatTelegramCard:
    def test_contains_ticker(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "SPY" in card

    def test_contains_direction(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "BUY" in card

    def test_contains_score(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "84" in card

    def test_contains_entry_stop_target(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "512.40" in card
        assert "510.80" in card
        assert "516.80" in card

    def test_contains_rr(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "2.75" in card

    def test_contains_regime(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "Trending" in card

    def test_contains_confirmations(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "HTF Alignment" in card
        assert "Volume Expansion" in card

    def test_fake_breakout_passed_label(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "PASSED" in card

    def test_separator_present(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card

        card = format_telegram_card(_make_valid_signal())
        assert "━" in card

    def test_max_4_confirmations_shown(self):
        from momentum_radar.ui.embed_formatter import format_telegram_card
        from momentum_radar.strategies.base import StrategySignal

        s = _make_valid_signal()
        s.confirmations = ["C1", "C2", "C3", "C4", "C5", "C6"]
        card = format_telegram_card(s)
        assert "C5" not in card
        assert "C4" in card


class TestFormatDiscordEmbed:
    def test_returns_dict(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed

        embed = format_discord_embed(_make_valid_signal())
        assert isinstance(embed, dict)

    def test_has_required_keys(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed

        embed = format_discord_embed(_make_valid_signal())
        assert "title" in embed
        assert "color" in embed
        assert "fields" in embed
        assert "footer" in embed

    def test_title_format(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed

        embed = format_discord_embed(_make_valid_signal())
        assert "SPY" in embed["title"]
        assert "BUY" in embed["title"]

    def test_sell_signal_is_red(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed, _COLOR_RED
        from momentum_radar.strategies.base import StrategySignal

        s = _make_valid_signal()
        s.direction = "SELL"
        embed = format_discord_embed(s)
        assert embed["color"] == _COLOR_RED

    def test_high_score_is_green(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed, _COLOR_GREEN

        s = _make_valid_signal()
        s.score = 85
        embed = format_discord_embed(s)
        assert embed["color"] == _COLOR_GREEN

    def test_fields_contain_trade_plan(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed

        embed = format_discord_embed(_make_valid_signal())
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "Trade Plan" in fields
        trade = fields["Trade Plan"]
        assert "512.40" in trade

    def test_fields_contain_confirmations(self):
        from momentum_radar.ui.embed_formatter import format_discord_embed

        embed = format_discord_embed(_make_valid_signal())
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "Confirmations" in fields
        assert "HTF Alignment" in fields["Confirmations"]


class TestFormatDailySummary:
    def test_returns_string(self):
        from momentum_radar.ui.embed_formatter import format_daily_summary

        card = format_daily_summary([_make_valid_signal()])
        assert isinstance(card, str)

    def test_contains_total_signals(self):
        from momentum_radar.ui.embed_formatter import format_daily_summary

        card = format_daily_summary([_make_valid_signal(), _make_valid_signal()])
        assert "2" in card

    def test_empty_signals_list(self):
        from momentum_radar.ui.embed_formatter import format_daily_summary

        card = format_daily_summary([])
        assert isinstance(card, str)
        assert "0" in card
