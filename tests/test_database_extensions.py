"""
tests/test_database_extensions.py – Unit tests for AlertPreference and
SqueezeAlertRecord database helpers.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch


@pytest.fixture(autouse=True)
def in_memory_db():
    """Initialise an in-memory SQLite database for every test."""
    from momentum_radar.storage.database import init_db
    import momentum_radar.storage.database as db_module

    init_db("sqlite:///:memory:")
    yield
    # Reset engine/session so tests don't bleed into each other
    db_module._ENGINE = None
    db_module._SessionLocal = None


# ---------------------------------------------------------------------------
# AlertPreference
# ---------------------------------------------------------------------------

class TestAlertPreference:
    def test_default_is_enabled(self):
        from momentum_radar.storage.database import get_alert_preference

        assert get_alert_preference("chat_123") is True

    def test_set_and_get_enabled(self):
        from momentum_radar.storage.database import set_alert_preference, get_alert_preference

        set_alert_preference("chat_1", True)
        assert get_alert_preference("chat_1") is True

    def test_set_and_get_disabled(self):
        from momentum_radar.storage.database import set_alert_preference, get_alert_preference

        set_alert_preference("chat_2", False)
        assert get_alert_preference("chat_2") is False

    def test_update_existing_preference(self):
        from momentum_radar.storage.database import set_alert_preference, get_alert_preference

        set_alert_preference("chat_3", True)
        set_alert_preference("chat_3", False)
        assert get_alert_preference("chat_3") is False

    def test_independent_chats(self):
        from momentum_radar.storage.database import set_alert_preference, get_alert_preference

        set_alert_preference("chat_a", True)
        set_alert_preference("chat_b", False)
        assert get_alert_preference("chat_a") is True
        assert get_alert_preference("chat_b") is False


# ---------------------------------------------------------------------------
# SqueezeAlertRecord / spam filter
# ---------------------------------------------------------------------------

class TestShouldSendSqueezeAlert:
    def test_first_alert_always_allowed(self):
        from momentum_radar.storage.database import should_send_squeeze_alert

        assert should_send_squeeze_alert("NEWT", 80) is True

    def test_recent_alert_same_score_suppressed(self):
        from momentum_radar.storage.database import (
            should_send_squeeze_alert,
            record_squeeze_alert,
        )

        record_squeeze_alert("SPAM", 70)
        # Immediately try again with the same score – should be suppressed
        assert should_send_squeeze_alert("SPAM", 70) is False

    def test_recent_alert_score_jump_allowed(self):
        from momentum_radar.storage.database import (
            should_send_squeeze_alert,
            record_squeeze_alert,
        )

        record_squeeze_alert("JUMP", 60)
        # Score increases by 10+ → should be allowed even within 6 hours
        assert should_send_squeeze_alert("JUMP", 70) is True

    def test_score_increase_below_threshold_suppressed(self):
        from momentum_radar.storage.database import (
            should_send_squeeze_alert,
            record_squeeze_alert,
        )

        record_squeeze_alert("SMALL", 60)
        # Only +5 increase – below the +10 threshold
        assert should_send_squeeze_alert("SMALL", 65) is False

    def test_old_alert_allowed(self):
        from momentum_radar.storage.database import (
            should_send_squeeze_alert,
            record_squeeze_alert,
            SqueezeAlertRecord,
            _ENGINE,
        )
        from sqlalchemy.orm import Session

        record_squeeze_alert("OLD", 70)

        # Manually back-date the record by 7 hours
        with Session(_ENGINE) as session:
            rec = session.get(SqueezeAlertRecord, "OLD")
            rec.last_alerted_at = datetime.utcnow() - timedelta(hours=7)
            session.commit()

        assert should_send_squeeze_alert("OLD", 70) is True

    def test_record_squeeze_alert_updates_score(self):
        from momentum_radar.storage.database import (
            record_squeeze_alert,
            SqueezeAlertRecord,
            _ENGINE,
        )
        from sqlalchemy.orm import Session

        record_squeeze_alert("UPD", 55)
        record_squeeze_alert("UPD", 80)

        with Session(_ENGINE) as session:
            rec = session.get(SqueezeAlertRecord, "UPD")
            assert rec.last_score == 80
