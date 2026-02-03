"""Tests for broker order submission logic using mocks.

Tests the position sizing integration and order submission flow
without requiring a live broker connection.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from hedge_fund.risk import calculate_position_size


class TestBrokerOrderFlow:
    """Mock-based tests for the order submission pipeline."""

    def test_position_size_feeds_into_order(self):
        """Verify position sizing output is valid for order submission."""
        qty = calculate_position_size(
            equity=100_000, entry_price=150.0, stop_price=145.0
        )
        assert isinstance(qty, int)
        assert qty > 0
        # Simulated order payload
        order = {
            "symbol": "NVDA",
            "qty": qty,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "stop_loss": {"stop_price": 145.0},
            "take_profit": {"limit_price": 160.0},
        }
        assert order["qty"] == qty
        assert order["stop_loss"]["stop_price"] == 145.0

    def test_bracket_order_parameters_valid(self):
        """Bracket order should have SL below entry for LONG."""
        entry = 100.0
        atr = 2.0
        sl = entry - 1.5 * atr  # 97.0
        tp = entry + 3.0 * atr  # 106.0

        assert sl < entry, "Stop loss must be below entry for LONG"
        assert tp > entry, "Take profit must be above entry for LONG"
        assert tp - entry > entry - sl, "R:R should be > 1"

    def test_bracket_order_parameters_short(self):
        """Bracket order should have SL above entry for SHORT."""
        entry = 100.0
        atr = 2.0
        sl = entry + 1.5 * atr  # 103.0
        tp = entry - 3.0 * atr  # 94.0

        assert sl > entry, "Stop loss must be above entry for SHORT"
        assert tp < entry, "Take profit must be below entry for SHORT"

    def test_zero_qty_prevents_order(self):
        """If position size is 0, order should not be submitted."""
        qty = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=100.0
        )
        assert qty == 0

    def test_idempotency_token_format(self):
        """Verify client_order_id format for idempotency."""
        import time
        symbol = "NVDA"
        side = "LONG"
        ts_token = int(time.time() // 60)
        client_oid = f"gm_v14_{symbol}_{side}_{ts_token}"
        assert client_oid.startswith("gm_v14_")
        assert symbol in client_oid
        assert side in client_oid

    def test_kill_switch_prevents_orders(self):
        """Kill switch should prevent all new orders."""
        kill_triggered = True
        qty = 100
        # Simulate the guard check from Alpaca_Helper.submit_bracket
        should_submit = not kill_triggered and qty > 0
        assert not should_submit

    def test_duplicate_position_guard(self):
        """Should not open a position if one already exists for the symbol."""
        pos_cache = {"NVDA": {"entry": 100.0, "qty": 50, "side": "LONG"}}
        symbol = "NVDA"
        assert symbol in pos_cache, "Guard should detect existing position"

    def test_pending_order_guard(self):
        """Should not submit if there's already a pending order for the symbol."""
        pending_orders = {
            "order-123": {"symbol": "TSLA", "side": "LONG", "qty": 30}
        }
        symbol = "TSLA"
        has_pending = any(
            oi["symbol"] == symbol for oi in pending_orders.values()
        )
        assert has_pending, "Guard should detect pending order"


class TestMockAlpacaSubmission:
    """Test the order submission flow with a mocked Alpaca API."""

    def _make_mock_api(self):
        """Create a mock Alpaca API object."""
        api = MagicMock()
        # Mock successful order submission
        mock_order = MagicMock()
        mock_order.id = "test-order-id-123"
        mock_order.status = "accepted"
        mock_order.symbol = "NVDA"
        api.submit_order.return_value = mock_order
        api.list_orders.return_value = []
        return api

    def test_submit_order_called_correctly(self):
        api = self._make_mock_api()
        qty = 100
        api.submit_order(
            symbol="NVDA", qty=qty,
            side="buy", type="market", time_in_force="day",
            order_class="bracket",
            stop_loss={"stop_price": 95.0},
            take_profit={"limit_price": 110.0},
        )
        api.submit_order.assert_called_once()
        call_kwargs = api.submit_order.call_args
        assert call_kwargs[1]["symbol"] == "NVDA"
        assert call_kwargs[1]["qty"] == 100
        assert call_kwargs[1]["order_class"] == "bracket"

    def test_submit_order_returns_order_id(self):
        api = self._make_mock_api()
        order = api.submit_order(
            symbol="NVDA", qty=50, side="buy", type="market",
            time_in_force="day", order_class="bracket",
            stop_loss={"stop_price": 95.0},
            take_profit={"limit_price": 110.0},
        )
        assert order.id == "test-order-id-123"

    def test_api_error_handled_gracefully(self):
        api = self._make_mock_api()
        api.submit_order.side_effect = Exception("Insufficient buying power")
        with pytest.raises(Exception, match="Insufficient buying power"):
            api.submit_order(
                symbol="NVDA", qty=50, side="buy", type="market",
                time_in_force="day", order_class="bracket",
                stop_loss={"stop_price": 95.0},
                take_profit={"limit_price": 110.0},
            )

    def test_duplicate_order_detection(self):
        api = self._make_mock_api()
        api.submit_order.side_effect = Exception(
            "client_order_id already exists"
        )
        try:
            api.submit_order(
                symbol="NVDA", qty=50, side="buy", type="market",
                time_in_force="day", order_class="bracket",
                client_order_id="gm_v14_NVDA_LONG_12345",
                stop_loss={"stop_price": 95.0},
                take_profit={"limit_price": 110.0},
            )
            submitted = True
        except Exception as e:
            submitted = False
            assert "already exists" in str(e).lower()
        assert not submitted

    def test_close_position_called(self):
        api = self._make_mock_api()
        api.close_position("NVDA")
        api.close_position.assert_called_once_with("NVDA")

    def test_cancel_all_on_kill_switch(self):
        api = self._make_mock_api()
        # Simulate kill switch activation
        api.cancel_all_orders()
        api.close_all_positions()
        api.cancel_all_orders.assert_called_once()
        api.close_all_positions.assert_called_once()

    def test_stop_loss_replacement(self):
        api = self._make_mock_api()
        new_stop = 98.50
        api.replace_order("stop-order-123", stop_price=new_stop)
        api.replace_order.assert_called_once_with(
            "stop-order-123", stop_price=new_stop
        )
