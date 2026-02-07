"""
hedge_fund.broker - Alpaca broker interface.

Extracted from bot.py. Handles position sync, bracket order submission,
stop management, kill switch, and equity tracking.

The Alpaca_Helper class accepts all external dependencies via constructor
parameters to avoid reliance on module-level globals.
"""

import time
import datetime
import logging
import threading
import json
from collections import defaultdict


class Alpaca_Helper:
    """
    Alpaca broker wrapper for position management and order execution.

    Manages:
        - Position sync with broker
        - Bracket order submission (market or limit)
        - Stop-loss replacement (rate-limited)
        - Kill switch (drawdown-based)
        - Equity tracking
        - Monte Carlo Governor priming from trade history
    """

    def __init__(self, api, db, settings, keys=None,
                 mc_governor=None, perf_monitor=None,
                 error_tracker=None, send_alert_fn=None,
                 get_daily_atr_fn=None, health_monitor=None):
        """
        Args:
            api: alpaca_trade_api.REST instance (already authenticated).
            db: Database_Helper instance for persistence.
            settings: dict of bot settings (RISK_PER_TRADE, KILL_SWITCH_DD, etc.).
            keys: dict of API keys (optional, for reference).
            mc_governor: MonteCarloGovernor instance (optional).
            perf_monitor: PerformanceMonitor instance (optional).
            error_tracker: ErrorTracker instance (optional).
            send_alert_fn: callable(subject, body, priority) for alerts (optional).
            get_daily_atr_fn: callable(ticker) -> float for ATR lookup (optional).
        """
        self.api = api
        self.db = db
        self._settings = settings
        self._keys = keys or {}
        self.mc_governor = mc_governor
        self.perf_monitor = perf_monitor
        self._error_tracker = error_tracker
        self._send_alert = send_alert_fn or (lambda s, b, p="normal": None)
        self._get_daily_atr = get_daily_atr_fn or (lambda t: None)
        self._health_monitor = health_monitor

        self.pos_cache = {}
        self._pos_lock = threading.RLock()
        self._stop_replace_last = defaultdict(float)
        self._kill_triggered = False

        # Load persisted pending orders from DB on startup
        self.pending_orders = self.db.load_pending_orders()
        if self.pending_orders:
            logging.info(f"Recovered {len(self.pending_orders)} pending orders from DB")
        self._init_equity()

        # Prime Governor with history so it works on restart
        if self.mc_governor:
            self._prime_governor()

        self.sync_positions()

    @property
    def kill_triggered(self):
        return self._kill_triggered

    def _init_equity(self):
        try:
            account = self.api.get_account()
            self.equity = float(account.equity)
            self.peak_equity = self.equity
            logging.info(f"Equity: ${self.equity:,.2f}")
        except Exception as e:
            logging.error(f"Equity fetch failed: {e}")
            self.equity = 50000.0
            self.peak_equity = 50000.0

    def refresh_equity(self):
        """Refresh equity every loop for accurate sizing."""
        try:
            account = self.api.get_account()
            self.equity = float(account.equity)
            self.peak_equity = max(self.peak_equity, self.equity)
            return self.equity
        except Exception as e:
            if self._error_tracker:
                self._error_tracker.record_failure("EquityRefresh", str(e))
            return self.equity

    def _prime_governor(self):
        """Load historical trade outcomes to prime the MC Governor."""
        try:
            trades = self.db.get_trade_outcomes(days=180)
            if not trades:
                return

            trades.reverse()  # oldest first

            count = 0
            for t in trades:
                pnl = t.get('pnl', 0) or 0
                r = t.get('pnl_r', 0) or 0
                days_ago = t.get('days_ago', 0)

                ts = datetime.datetime.now() - datetime.timedelta(days=days_ago)

                if abs(r) > 0.01 and abs(pnl) > 0:
                    risk_dollars = abs(pnl / r)
                    self.mc_governor.add_trade(pnl, risk_dollars, side=t.get('side'), timestamp=ts)
                    count += 1

            logging.info(f"Primed Monte Carlo with {count} historical trades")
        except Exception as e:
            logging.warning(f"Failed to prime Governor: {e}")

    def sync_positions(self):
        """Sync with ACTUAL broker positions + rebuild pending from broker open orders."""
        try:
            # Detect closed positions first
            if self.mc_governor and self.pos_cache:
                current_symbols = {p.symbol for p in self.api.list_positions()}
                cached_symbols = set(self.pos_cache.keys())
                closed_symbols = cached_symbols - current_symbols

                for s in closed_symbols:
                    try:
                        cached = self.pos_cache[s]
                        entry = cached.get('entry', 0)
                        sl = cached.get('sl', 0)
                        qty = cached.get('qty', 0)
                        side = cached.get('side')

                        init_risk_per_share = cached.get("init_risk") or abs(entry - sl)
                        risk_dollars = init_risk_per_share * qty if qty > 0 else 0

                        acts = self.api.get_activities(activity_types='FILL', direction='desc', limit=50)
                        want_side = "sell" if side == "LONG" else "buy"

                        exit_price = entry
                        filled_qty = 0
                        found_fill = False

                        for a in acts:
                            if getattr(a, "symbol", None) != s:
                                continue
                            if getattr(a, "side", "").lower() != want_side:
                                continue
                            exit_price = float(a.price)
                            filled_qty = float(a.qty)
                            found_fill = True
                            break

                        if found_fill and filled_qty > 0:
                            pnl = (exit_price - entry) * filled_qty if side == 'LONG' else (entry - exit_price) * filled_qty
                            pnl_r = pnl / risk_dollars if risk_dollars > 0 else 0.0
                            logging.info(f"Trade Closed {s}: PnL=${pnl:.2f} R={pnl_r:+.2f} Risk=${risk_dollars:.2f}")
                            self.mc_governor.add_trade(pnl, risk_dollars, side=side)

                            if self.perf_monitor:
                                self.perf_monitor.add_trade_result(pnl > 0)

                            features = cached.get('entry_features', {})
                            try:
                                self.db.log_trade_outcome(
                                    symbol=s, side=side,
                                    entry_price=entry, exit_price=exit_price,
                                    pnl=pnl, pnl_r=pnl_r, outcome=pnl_r,
                                    features=features,
                                    reason=cached.get('reason', 'Signal')
                                )
                            except Exception as e_log:
                                logging.warning(f"Trade outcome log failed for {s}: {e_log}")
                        else:
                            logging.info(f"Trade Closed {s} (Fill not found in recent acts)")

                    except Exception as e:
                        logging.warning(f"Failed to process closed trade {s}: {e}")

            # Get real positions from broker
            broker_positions = {p.symbol: p for p in self.api.list_positions()}

            # Rebuild pending_orders from broker's open orders
            try:
                open_orders = self.api.list_orders(status='open', nested=True)
                broker_order_ids = set()

                for order in open_orders:
                    parent_id = getattr(order, 'parent_order_id', None)
                    if parent_id is not None:
                        continue

                    order_class = getattr(order, 'order_class', None)
                    legs = getattr(order, 'legs', None)
                    order_type = getattr(order, 'type', None)

                    is_bracket_parent = (
                        order_class in ['bracket', 'oco', 'oto'] and
                        order_type in ['market', 'limit', None]
                    )

                    if not is_bracket_parent:
                        continue

                    broker_order_ids.add(order.id)

                    if order.id not in self.pending_orders:
                        sl = 0.0
                        tp = 0.0
                        for leg in (legs or []):
                            tp_candidate = getattr(leg, 'limit_price', None)
                            sl_candidate = getattr(leg, 'stop_price', None)
                            if tp_candidate is not None:
                                tp = float(tp_candidate or 0)
                            if sl_candidate is not None:
                                sl = float(sl_candidate or 0)

                        price = float(order.limit_price or order.filled_avg_price or 0)
                        if price <= 0:
                            try:
                                lt = self.api.get_latest_trade(order.symbol)
                                price = float(lt.price)
                            except Exception:
                                price = 0.0

                        side = 'LONG' if order.side == 'buy' else 'SHORT'
                        atr = abs(price - sl) / 1.5 if sl > 0 and price > 0 else 0

                        self.pending_orders[order.id] = {
                            'symbol': order.symbol,
                            'side': side,
                            'qty': int(order.qty),
                            'price': price,
                            'sl': sl,
                            'tp': tp,
                            'atr': atr,
                            'ts': order.created_at.isoformat() if order.created_at else datetime.datetime.now().isoformat()
                        }

                        self.db.save_pending_order(order.id, order.symbol, side, int(order.qty), price, sl, tp, atr)
                        logging.info(f"Recovered bracket order: {order.symbol} SL=${sl:.2f} TP=${tp:.2f}")

                # Remove pending orders that no longer exist at broker
                for order_id in list(self.pending_orders.keys()):
                    if order_id not in broker_order_ids:
                        try:
                            order = self.api.get_order(order_id)
                            if order.status in ['filled', 'canceled', 'expired', 'rejected']:
                                logging.info(f"Order {order.status}: {self.pending_orders[order_id]['symbol']}")
                                self.db.delete_pending_order(order_id)
                                del self.pending_orders[order_id]
                        except Exception:
                            self.db.delete_pending_order(order_id)
                            del self.pending_orders[order_id]
            except Exception as e:
                logging.debug(f"Open orders check failed: {e}")

            # Check pending orders - move to pos_cache if filled
            for order_id, order_info in list(self.pending_orders.items()):
                try:
                    order = self.api.get_order(order_id)
                    symbol = order_info['symbol']

                    if order.status == 'filled':
                        if symbol in broker_positions:
                            p = broker_positions[symbol]
                            entry = float(p.avg_entry_price)
                            qty = int(abs(float(p.qty)))
                            side = order_info['side']
                            sl = order_info.get('sl', 0)
                            tp = order_info.get('tp', 0)
                            atr = order_info.get('atr', 0) or self._get_daily_atr(symbol) or entry * 0.02

                            if sl == 0 or tp == 0:
                                atr = self._get_daily_atr(symbol) or entry * 0.02
                                sl = entry - 1.5*atr if side == 'LONG' else entry + 1.5*atr
                                tp = entry + 3.0*atr if side == 'LONG' else entry - 3.0*atr

                            init_risk = abs(entry - sl) if sl else abs(entry * 0.02)

                            self.pos_cache[symbol] = {
                                'entry': entry, 'qty': qty, 'side': side,
                                'ts': datetime.datetime.now().isoformat(),
                                'atr': atr, 'pyramided': False,
                                'sl': sl, 'tp': tp,
                                'init_sl': sl,
                                'init_risk': init_risk,
                                'ratcheted': False,
                                'be_moved': False,
                                'entry_features': order_info.get('entry_features', {})
                            }
                            self.db.update_position(symbol, entry, qty, side, sl, tp, atr, False)
                            self.db.log_trade(symbol, side, qty, entry, 0, "Filled")
                            logging.info(f"Order filled: {side} {symbol} x{qty} @ ${entry:.2f}")

                        self.db.delete_pending_order(order_id)
                        del self.pending_orders[order_id]

                    elif order.status in ['canceled', 'expired', 'rejected']:
                        logging.warning(f"Order {order.status}: {symbol}")
                        self.db.delete_pending_order(order_id)
                        del self.pending_orders[order_id]

                except Exception as e:
                    logging.debug(f"Order check failed: {e}")

            # Sync pos_cache with actual broker positions
            for symbol in list(self.pos_cache.keys()):
                if symbol not in broker_positions:
                    del self.pos_cache[symbol]
                    self.db.delete_position(symbol)

            for symbol, p in broker_positions.items():
                if symbol not in self.pos_cache:
                    entry = float(p.avg_entry_price)
                    qty = int(abs(float(p.qty)))
                    side = 'LONG' if float(p.qty) > 0 else 'SHORT'
                    atr = self._get_daily_atr(symbol) or entry * 0.02
                    sl = entry - 1.5*atr if side == 'LONG' else entry + 1.5*atr
                    tp = entry + 3.0*atr if side == 'LONG' else entry - 3.0*atr

                    init_risk = abs(entry - sl)

                    self.pos_cache[symbol] = {
                        'entry': entry, 'qty': qty, 'side': side,
                        'ts': datetime.datetime.now().isoformat(),
                        'atr': atr, 'pyramided': False,
                        'sl': sl, 'tp': tp,
                        'init_sl': sl,
                        'init_risk': init_risk,
                        'ratcheted': False,
                        'be_moved': False
                    }
                    self.db.update_position(symbol, entry, qty, side, sl, tp, atr, False)
                    logging.info(f"Position synced from broker: {side} {symbol}")

            self.refresh_equity()
            if self._health_monitor:
                self._health_monitor.mark_broker_sync()
            logging.info(f"Sync: {len(self.pos_cache)} positions, {len(self.pending_orders)} pending")
        except Exception as e:
            if self._error_tracker:
                self._error_tracker.record_failure("SyncPositions", str(e))

    def _find_open_stop_order_id(self, symbol):
        """Find the open STOP leg for a symbol (from a filled bracket)."""
        try:
            orders = self.api.list_orders(status='open', nested=True)
            for o in orders:
                if getattr(o, "symbol", None) != symbol:
                    continue
                otype = getattr(o, "type", None)
                if otype in ["stop", "stop_limit"]:
                    return o.id
        except Exception as e:
            logging.debug(f"_find_open_stop_order_id failed for {symbol}: {e}")
        return None

    def replace_stop(self, symbol, new_stop):
        """Replace the open stop-loss leg price for an existing bracket."""
        now = time.time()
        if now - self._stop_replace_last[symbol] < self._settings.get("STOP_REPLACE_COOLDOWN_SEC", 60):
            return False

        stop_id = self._find_open_stop_order_id(symbol)
        if not stop_id:
            logging.debug(f"No open stop leg found for {symbol}")
            return False

        try:
            order = self.api.get_order(stop_id)
            order_type = getattr(order, 'type', None)

            payload = {'stop_price': round(float(new_stop), 2)}

            if order_type == 'stop_limit':
                old_stop = float(getattr(order, 'stop_price', 0) or 0)
                old_limit = float(getattr(order, 'limit_price', 0) or 0)
                if old_limit and old_stop:
                    offset = old_limit - old_stop
                    payload['limit_price'] = round(float(new_stop) + offset, 2)
                else:
                    payload['limit_price'] = payload['stop_price']

            self.api.replace_order(stop_id, **payload)
            self._stop_replace_last[symbol] = now
            logging.info(f"SL Replaced {symbol} -> ${new_stop:.2f}")
            return True
        except Exception as e:
            logging.warning(f"replace_stop failed for {symbol}: {e}")
            return False

    def replace_stop_loss(self, symbol, new_stop):
        """Alias for legacy calls."""
        return self.replace_stop(symbol, new_stop)

    def calculate_position_size(self, entry_price, stop_price, vix_mult=1.0):
        """Risk-based sizing: qty = (equity * risk_pct) / stop_distance."""
        risk_per_trade = self.equity * self._settings['RISK_PER_TRADE']
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            return 0

        qty = int((risk_per_trade * vix_mult) / stop_distance)
        max_qty = int(self.equity * 0.20 / entry_price)
        qty = min(max(qty, 0), max_qty)

        if self._settings.get('USE_MARKET_ORDERS', False):
            haircut = self._settings.get('SLIPPAGE_HAIRCUT', 0.10)
            qty = int(qty * (1 - haircut))

        return qty

    def submit_bracket(self, t, side, qty, current_price, sl, tp, atr_override=None, entry_features=None):
        """Submit a bracket order with SL and TP."""
        if self._kill_triggered or qty <= 0:
            return False

        # Don't stack pending or open positions
        with self._pos_lock:
            if t in self.pos_cache:
                logging.debug(f"Already have open position for {t}")
                return False
            for oi in self.pending_orders.values():
                if oi['symbol'] == t:
                    logging.debug(f"Already have pending order for {t}")
                    return False

        ts_token = int(time.time() // 60)
        client_oid = f"gm_v14_{t}_{side}_{ts_token}"

        try:
            open_orders = self.api.list_orders(status='open', limit=50)
            if any(getattr(o, 'client_order_id', '') == client_oid for o in open_orders):
                logging.debug(f"Duplicate CID already open at broker: {client_oid}")
                return False
        except Exception:
            pass

        logging.info(f"Submitting {side} {t} x{qty} (CID: {client_oid})")

        self.db.upsert_order(client_oid, t, side, qty, "BRACKET", "INTENT_DECLARED",
                             raw_json=json.dumps({'sl': sl, 'tp': tp, 'price': current_price}))

        order_start = time.perf_counter()
        try:
            order_side = 'buy' if side == 'LONG' else 'sell'

            if self._settings['USE_MARKET_ORDERS']:
                order = self.api.submit_order(
                    symbol=t, qty=int(qty),
                    side=order_side, type='market', time_in_force='day',
                    order_class='bracket',
                    client_order_id=client_oid,
                    stop_loss={'stop_price': round(float(sl), 2)},
                    take_profit={'limit_price': round(float(tp), 2)}
                )
            else:
                order = self.api.submit_order(
                    symbol=t, qty=int(qty),
                    side=order_side, type='limit', limit_price=round(float(current_price), 2),
                    time_in_force='gtc',
                    order_class='bracket',
                    client_order_id=client_oid,
                    stop_loss={'stop_price': round(float(sl), 2)},
                    take_profit={'limit_price': round(float(tp), 2)}
                )

            self.db.upsert_order(client_oid, t, side, qty, "BRACKET", "SUBMITTED", broker_id=order.id)

            atr = float(atr_override) if atr_override else abs(float(current_price) - float(sl)) / 1.5
            self.db.save_pending_order(order.id, t, side, int(qty), float(current_price), float(sl), float(tp), float(atr))

            with self._pos_lock:
                self.pending_orders[order.id] = {
                    'symbol': t, 'side': side, 'qty': int(qty),
                    'price': float(current_price), 'sl': float(sl), 'tp': float(tp),
                    'atr': float(atr), 'ts': datetime.datetime.now().isoformat(),
                    'entry_features': entry_features or {}
                }

            logging.info(f"Bracket Sent: {t} {side} (ID: {order.id})")
            if self._health_monitor:
                self._health_monitor.record_order_submission_latency(time.perf_counter() - order_start)
            return True

        except Exception as e:
            msg = str(e).lower()
            if "client_order_id" in msg and ("already exists" in msg or "duplicate" in msg):
                logging.warning(f"Duplicate CID; assuming sync will recover: {client_oid}")
                self.db.upsert_order(client_oid, t, side, int(qty), "BRACKET", "DUPLICATE_CID", raw_json=str(e))
                return False

            if self._error_tracker:
                self._error_tracker.record_failure(f"Order_{t}", str(e))
            self.db.upsert_order(client_oid, t, side, int(qty), "BRACKET", "FAILED", raw_json=str(e))
            logging.error(f"Bracket Failed {t}: {e}")
            if self._health_monitor:
                self._health_monitor.record_order_submission_latency(time.perf_counter() - order_start)
            return False

    def shutdown(self):
        """Close network sessions/resources cleanly."""
        try:
            session = getattr(self.api, "_session", None)
            if session is not None:
                session.close()
        except Exception as e:
            logging.debug(f"Broker session close failed: {e}")

    def close(self, symbol, reason="Manual"):
        """Close a position, calculate P&L, and log outcome."""
        try:
            current_price = None
            try:
                quote = self.api.get_latest_trade(symbol)
                current_price = float(quote.price)
            except Exception as e:
                logging.debug(f"Could not get latest price for {symbol}: {e}")

            self.api.close_position(symbol)

            if symbol in self.pos_cache:
                with self._pos_lock:
                    pos = self.pos_cache[symbol]
                    entry = pos['entry']
                    qty = pos['qty']
                    side = pos['side']
                    atr = pos.get('atr', entry * 0.02)

                    if current_price:
                        if side == 'LONG':
                            pnl = (current_price - entry) * qty
                        else:
                            pnl = (entry - current_price) * qty
                    else:
                        pnl = 0

                    init_risk = pos.get('init_risk')
                    if init_risk and init_risk > 0 and qty > 0:
                        pnl_r = pnl / (init_risk * qty)
                    else:
                        risk_per_share = atr * self._settings.get('STOP_MULT', 1.5) if atr else entry * 0.03
                        pnl_r = (pnl / qty / risk_per_share) if risk_per_share and qty else 0

                    outcome = float(pnl_r) if pnl_r is not None else 0.0

                    entry_features = pos.get('entry_features', {})
                    if entry_features and current_price:
                        try:
                            self.db.log_trade_outcome(
                                symbol=symbol, side=side,
                                entry_price=entry, exit_price=current_price,
                                pnl=pnl, pnl_r=pnl_r, outcome=outcome,
                                features=entry_features, reason=reason
                            )
                        except Exception as e:
                            logging.debug(f"Failed to log trade outcome: {e}")

                    self.db.log_trade(symbol, side, qty, entry, pnl, f"Closed: {reason}")
                    self.db.delete_position(symbol)
                    del self.pos_cache[symbol]

                    pnl_str = f"${pnl:+,.2f}" if pnl != 0 else "(unknown)"
                    r_str = f"({pnl_r:+.2f}R)" if pnl_r != 0 else ""
                    logging.info(f"Closed {symbol} ({reason}) P&L: {pnl_str} {r_str}")
            else:
                logging.info(f"Closed {symbol} ({reason})")
        except Exception as e:
            logging.error(f"Close failed {symbol}: {e}")

    def close_position(self, symbol, reason="Manual"):
        """Alias for close()."""
        return self.close(symbol, reason)

    def check_kill(self):
        if self._kill_triggered:
            return True
        try:
            self.equity = float(self.api.get_account().equity)
            self.peak_equity = max(self.peak_equity, self.equity)
            drawdown = 1 - (self.equity / self.peak_equity)

            if drawdown >= self._settings['KILL_SWITCH_DD']:
                self._send_alert("KILL SWITCH", f"DD: {drawdown:.1%}", "high")
                logging.critical(f"KILL: DD={drawdown:.1%}")
                self.api.cancel_all_orders()
                self.api.close_all_positions()
                self._kill_triggered = True
                return True
        except Exception as e:
            if self._error_tracker:
                self._error_tracker.record_failure("KillCheck", str(e))
        return False
