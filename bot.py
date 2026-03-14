# ==============================================================================
# V12.7 HYBRID BOT — DAILY ALPHA + INTRADAY EXECUTION
#
# Architecture:
#   1. After market close: train daily ML model, generate cross-sectional watchlist
#   2. At market open: enter positions from watchlist via Alpaca bracket orders
#   3. During market: manage trailing stops, partial profits, timeout exits
#   4. Repeat
#
# Reuses: Alpaca_Helper, Polygon_Helper, Database_Helper, MonteCarloGovernor,
#          DailyRiskManager, EarningsGuard, ErrorTracker, Dashboard, send_alert
# ==============================================================================

# --- COLAB SETUP ---
try:
    import google.colab
    import subprocess
    import sys
    import os
    import time

    # 1. Strict Version Control (Prevent Dependency Hell)
    try:
        import numpy as np
        import pandas as pd
        import scipy

        bad_np = np.__version__.startswith('2.')
        bad_pd = pd.__version__ != '2.2.2'

        if bad_np or bad_pd:
            print(f"Incompatible environment detected (NumPy {np.__version__}, Pandas {pd.__version__}).")
            print("Aligning core dependencies (~60s)...", flush=True)
            os.system(f"{sys.executable} -m pip uninstall -y numpy pandas scipy pandas_ta")
            cmd = (
                f"{sys.executable} -m pip install "
                "numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 "
                "--force-reinstall --no-cache-dir --only-binary=:all:"
            )
            print(f"Executing: {cmd}", flush=True)
            ret = os.system(cmd)
            if ret != 0:
                os.system(f"{sys.executable} -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 --force-reinstall")
            print("Core stack aligned. AUTO-RESTARTING...", flush=True)
            time.sleep(1)
            os.kill(os.getpid(), 9)

    except ImportError:
        os.system(f"{sys.executable} -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 --only-binary=:all:")

    print(f"Core: NumPy {np.__version__} | Pandas {pd.__version__} | Scipy {scipy.__version__}")

    print("Installing dependencies...", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'xgboost'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'alpaca-trade-api', 'pyarrow', 'yfinance'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tzdata'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'rich'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'optuna', 'joblib'], check=True)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'lightgbm'], check=True)
    except Exception:
        pass
    print("Dependencies ready!")

    # GPU CHECK
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
        import numpy as np
        print(f"XGBoost Version: {xgb.__version__}")
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        try:
            model = XGBRegressor(device="cuda", verbosity=0)
            model.fit(X, y)
            print("GPU DETECTED & WORKING (device='cuda')")
        except Exception:
            try:
                model = XGBRegressor(tree_method='gpu_hist', verbosity=0)
                model.fit(X, y)
                print("GPU DETECTED & WORKING (tree_method='gpu_hist')")
            except Exception:
                print("Falling back to CPU mode.")
    except Exception as e:
        print(f"Warning during GPU check: {e}")
except ImportError:
    pass
except SystemExit as e:
    print(str(e))

import sys
import os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env")
except ImportError:
    print("python-dotenv not installed. Using system environment variables.")

print("\n" + "=" * 60)
print("LAUNCHING V12.7 HYBRID BOT...")
print("=" * 60 + "\n")

import threading, time, json, sqlite3, logging, warnings, gc, math, traceback
import datetime
from datetime import timedelta, timezone
import requests
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# Trading / Data
import alpaca_trade_api as tradeapi
import yfinance as yf
from logging.handlers import RotatingFileHandler

# --- "ONE BRAIN" IMPORTS ---
from hedge_fund.governance import MonteCarloGovernor
from hedge_fund.config import load_app_config, load_optimal_params, apply_to_settings
from hedge_fund.dashboard import Dashboard as SharedDashboard
from hedge_fund.data_providers import Polygon_Helper as _Polygon_Helper
from hedge_fund.broker import Alpaca_Helper as _Alpaca_Helper
from hedge_fund.reliability import FailureThresholds, ReliabilityMonitor, classify_failure, structured_failure_log

# Optional UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ==============================================================================
# 0. HARDWARE OPTIMIZATION & LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

NUM_CORES = os.cpu_count() or 4
IO_WORKERS = min(32, NUM_CORES * 4)
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)

# --- Timezone helpers ---
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    try:
        from dateutil.tz import gettz
        ET = gettz("America/New_York")
    except ImportError:
        ET = datetime.timezone(datetime.timedelta(hours=-5))

logging.info(f"HARDWARE: {NUM_CORES} cores | {IO_WORKERS} I/O workers")

# --- STORAGE SETUP ---
APP_CONFIG = load_app_config()
DRIVE_ROOT = APP_CONFIG.data_root

try:
    from google.colab import drive
    drive.mount('/content/drive')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna', 'joblib'], check=True)
except ImportError:
    pass

# ==============================================================================
# 1. API KEYS & SETTINGS
# ==============================================================================
def require_env(name):
    val = os.getenv(name)
    if not val:
        if name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
            raise RuntimeError(f"Missing required env var: {name}")
        return ""
    return val

KEYS = {
    "FMP": os.getenv("FMP_API_KEY", ""),
    "POLY": os.getenv("POLYGON_API_KEY", ""),
    "ALPACA_KEY": require_env("ALPACA_API_KEY"),
    "ALPACA_SEC": require_env("ALPACA_SECRET_KEY"),
    "DISCORD": os.getenv("DISCORD_WEBHOOK", ""),
}
ALPACA_URL = "https://paper-api.alpaca.markets"

SETTINGS = {
    "RISK_PER_TRADE": 0.01,
    "MAX_POSITIONS": 6,
    "KILL_SWITCH_DD": 0.10,
    "USE_MARKET_ORDERS": True,
    "STOP_REPLACE_COOLDOWN_SEC": 60,

    # V12.7 Hybrid Settings (from Optuna best config)
    'V12_SL_ATR_MULT': 1.3,
    'V12_TP_RR': 3.6,
    'V12_MAX_HOLD_DAYS': 10,
    'V12_TOP_N': 3,
    'V12_ENTRY_THRESHOLD': 0.43,
    'V12_PARTIAL_EXIT_ATR': 1.5,
    'V12_MIN_SPREAD': 0.007,
    'V12_RISK_PER_TRADE': 0.01,
    'V12_LOOKBACK_DAYS': 1500,
    'V12_TRAIN_DAYS': 250,
    'V12_FORWARD_DAYS': 5,
}

# Auto-load optimized params from backtester
_optimal = load_optimal_params(config=APP_CONFIG)
if _optimal:
    _updates = apply_to_settings(SETTINGS, _optimal)
    if _updates:
        logging.info(f"Applied {len(_updates)} params from backtester optimization")

MODEL_LOCK = threading.RLock()
KILL_TRIGGERED = False

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        RotatingFileHandler(os.path.join(APP_CONFIG.log_dir, "bot.log"), maxBytes=1e7, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
warnings.filterwarnings('ignore')

# ==============================================================================
# 2. INFRASTRUCTURE CLASSES (kept from v14.3)
# ==============================================================================

class ErrorTracker:
    def __init__(self):
        self.failures = defaultdict(int)
        self.last_success = defaultdict(lambda: datetime.datetime.now())
        self.reliability = ReliabilityMonitor("bot", FailureThresholds(degraded_after=3, safe_stop_after=5))

    def record_failure(self, component, error, symbol="global", endpoint="internal", retry_count=0):
        self.failures[component] += 1
        self.reliability.record_failure(component)
        err = error if isinstance(error, Exception) else Exception(str(error))
        structured_failure_log(
            component=component, symbol=symbol, endpoint=endpoint,
            retry_count=retry_count, error=err, logger=logging.error,
        )
        cls = classify_failure(err)
        logging.error(f"ERR {component} failed ({self.failures[component]}x, {cls}): {error}")
        if self.reliability.is_degraded(component):
            logging.warning(f"WARN {component} entering degraded mode")
        if self.reliability.should_safe_stop(component):
            logging.critical(f"CRIT {component} safe-stop threshold reached")
            send_alert(f"CRIT {component} BROKEN", f"Failed {self.failures[component]}x", "high")

    def record_success(self, component):
        if self.failures[component] > 0:
            logging.info(f"OK {component} recovered")
        self.failures[component] = 0
        self.reliability.record_success(component)
        self.last_success[component] = datetime.datetime.now()

ERROR_TRACKER = ErrorTracker()


class DailyRiskManager:
    """Daily loss limit: halt trading if intraday losses exceed threshold."""

    def __init__(self, max_daily_loss_pct=0.02):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.session_start_equity = None
        self.trading_halted = False
        self.halt_reason = None

    def initialize_session(self, current_equity):
        self.session_start_equity = current_equity
        self.trading_halted = False
        self.halt_reason = None
        logging.info(f"DailyRisk: Session started. Equity=${current_equity:,.2f}, "
                     f"Max loss=${current_equity * self.max_daily_loss_pct:,.2f}")

    def check_can_trade(self, current_equity):
        if self.trading_halted:
            return False
        if self.session_start_equity is None:
            return True
        daily_pnl = current_equity - self.session_start_equity
        daily_pnl_pct = daily_pnl / self.session_start_equity
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit hit: {daily_pnl_pct:.2%} (${daily_pnl:,.2f})"
            logging.critical(f"TRADING HALTED: {self.halt_reason}")
            try:
                send_alert("DAILY LOSS LIMIT", self.halt_reason, "high")
            except Exception:
                pass
            return False
        return True


# ==============================================================================
# 3. DATABASE
# ==============================================================================
class Database_Helper:
    def __init__(self, config=APP_CONFIG):
        self.config = config
        db_path = self.config.db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.create_tables()
        logging.info(f"Database: {db_path}")

    def create_tables(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS positions
                         (symbol TEXT PRIMARY KEY, entry REAL, qty INT, side TEXT,
                          ts TEXT, stop_loss REAL, take_profit REAL, atr REAL, pyramided INT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS trades
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT,
                          qty INT, price REAL, pnl REAL, reason TEXT,
                          ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS equity_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, equity REAL,
                          ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS dashboard_state
                         (id INTEGER PRIMARY KEY CHECK (id = 1),
                          equity REAL, regime TEXT, universe_size INT,
                          hedge_active INT, vix REAL, ts TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS pending_orders
                         (order_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty INT,
                          price REAL, sl REAL, tp REAL, atr REAL, ts TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS trade_outcomes
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          symbol TEXT, side TEXT,
                          entry_price REAL, exit_price REAL,
                          pnl REAL, pnl_r REAL, outcome REAL,
                          reason TEXT,
                          ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS bot_state
                         (id INTEGER PRIMARY KEY CHECK (id=1),
                          last_processed_ts TEXT, model_version TEXT,
                          feature_hash TEXT, risk_state_hash TEXT, updated_at TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS orders
                         (client_order_id TEXT PRIMARY KEY, broker_order_id TEXT,
                          symbol TEXT, side TEXT, qty INT, type TEXT, status TEXT,
                          created_ts TEXT, updated_ts TEXT, raw_json TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS fills
                         (broker_fill_id TEXT PRIMARY KEY, client_order_id TEXT,
                          symbol TEXT, qty INT, price REAL, ts TEXT, raw_json TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS events
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          ts TEXT, type TEXT, payload_json TEXT)''')
            self.conn.commit()

    def log_event(self, event_type, payload):
        with self.lock:
            try:
                self.conn.execute(
                    "INSERT INTO events (ts, type, payload_json) VALUES (?,?,?)",
                    (datetime.datetime.now().isoformat(), event_type, json.dumps(payload))
                )
                self.conn.commit()
            except Exception as e:
                logging.error(f"DB Log Event failed: {e}")

    def upsert_order(self, client_order_id, symbol, side, qty, order_type, status, broker_id=None, raw_json=None):
        with self.lock:
            now = datetime.datetime.now().isoformat()
            exists = self.conn.execute("SELECT created_ts FROM orders WHERE client_order_id=?", (client_order_id,)).fetchone()
            created_ts = exists[0] if exists else now
            self.conn.execute("""
                INSERT OR REPLACE INTO orders
                (client_order_id, broker_order_id, symbol, side, qty, type, status, created_ts, updated_ts, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (client_order_id, broker_id, symbol, side, qty, order_type, status, created_ts, now, raw_json))
            self.conn.commit()

    def save_pending_order(self, order_id, symbol, side, qty, price, sl, tp, atr):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO pending_orders VALUES (?,?,?,?,?,?,?,?,?)",
                (order_id, symbol, side, qty, price, sl, tp, atr, datetime.datetime.now().isoformat())
            )
            self.conn.commit()

    def delete_pending_order(self, order_id):
        with self.lock:
            self.conn.execute("DELETE FROM pending_orders WHERE order_id=?", (order_id,))
            self.conn.commit()

    def load_pending_orders(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute("SELECT order_id, symbol, side, qty, price, sl, tp, atr, ts FROM pending_orders")
            rows = c.fetchall()
            return {
                row[0]: {'symbol': row[1], 'side': row[2], 'qty': row[3],
                         'price': row[4], 'sl': row[5], 'tp': row[6], 'atr': row[7], 'ts': row[8]}
                for row in rows
            }

    def log_trade(self, symbol, side, qty, price, pnl=0, reason="Signal"):
        with self.lock:
            self.conn.execute(
                "INSERT INTO trades (symbol, side, qty, price, pnl, reason) VALUES (?,?,?,?,?,?)",
                (symbol, side, qty, price, pnl, reason)
            )
            self.conn.commit()

    def update_position(self, symbol, entry, qty, side, sl, tp, atr, pyramided=False):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?)",
                (symbol, entry, qty, side, datetime.datetime.now().isoformat(), sl, tp, atr, 1 if pyramided else 0)
            )
            self.conn.commit()

    def delete_position(self, symbol):
        with self.lock:
            self.conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            self.conn.commit()

    def log_equity(self, equity):
        with self.lock:
            self.conn.execute("INSERT INTO equity_history (equity) VALUES (?)", (equity,))
            self.conn.commit()

    def save_dashboard_state(self, equity, regime, universe_size, hedge_active, vix):
        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO dashboard_state
                   (id, equity, regime, universe_size, hedge_active, vix, ts)
                   VALUES (1, ?, ?, ?, ?, ?, ?)""",
                (equity, regime, universe_size, 1 if hedge_active else 0, vix,
                 datetime.datetime.now().isoformat())
            )
            self.conn.commit()

    def log_trade_outcome(self, symbol, side, entry_price, exit_price, pnl, pnl_r, reason="Signal"):
        with self.lock:
            self.conn.execute(
                """INSERT INTO trade_outcomes
                   (symbol, side, entry_price, exit_price, pnl, pnl_r, outcome, reason)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (symbol, side, entry_price, exit_price, pnl, pnl_r, pnl_r, reason)
            )
            self.conn.commit()

    def get_trade_outcomes(self, days=90):
        with self.lock:
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
            c = self.conn.cursor()
            c.execute("""
                SELECT symbol, side, entry_price, exit_price, pnl, pnl_r, reason, ts
                FROM trade_outcomes WHERE ts > ? ORDER BY ts DESC
            """, (cutoff,))
            return c.fetchall()


# ==============================================================================
# 4. ALERTS
# ==============================================================================
def send_alert(subject, body, priority="normal"):
    if KEYS['DISCORD']:
        try:
            color = 65280 if priority == "trade" else 16711680 if priority == "high" else 3447003
            requests.post(KEYS['DISCORD'], json={"embeds": [{"title": subject, "description": body, "color": color}]}, timeout=5)
        except Exception:
            pass


# ==============================================================================
# 5. DAILY ATR HELPER
# ==============================================================================
def get_daily_atr(ticker, period=14):
    """Use daily ATR for risk sizing."""
    try:
        tk = yf.Ticker(ticker)
        daily = tk.history(period='1mo', interval='1d')
        if len(daily) < period + 1:
            return None
        high = daily['High']
        low = daily['Low']
        close = daily['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr)
    except Exception as e:
        logging.debug(f"get_daily_atr({ticker}) failed: {e}")
        return None


class DailyATRCache:
    def __init__(self, ttl_sec=6 * 3600):
        self.ttl = ttl_sec
        self.cache = {}

    def get(self, ticker):
        now = time.time()
        if ticker in self.cache:
            ts, atr = self.cache[ticker]
            if now - ts < self.ttl:
                return atr
        atr = get_daily_atr(ticker)
        if atr and np.isfinite(atr) and atr > 0:
            self.cache[ticker] = (now, float(atr))
            return float(atr)
        return None

ATR_CACHE = DailyATRCache(ttl_sec=6 * 3600)


# ==============================================================================
# 6. EARNINGS GUARD
# ==============================================================================
class EarningsGuard:
    MAX_CACHE_SIZE = 500

    def __init__(self):
        self.earnings_cache = {}

    def check_safe(self, ticker):
        now = datetime.datetime.now().date()
        if ticker in self.earnings_cache and self.earnings_cache[ticker] > now:
            return True
        try:
            url = f"https://financialmodelingprep.com/api/v3/earnings-calendar-confirmed?symbol={ticker}&apikey={KEYS['FMP']}"
            res = requests.get(url, timeout=3).json()
            if res:
                earnings_date = pd.to_datetime(res[0]['date']).date()
                if 0 <= (earnings_date - now).days <= 1:
                    logging.warning(f"EARNINGS SKIP: {ticker}")
                    return False
            if len(self.earnings_cache) >= self.MAX_CACHE_SIZE:
                keys_to_remove = list(self.earnings_cache.keys())[:self.MAX_CACHE_SIZE // 2]
                for k in keys_to_remove:
                    del self.earnings_cache[k]
            self.earnings_cache[ticker] = now + datetime.timedelta(days=1)
        except Exception as e:
            logging.debug(f"EarningsGuard.check_safe({ticker}) failed: {e}")
        return True

    def check_news(self, ticker):
        """Check FMP News for catastrophic keywords."""
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=5&apikey={KEYS['FMP']}"
            res = requests.get(url, timeout=3).json()
            block_list = ["fraud", "subpoena", "investigation", "sec ", "bankruptcy", "chapter 11"]
            warn_list = ["lawsuit", "class action", "miss", "plunge", "tumble", "crash"]
            for item in res:
                pub_date = pd.to_datetime(item.get('publishedDate')).date()
                if (datetime.date.today() - pub_date).days > 1:
                    continue
                content = (item.get('title', '') + " " + item.get('text', '')).lower()
                if any(x in content for x in block_list):
                    logging.warning(f"NEWS BLOCK {ticker}: {item.get('title', '')[:50]}...")
                    return False, 2
                if any(x in content for x in warn_list):
                    return True, 1
            return True, 0
        except Exception:
            return True, 0


# ==============================================================================
# 7. RATE LIMITER
# ==============================================================================
class RateLimiter:
    def __init__(self, rate_per_sec=6.0, burst=10):
        self._rate = rate_per_sec
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, timeout=30.0):
        deadline = time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                self._tokens = min(self._burst,
                                   self._tokens + (now - self._last_refill) * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.time() > deadline:
                return False
            time.sleep(0.02)


# ==============================================================================
# 8. WRAPPER CLASSES (thin wrappers around hedge_fund/ package)
# ==============================================================================
class Polygon_Helper(_Polygon_Helper):
    """Polygon REST data (core logic in hedge_fund/data_providers.py)"""
    def __init__(self):
        super().__init__(keys=KEYS, drive_root=DRIVE_ROOT,
                         error_tracker=ERROR_TRACKER, io_workers=IO_WORKERS)


class Alpaca_Helper(_Alpaca_Helper):
    """Alpaca broker interface (core logic in hedge_fund/broker.py)"""
    def __init__(self, db, mc_governor=None):
        api = tradeapi.REST(KEYS['ALPACA_KEY'], KEYS['ALPACA_SEC'], ALPACA_URL, 'v2')
        super().__init__(
            api=api, db=db, settings=SETTINGS, keys=KEYS,
            mc_governor=mc_governor,
            error_tracker=ERROR_TRACKER, send_alert_fn=send_alert,
            get_daily_atr_fn=get_daily_atr,
        )

    def check_kill(self):
        global KILL_TRIGGERED
        result = super().check_kill()
        if self._kill_triggered:
            KILL_TRIGGERED = True
        return result


# ==============================================================================
# 9. DASHBOARD (V12.7 — simplified for daily hybrid)
# ==============================================================================
class Dashboard:
    def __init__(self):
        self.rich_available = False
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
            self.console = Console()
            self.rich_available = True
        except ImportError:
            pass

    def render(self, state):
        if self.rich_available:
            self._render_rich(state)
        else:
            self._render_ascii(state)

    def _render_rich(self, state):
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text

        equity_color = "green" if state.get('pnl_day', 0) >= 0 else "red"
        n_pos = len(state.get('positions', []))

        metrics = [
            f"[bold cyan]Equity:[/bold cyan] [bold {equity_color}]${state['equity']:,.2f}[/]",
            f"[bold yellow]Day PnL:[/bold yellow] ${state.get('pnl_day', 0):,.2f}",
            f"[bold blue]Univ:[/bold blue] {state.get('universe_size', 0)}",
            f"[bold magenta]Positions:[/bold magenta] {n_pos}",
        ]

        self.console.clear()
        self.console.rule("[bold gold1]V12.7 HYBRID BOT[/bold gold1]")
        self.console.print(Panel(Columns([Text.from_markup(m) for m in metrics]), box=box.ROUNDED, expand=True))

        # Positions Table
        pos_table = Table(title="[bold green]ACTIVE POSITIONS[/bold green]", box=box.SIMPLE_HEAD, expand=True)
        pos_table.add_column("Sym", style="cyan")
        pos_table.add_column("Side", style="white")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("SL", justify="right")
        pos_table.add_column("Days", justify="right")
        pos_table.add_column("Partial", style="white")

        for p in state.get('positions', []):
            pos_table.add_row(
                p.get('symbol', ''), p.get('side', ''),
                f"{p.get('entry', 0):.2f}", f"{p.get('sl', 0):.2f}",
                str(p.get('days_held', 0)),
                "Y" if p.get('partial_taken', False) else "N",
            )
        if not state.get('positions'):
            pos_table.add_row("-", "-", "-", "-", "-", "-")

        self.console.print(pos_table)

        # Watchlist
        if state.get('watchlist'):
            wl_table = Table(title="[bold yellow]TOMORROW'S WATCHLIST[/bold yellow]", box=box.SIMPLE_HEAD, expand=True)
            wl_table.add_column("Sym", style="cyan")
            wl_table.add_column("Side", style="white")
            wl_table.add_column("Score", justify="right")
            for item in state['watchlist']:
                wl_table.add_row(item['symbol'], item['side'], f"{item['score']:.4f}")
            self.console.print(wl_table)

        if state.get('logs'):
            self.console.rule("[bold dim]Events[/bold dim]")
            for log in state['logs'][-5:]:
                self.console.print(f"[dim]{log}[/dim]")

    def _render_ascii(self, state):
        print("\n" + "=" * 60)
        print(f"V12.7 HYBRID | ${state['equity']:,.2f} | PnL: ${state.get('pnl_day', 0):,.2f}")
        print("-" * 60)
        print("POSITIONS:")
        print(f"{'SYM':<6} {'SIDE':<6} {'ENTRY':<8} {'SL':<8} {'DAYS':<5}")
        for p in state.get('positions', []):
            print(f"{p.get('symbol',''):<6} {p.get('side',''):<6} "
                  f"{p.get('entry',0):<8.2f} {p.get('sl',0):<8.2f} {p.get('days_held',0):<5}")
        if state.get('watchlist'):
            print("-" * 60)
            print("WATCHLIST:")
            for item in state['watchlist']:
                print(f"  {item['side']:<6} {item['symbol']:<6} score={item['score']:.4f}")
        print("=" * 60)

    def render_loading(self, message):
        print(f"  {message}")


# ==============================================================================
# 10. DAILY SIGNAL ENGINE (V12.7)
# ==============================================================================
class DailySignalEngine:
    """
    Generates daily cross-sectional watchlist using V12.7 ML model.

    Call flow:
    1. After market close: engine.update_daily_data()
    2. engine.train_model() — walk-forward on accumulated daily bars
    3. engine.generate_signals() — returns tomorrow's watchlist
    """

    def __init__(self, tickers, polygon_helper, settings):
        self.tickers = tickers
        self.poly = polygon_helper
        self.settings = settings
        self.daily_cache = {}
        self.model_predictions = {}
        self.watchlist = {}
        self.last_train_date = None
        self.model_trained = False

    def update_daily_data(self):
        """Download latest daily bars and compute features."""
        from hedge_fund.daily_features import compute_daily_features

        raw_cache = {}
        for t in self.tickers:
            try:
                raw = self.poly.fetch_daily_data(t, days=self.settings['V12_LOOKBACK_DAYS'])
                if raw is not None and len(raw) > 200:
                    # Normalize daily bar timestamps (same fix as backtester_v12.py)
                    if hasattr(raw.index, 'tz') and raw.index.tz is not None:
                        raw.index = raw.index.tz_localize(None)
                    if hasattr(raw.index, 'normalize'):
                        raw.index = raw.index.normalize()
                    raw_cache[t] = raw
                    logging.info(f"Daily data: {t} has {len(raw)} bars")
            except Exception as e:
                logging.warning(f"Daily data fetch failed for {t}: {e}")

        for t, raw in raw_cache.items():
            try:
                df = compute_daily_features(raw, ticker=t, universe_daily=raw_cache)
                self.daily_cache[t] = df
            except Exception as e:
                logging.warning(f"Feature computation failed for {t}: {e}")

        logging.info(f"Daily data updated: {len(self.daily_cache)}/{len(self.tickers)} tickers")

    def train_model(self):
        """Walk-forward train the daily model on all available data."""
        from hedge_fund.daily_model import walk_forward_daily
        from hedge_fund.daily_features import DAILY_FEATURES

        self.model_predictions = {}

        for t, df in self.daily_cache.items():
            try:
                avail = [f for f in DAILY_FEATURES if f in df.columns]
                if len(avail) < 5:
                    logging.warning(f"Skipping {t}: only {len(avail)} features available")
                    continue
                result = walk_forward_daily(
                    df, avail,
                    train_days=self.settings['V12_TRAIN_DAYS'],
                    test_days=60,
                    step_days=60,
                )
                if result is not None and len(result) > 0:
                    self.model_predictions[t] = result
                    logging.info(f"Daily model: {t} has {len(result)} predictions")
            except Exception as e:
                logging.warning(f"Model training failed for {t}: {e}")

        self.model_trained = bool(self.model_predictions)
        self.last_train_date = datetime.date.today()
        logging.info(f"Model training complete: {len(self.model_predictions)} tickers trained")
        return self.model_trained

    def generate_signals(self):
        """Generate watchlist from predictions with T+1 shift."""
        from hedge_fund.daily_model import generate_watchlist

        if not self.model_trained:
            return {}

        self.watchlist = generate_watchlist(
            self.model_predictions,
            top_n=self.settings['V12_TOP_N'],
            bottom_n=self.settings['V12_TOP_N'],
            min_spread=self.settings['V12_MIN_SPREAD'],
        )
        logging.info(f"Watchlist generated: {len(self.watchlist)} days of signals")
        return self.watchlist

    def get_tomorrow_signals(self):
        """Get signals for the next trading day."""
        today = datetime.date.today()
        for d in sorted(self.watchlist.keys()):
            if d > today:
                return self.watchlist[d]
        # Fallback: return today's signals if available
        return self.watchlist.get(today, {})

    def get_latest_signals(self):
        """Get the most recent signals (for startup when model already trained)."""
        if not self.watchlist:
            return {}
        latest_date = max(self.watchlist.keys())
        return self.watchlist[latest_date]


# ==============================================================================
# 11. POSITION MANAGER (V12.7)
# ==============================================================================
class PositionManager:
    """
    Manages live positions using V12.7 bracket logic.

    Handles:
    - New entries from daily watchlist via Alpaca bracket orders
    - Daily SL/TP monitoring
    - Trailing stop updates after partial profit
    - Timeout exits
    - Regime-aware size scaling
    """

    def __init__(self, alpaca_helper, settings, db=None, mc_governor=None):
        self.alpaca = alpaca_helper
        self.settings = settings
        self.db = db
        self.mc_governor = mc_governor
        self.positions = {}
        self.recent_outcomes = {'LONG': [], 'SHORT': []}
        self.REGIME_LOOKBACK = 10
        self.REGIME_MIN_WR = 0.35

    def restore_positions_from_broker(self, polygon_helper):
        """On startup, rebuild position state from Alpaca's open positions."""
        try:
            self.alpaca.sync_positions()
            broker_positions = self.alpaca.pos_cache or {}

            for symbol, pos_data in broker_positions.items():
                if symbol in self.positions:
                    continue

                qty = abs(int(pos_data.get('qty', 0)))
                if qty == 0:
                    continue

                side = pos_data.get('side', 'long')
                direction = 'LONG' if side == 'long' else 'SHORT'
                entry_price = float(pos_data.get('avg_entry_price', 0))

                # Get daily ATR for this ticker
                daily_atr = ATR_CACHE.get(symbol)
                if not daily_atr:
                    try:
                        daily_bars = polygon_helper.fetch_daily_data(symbol, days=30)
                        if daily_bars is not None and len(daily_bars) > 20:
                            tr = daily_bars[['High', 'Low', 'Close']].copy()
                            tr['TR'] = pd.concat([
                                tr['High'] - tr['Low'],
                                (tr['High'] - tr['Close'].shift()).abs(),
                                (tr['Low'] - tr['Close'].shift()).abs()
                            ], axis=1).max(axis=1)
                            daily_atr = float(tr['TR'].rolling(20).mean().iloc[-1])
                    except Exception:
                        daily_atr = entry_price * 0.02  # Fallback: 2% ATR

                if not daily_atr or daily_atr <= 0:
                    daily_atr = entry_price * 0.02

                sl_dist = self.settings['V12_SL_ATR_MULT'] * daily_atr

                if direction == 'LONG':
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + sl_dist * self.settings['V12_TP_RR']
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - sl_dist * self.settings['V12_TP_RR']

                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_date': datetime.date.today() - datetime.timedelta(days=1),
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'sl_dist': sl_dist,
                    'daily_atr': daily_atr,
                    'conviction': 0.5,
                    'partial_taken': False,
                    'best_price': entry_price,
                    'size': 1.0,
                    'qty': qty,
                }
                logging.info(f"Restored position: {direction} {symbol} x{qty} @ {entry_price:.2f}")

            logging.info(f"Restored {len(self.positions)} positions from broker")
        except Exception as e:
            logging.warning(f"Position restore failed: {e}")

    def enter_positions(self, signals, polygon_helper):
        """Enter new positions from daily watchlist signals."""
        sl_mult = self.settings['V12_SL_ATR_MULT']
        tp_rr = self.settings['V12_TP_RR']
        risk_pct = self.settings['V12_RISK_PER_TRADE']

        for direction_key in ['longs', 'shorts']:
            direction = 'LONG' if direction_key == 'longs' else 'SHORT'

            for ticker, conviction in signals.get(direction_key, []):
                # Skip if already in position
                if ticker in self.positions:
                    continue
                broker_pos = self.alpaca.pos_cache or {}
                if ticker in broker_pos:
                    continue

                # Regime gate: reduce size if recent direction is failing
                regime_size = 1.0
                recent = self.recent_outcomes.get(direction, [])
                if len(recent) >= self.REGIME_LOOKBACK:
                    recent_wr = sum(recent) / len(recent)
                    if recent_wr < self.REGIME_MIN_WR:
                        regime_size = 0.5

                # Get daily ATR for bracket sizing
                daily_atr = ATR_CACHE.get(ticker)
                current_price = None

                if not daily_atr:
                    try:
                        daily_bars = polygon_helper.fetch_daily_data(ticker, days=30)
                        if daily_bars is not None and len(daily_bars) > 20:
                            tr = daily_bars[['High', 'Low', 'Close']].copy()
                            tr['TR'] = pd.concat([
                                tr['High'] - tr['Low'],
                                (tr['High'] - tr['Close'].shift()).abs(),
                                (tr['Low'] - tr['Close'].shift()).abs()
                            ], axis=1).max(axis=1)
                            daily_atr = float(tr['TR'].rolling(20).mean().iloc[-1])
                            current_price = float(daily_bars['Close'].iloc[-1])
                    except Exception as e:
                        logging.warning(f"ATR fetch failed for {ticker}: {e}")
                        continue

                if not daily_atr or daily_atr <= 0 or not np.isfinite(daily_atr):
                    continue

                # Get current price if not already obtained
                if current_price is None:
                    try:
                        snap = polygon_helper.fetch_snapshot_prices([ticker])
                        if ticker in snap:
                            current_price = snap[ticker]['price']
                    except Exception:
                        pass

                if current_price is None:
                    try:
                        daily_bars = polygon_helper.fetch_daily_data(ticker, days=5)
                        if daily_bars is not None and len(daily_bars) > 0:
                            current_price = float(daily_bars['Close'].iloc[-1])
                    except Exception:
                        pass

                if current_price is None or current_price <= 0:
                    logging.warning(f"Cannot get price for {ticker}, skipping")
                    continue

                # Compute bracket levels using DAILY ATR
                sl_dist = sl_mult * daily_atr
                tp_dist = sl_mult * tp_rr * daily_atr

                if direction == 'LONG':
                    sl_price = current_price - sl_dist
                    tp_price = current_price + tp_dist
                else:
                    sl_price = current_price + sl_dist
                    tp_price = current_price - tp_dist

                # Position sizing: risk_pct of equity / SL distance
                equity = self.alpaca.equity
                risk_dollars = equity * risk_pct * regime_size

                # Monte Carlo governor adjustment
                if self.mc_governor:
                    self.mc_governor.apply_adjustments()
                    risk_dollars *= self.mc_governor.get_risk_scalar()

                qty = int(risk_dollars / sl_dist) if sl_dist > 0 else 0
                if qty <= 0:
                    continue

                # Max position size: 20% of equity
                max_qty = int(equity * 0.20 / current_price)
                qty = min(qty, max_qty)

                # Submit bracket order via Alpaca
                side = direction
                success = self.alpaca.submit_bracket(
                    ticker, side, qty, current_price,
                    round(sl_price, 2), round(tp_price, 2),
                    atr_override=daily_atr,
                )

                if success:
                    self.positions[ticker] = {
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_date': datetime.date.today(),
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'sl_dist': sl_dist,
                        'daily_atr': daily_atr,
                        'conviction': conviction,
                        'partial_taken': False,
                        'best_price': current_price,
                        'size': regime_size,
                        'qty': qty,
                    }
                    logging.info(f"Entered {direction} {ticker} x{qty} @ {current_price:.2f} "
                                 f"SL={sl_price:.2f} TP={tp_price:.2f}")
                    send_alert(
                        f"ENTRY: {direction} {ticker}",
                        f"Qty: {qty} | Entry: ${current_price:.2f} | SL: ${sl_price:.2f} | TP: ${tp_price:.2f}",
                        "trade"
                    )

    def update_positions(self, polygon_helper):
        """Daily position management: trailing stops, partials, timeouts."""
        today = datetime.date.today()
        to_close = []

        # Sync with broker first
        try:
            self.alpaca.sync_positions()
        except Exception as e:
            logging.warning(f"Position sync failed: {e}")

        broker_positions = self.alpaca.pos_cache or {}

        for ticker, pos in list(self.positions.items()):
            # If broker closed the position (SL/TP hit), clean up
            if ticker not in broker_positions:
                to_close.append((ticker, 'broker_closed'))
                continue

            # Get current price
            current_price = None
            try:
                snap = polygon_helper.fetch_snapshot_prices([ticker])
                if ticker in snap:
                    current_price = snap[ticker]['price']
            except Exception:
                pass

            if current_price is None:
                try:
                    bp = broker_positions.get(ticker, {})
                    current_price = float(bp.get('current_price', 0))
                except Exception:
                    pass

            if not current_price or current_price <= 0:
                continue

            # Count days held
            days_held = (today - pos['entry_date']).days
            trading_days = max(1, int(days_held * 5 / 7))

            # Update best price for trailing
            if pos['direction'] == 'LONG':
                if current_price > pos['best_price']:
                    pos['best_price'] = current_price
            else:
                if current_price < pos['best_price']:
                    pos['best_price'] = current_price

            # Check partial profit
            partial_atr = self.settings['V12_PARTIAL_EXIT_ATR']
            if not pos['partial_taken']:
                if pos['direction'] == 'LONG':
                    partial_level = pos['entry_price'] + partial_atr * pos['daily_atr']
                    if current_price >= partial_level:
                        pos['partial_taken'] = True
                        pos['sl_price'] = pos['entry_price']  # Move to breakeven
                        self._update_stop_at_broker(ticker, pos['sl_price'], pos['direction'])
                        logging.info(f"Partial taken on {ticker}, SL moved to breakeven")
                else:
                    partial_level = pos['entry_price'] - partial_atr * pos['daily_atr']
                    if current_price <= partial_level:
                        pos['partial_taken'] = True
                        pos['sl_price'] = pos['entry_price']
                        self._update_stop_at_broker(ticker, pos['sl_price'], pos['direction'])
                        logging.info(f"Partial taken on {ticker}, SL moved to breakeven")

            # Trailing stop (after partial)
            if pos['partial_taken']:
                if pos['direction'] == 'LONG':
                    trail_level = pos['best_price'] - 1.0 * pos['daily_atr']
                    if trail_level > pos['sl_price']:
                        pos['sl_price'] = trail_level
                        self._update_stop_at_broker(ticker, trail_level, 'LONG')
                        logging.info(f"Trail SL updated for {ticker}: {trail_level:.2f}")
                else:
                    trail_level = pos['best_price'] + 1.0 * pos['daily_atr']
                    if trail_level < pos['sl_price']:
                        pos['sl_price'] = trail_level
                        self._update_stop_at_broker(ticker, trail_level, 'SHORT')
                        logging.info(f"Trail SL updated for {ticker}: {trail_level:.2f}")

            # Timeout exit
            if trading_days >= self.settings['V12_MAX_HOLD_DAYS']:
                logging.info(f"Timeout exit for {ticker} after {trading_days} trading days")
                try:
                    self.alpaca.close(ticker, reason="V12 timeout")
                except Exception as e:
                    logging.warning(f"Timeout close failed for {ticker}: {e}")
                to_close.append((ticker, 'timeout'))

        # Clean up closed positions and track outcomes
        for ticker, reason in to_close:
            pos = self.positions.pop(ticker, None)
            if pos:
                # Calculate PnL from broker data
                try:
                    bp = broker_positions.get(ticker, {})
                    exit_price = float(bp.get('current_price', pos['entry_price']))
                    if pos['direction'] == 'LONG':
                        pnl_r = (exit_price - pos['entry_price']) / pos['sl_dist']
                    else:
                        pnl_r = (pos['entry_price'] - exit_price) / pos['sl_dist']

                    self.recent_outcomes[pos['direction']].append(1 if pnl_r > 0 else 0)
                    if len(self.recent_outcomes[pos['direction']]) > self.REGIME_LOOKBACK:
                        self.recent_outcomes[pos['direction']].pop(0)

                    # Log to database
                    if self.db:
                        self.db.log_trade_outcome(
                            ticker, pos['direction'], pos['entry_price'], exit_price,
                            pnl_r * pos['sl_dist'] * pos['qty'], pnl_r, reason
                        )

                    # Log to MC governor
                    if self.mc_governor:
                        risk_dollars = pos['sl_dist'] * pos['qty']
                        pnl_dollars = pnl_r * risk_dollars
                        self.mc_governor.add_trade(pnl_dollars, risk_dollars, pos['direction'])

                    logging.info(f"Closed {pos['direction']} {ticker}: PnL={pnl_r:.2f}R ({reason})")
                    send_alert(
                        f"EXIT: {pos['direction']} {ticker}",
                        f"PnL: {pnl_r:.2f}R | Reason: {reason}",
                        "trade"
                    )
                except Exception as e:
                    logging.warning(f"PnL calc failed for {ticker}: {e}")

    def _update_stop_at_broker(self, ticker, new_sl, direction):
        """Update the stop-loss order at Alpaca."""
        try:
            orders = self.alpaca.api.list_orders(status='open')
            for order in orders:
                if order.symbol != ticker:
                    continue
                if order.type != 'stop':
                    continue
                # For a LONG position, the stop order is a SELL
                # For a SHORT position, the stop order is a BUY
                expected_side = 'sell' if direction == 'LONG' else 'buy'
                if order.side == expected_side:
                    self.alpaca.api.replace_order(
                        order.id,
                        stop_price=str(round(new_sl, 2)),
                        qty=str(order.qty),
                    )
                    logging.info(f"Updated stop for {ticker} to {new_sl:.2f}")
                    return True
        except Exception as e:
            logging.warning(f"Failed to update stop for {ticker}: {e}")
        return False

    def get_position_display(self):
        """Get positions formatted for dashboard display."""
        today = datetime.date.today()
        display = []
        for t, pos in self.positions.items():
            display.append({
                'symbol': t,
                'side': pos['direction'],
                'entry': pos['entry_price'],
                'sl': pos['sl_price'],
                'days_held': (today - pos['entry_date']).days,
                'partial_taken': pos['partial_taken'],
            })
        return display


# ==============================================================================
# 12. MAIN LOOP — V12.7 HYBRID BOT
# ==============================================================================
def run_hybrid_bot():
    """
    V12.7 Hybrid Bot Main Loop.

    Schedule:
    - 4:15 PM ET: Download daily bars, compute features, train model
    - 4:30 PM ET: Generate tomorrow's watchlist
    - 9:30 AM ET: Enter positions from watchlist
    - During market: Monitor positions (trailing stops, partials) every 30s
    - 3:45 PM ET: Check for timeout exits
    """
    print("=" * 60)
    print("V12.7 HYBRID BOT — DAILY ALPHA + INTRADAY EXECUTION")
    print("=" * 60)

    # Initialize infrastructure
    db = Database_Helper()
    poly = Polygon_Helper()
    mc_governor = MonteCarloGovernor(SETTINGS)
    daily_risk = DailyRiskManager(max_daily_loss_pct=0.02)
    earnings = EarningsGuard()
    dashboard = Dashboard()
    alpaca = Alpaca_Helper(db, mc_governor)

    # V12.7 ticker universe
    TICKERS = [
        'RKLB', 'ASTS', 'AMD', 'NVDA', 'PLTR', 'COIN',
        'GS', 'GE', 'COST', 'JPM', 'UNH', 'CAT',
        'XOM', 'JNJ', 'PG',
    ]

    # Initialize V12.7 engines
    signal_engine = DailySignalEngine(TICKERS, poly, SETTINGS)
    position_mgr = PositionManager(alpaca, SETTINGS, db=db, mc_governor=mc_governor)

    # State tracking
    last_model_train = None
    todays_signals = {}
    entries_done_today = False
    last_equity_log = datetime.datetime.now()
    last_position_check = datetime.datetime.now()

    print(f"Universe: {len(TICKERS)} tickers")
    print(f"Risk per trade: {SETTINGS['V12_RISK_PER_TRADE'] * 100:.1f}%")
    print(f"SL: {SETTINGS['V12_SL_ATR_MULT']:.1f} ATR | "
          f"TP: {SETTINGS['V12_SL_ATR_MULT'] * SETTINGS['V12_TP_RR']:.1f} ATR "
          f"(R:R 1:{SETTINGS['V12_TP_RR']:.1f})")
    print(f"Hold: {SETTINGS['V12_MAX_HOLD_DAYS']} days | TopN: {SETTINGS['V12_TOP_N']}")

    # Restore any existing positions from broker
    print("\nRestoring positions from broker...")
    position_mgr.restore_positions_from_broker(poly)

    # Initial model training
    print("\nInitial model training (this takes 2-5 minutes)...")
    signal_engine.update_daily_data()
    if signal_engine.train_model():
        signal_engine.generate_signals()
        todays_signals = signal_engine.get_tomorrow_signals()
        last_model_train = datetime.date.today()
        n_days = len(signal_engine.watchlist)
        print(f"Model trained. Watchlist has {n_days} days of signals.")
        if todays_signals:
            longs = todays_signals.get('longs', [])
            shorts = todays_signals.get('shorts', [])
            print(f"Next signals: {len(longs)} longs, {len(shorts)} shorts")
    else:
        print("WARNING: Model training failed. Will retry after market close.")

    while True:
        try:
            # Check kill switch
            if alpaca.check_kill():
                time.sleep(60)
                continue

            # Get current time in ET
            now_et = datetime.datetime.now(ET)
            today = datetime.date.today()

            # Check market hours via Alpaca clock
            try:
                clock = alpaca.api.get_clock()
                is_open = clock.is_open
            except Exception:
                is_open = (now_et.weekday() < 5 and
                           9 * 60 + 30 <= now_et.hour * 60 + now_et.minute < 16 * 60)

            # Refresh equity
            try:
                alpaca.refresh_equity()
            except Exception:
                pass

            # ── AFTER HOURS: TRAIN MODEL + GENERATE SIGNALS ──
            if (now_et.hour == 16 and now_et.minute >= 15 and
                    last_model_train != today):

                print(f"\n[{now_et.strftime('%H:%M')}] After-hours: Updating model...")
                signal_engine.update_daily_data()

                if signal_engine.train_model():
                    signal_engine.generate_signals()
                    todays_signals = signal_engine.get_tomorrow_signals()
                    last_model_train = today
                    entries_done_today = False

                    if todays_signals:
                        longs = todays_signals.get('longs', [])
                        shorts = todays_signals.get('shorts', [])
                        print(f"Tomorrow's watchlist:")
                        for t, s in longs:
                            print(f"  LONG  {t} (score: {s:.4f})")
                        for t, s in shorts:
                            print(f"  SHORT {t} (score: {s:.4f})")
                        send_alert(
                            "V12.7 Watchlist Generated",
                            f"Longs: {[t for t,_ in longs]}\nShorts: {[t for t,_ in shorts]}",
                            "normal"
                        )
                    else:
                        print("No signals for tomorrow (low conviction day)")
                else:
                    print("Model training failed. Will retry tomorrow.")

                time.sleep(60)
                continue

            # ── PRE-MARKET ──
            if not is_open:
                # Reset daily flags at midnight
                if now_et.hour == 0 and now_et.minute < 2:
                    entries_done_today = False
                    daily_risk.initialize_session(alpaca.equity)

                time.sleep(60)
                continue

            # ── MARKET OPEN: INITIALIZE DAILY RISK ──
            if daily_risk.session_start_equity is None:
                daily_risk.initialize_session(alpaca.equity)

            # ── MARKET OPEN: ENTER POSITIONS (9:30 - 10:00 AM) ──
            if (now_et.hour == 9 and now_et.minute >= 30 and
                    not entries_done_today and todays_signals):

                # Wait a few minutes for opening noise to settle
                if now_et.minute >= 35:
                    # Filter out earnings tickers
                    safe_signals = {}
                    for key in ['longs', 'shorts']:
                        safe_signals[key] = [
                            (t, s) for t, s in todays_signals.get(key, [])
                            if earnings.check_safe(t)
                        ]

                    if safe_signals.get('longs') or safe_signals.get('shorts'):
                        print(f"\n[{now_et.strftime('%H:%M')}] Entering positions...")

                        if daily_risk.check_can_trade(alpaca.equity):
                            position_mgr.enter_positions(safe_signals, poly)
                        else:
                            print("Daily loss limit reached — skipping entries")

                    entries_done_today = True

            # ── DURING MARKET: MANAGE POSITIONS ──
            now = datetime.datetime.now()
            if is_open and (now - last_position_check).total_seconds() >= 60:
                position_mgr.update_positions(poly)
                last_position_check = now

            # ── EQUITY LOG (every 5 minutes) ──
            if is_open and (now - last_equity_log).total_seconds() >= 300:
                db.log_equity(alpaca.equity)
                last_equity_log = now

            # ── DASHBOARD UPDATE ──
            if is_open:
                # Build watchlist display
                watchlist_display = []
                if todays_signals:
                    for t, s in todays_signals.get('longs', []):
                        watchlist_display.append({'symbol': t, 'side': 'LONG', 'score': s})
                    for t, s in todays_signals.get('shorts', []):
                        watchlist_display.append({'symbol': t, 'side': 'SHORT', 'score': s})

                logs = []
                if position_mgr.mc_governor:
                    scalar = position_mgr.mc_governor.get_risk_scalar()
                    if scalar < 1.0:
                        logs.append(f"MC Governor: risk scalar = {scalar:.2f}")

                long_wr = position_mgr.recent_outcomes.get('LONG', [])
                short_wr = position_mgr.recent_outcomes.get('SHORT', [])
                if long_wr:
                    logs.append(f"Recent L WR: {sum(long_wr)/len(long_wr):.0%} ({len(long_wr)} trades)")
                if short_wr:
                    logs.append(f"Recent S WR: {sum(short_wr)/len(short_wr):.0%} ({len(short_wr)} trades)")

                dashboard.render({
                    'equity': alpaca.equity,
                    'universe_size': len(TICKERS),
                    'pnl_day': alpaca.equity - (daily_risk.session_start_equity or alpaca.equity),
                    'positions': position_mgr.get_position_display(),
                    'watchlist': watchlist_display,
                    'logs': logs,
                })

            time.sleep(30)  # Check every 30 seconds during market hours

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    run_hybrid_bot()
