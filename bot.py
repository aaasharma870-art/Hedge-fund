# ==============================================================================
# GOD MODE HEDGE FUND MONITOR (v14.3 - PRODUCTION)
# "Hybrid God Mode: Specialist + Grinder"
#
# CRITICAL UPGRADES (v14.3):
#   1. "One Brain" Architecture: Imports core logic from hedge_fund/ package
#   2. Hybrid Strategy: TIER_1 (Specialist) and TIER_2 (Grinder) execution
#   3. Monte Carlo Governor: Dynamic risk scaling based on equity drawdown
#   4. Portfolio Optimizer: Mean-Variance allocation with Beta constraints
#   5. Attribution Analysis: ML-driven pattern recognition for trades
# ==============================================================================

# --- COLAB SETUP ---
try:
    import google.colab
    import subprocess
    import sys
    import os
    import time

    # 1. Check NumPy Version (Crucial for GPU/XGBoost compatibility)
    # 1. Strict Version Control (Prevent Dependency Hell)
    try:
        import numpy as np
        import pandas as pd
        import scipy

        # Check for ANY incompatible versions
        bad_np = np.__version__.startswith('2.')
        bad_pd = pd.__version__ != '2.2.2' # FIX: Match pinned version in installer

        if bad_np or bad_pd:
            print(f"⚠️ Incompatible environment detected (NumPy {np.__version__}, Pandas {pd.__version__}).")
            print("⏳ Aligning core dependencies (Allocating ~60s)...", flush=True)

            # NUKE AND PAV: Uninstall everything first to clear bad caches
            # Use os.system to ensure output is VISIBLE to user
            os.system(f"{sys.executable} -m pip uninstall -y numpy pandas scipy pandas_ta")

            # INSTALL CLEAN TRIAD (Updated for Py3.10+ wheels)
            # Numpy 1.26.4 + Pandas 2.2.2 + Scipy 1.13.1 = GOLDEN RATIO for Colab
            cmd = (
                f"{sys.executable} -m pip install "
                "numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 "
                "--force-reinstall --no-cache-dir --only-binary=:all:"
            )
            print(f"▶️ Executing: {cmd}", flush=True)
            ret = os.system(cmd)

            if ret != 0:
                print("⚠️ Binary install failed. Trying standard install...", flush=True)
                os.system(f"{sys.executable} -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 --force-reinstall")

            print("✅ Core stack aligned. 🔄 AUTO-RESTARTING...", flush=True)
            time.sleep(1)
            os.kill(os.getpid(), 9)

    except ImportError:
        # Initial install
        os.system(f"{sys.executable} -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 --only-binary=:all:")

    print(f"✅ Core: NumPy {np.__version__} | Pandas {pd.__version__} | Scipy {scipy.__version__}")

    print("📦 Installing Apps...", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas_ta'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'xgboost'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'alpaca-trade-api', 'pyarrow', 'yfinance'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'websockets'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tzdata'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'rich'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'optuna', 'joblib'], check=True)

    print("✅ Dependencies ready!")

    # GPU CHECK
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
        import numpy as np
        print(f"🔎 XGBoost Version: {xgb.__version__}")
        # Quick GPU Test
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        try:
            # Modern XGBoost (2.0+) uses device="cuda"
            model = XGBRegressor(device="cuda", verbosity=2) # Verbose=2 to show CUDA info
            model.fit(X, y)
            print("✅ GPU DETECTED & WORKING (device='cuda')")
            print("   (GPU usage will stay low until TRAINING starts)")
        except Exception:
            try:
                # Legacy XGBoost uses tree_method='gpu_hist'
                model = XGBRegressor(tree_method='gpu_hist', verbosity=2)
                model.fit(X, y)
                print("✅ GPU DETECTED & WORKING (tree_method='gpu_hist')")
            except Exception as e:
                print(f"❌ GPU INIT FAILED: {e}")
                print("⚠️ Falling back to CPU mode.")
    except Exception as e:
        print(f"Warning during GPU check: {e}")
except ImportError:
    pass
except SystemExit as e:
    print(str(e))

import sys
import os
# Fix Windows Unicode Encode Error
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables.")

print("\n" + "="*60)
print("🚀 LAUNCHING HEDGE FUND MONITOR...")
print("="*60 + "\n")

import threading, time, json, sqlite3, logging, warnings, gc, math, traceback, random
import datetime
from datetime import timedelta, timezone
import concurrent.futures, requests
import optuna
import numpy as np
import pandas as pd
import importlib.metadata # FIX: Required for pandas_ta-openbb on some envs
try:
    import pandas_ta as ta
except ImportError:
    import pandas_ta_openbb as ta
from collections import defaultdict, deque
import joblib

# Scientific / ML
from scipy.stats import beta, norm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBRegressor
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
try:
    import websocket as ws_lib
    HAS_WS = True
except ImportError:
    HAS_WS = False

# Trading / Data
import alpaca_trade_api as tradeapi
import yfinance as yf
from logging.handlers import RotatingFileHandler

# --- "ONE BRAIN" IMPORTS ---
from hedge_fund.governance import MonteCarloGovernor
from hedge_fund.optimization import PortfolioOptimizer
from hedge_fund.risk import OvernightGapModel, SlippageCalculator
from hedge_fund.analysis import run_attribution_analysis
from hedge_fund.features import CrossSectionalRanker
from hedge_fund.config import load_optimal_params, apply_to_settings
from hedge_fund.dashboard import Dashboard as SharedDashboard
from hedge_fund.scanner import CandidateScanner

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
# Configure logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

NUM_CORES = os.cpu_count() or 4
IO_WORKERS = min(32, NUM_CORES * 4)  # High concurrency for API calls
CPU_WORKERS = NUM_CORES              # Full utilization for Training

# Force XGBoost to use all cores if on CPU
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)

# --- Timezone helpers (US/Eastern canonical) ---
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    try:
        from dateutil.tz import gettz
        ET = gettz("America/New_York")
    except ImportError:
        ET = datetime.timezone(datetime.timedelta(hours=-5))  # naive EST fallback

def is_regular_hours(ts):
    """True if timestamp falls within 9:30-16:00 ET on a weekday."""
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        ts_et = ts.astimezone(ET)
    elif hasattr(ts, 'tz_localize'):
        try:
            ts_et = ts.tz_localize(ET)
        except Exception:
            ts_et = ts  # already localized or naive
    else:
        ts_et = ts
    h, m = ts_et.hour, ts_et.minute
    if ts_et.weekday() >= 5:
        return False
    if h < 9 or h >= 16:
        return False
    if h == 9 and m < 30:
        return False
    return True

def reserve_gpu():
    """Wake up the GPU and reserve memory so it shows as 'Active' immediately."""
    try:
        import torch
        if torch.cuda.is_available():
            # Allocate 500MB dummy tensor to light up the dashboard
            _ = torch.ones((10000, 10000)).cuda()
            logging.info(f"🔥 GPU ACTIVATED: {torch.cuda.get_device_name(0)} (VRAM Reserved)")
            return True
    except:
        pass
    return False

# Trigger immediate wake-up
HAS_GPU = reserve_gpu()
logging.info(f"🔥 HARDWARE: {NUM_CORES} CS | {IO_WORKERS} I/O Workers | Optimized for Colab Pro+")


# ==============================================================================
# 1. INFRASTRUCTURE
# ==============================================================================


# (shared modules already imported above in "ONE BRAIN" section)




# --- MAIN BOT LOGIC ---
# --- STORAGE SETUP ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import sys, subprocess
    # FIX: Install optuna/joblib for Colab if missing
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna', 'joblib'], check=True)

    ENV = 'colab'
    DRIVE_ROOT = "/content/drive/MyDrive/HedgeFund_GodMode"
except ImportError:
    ENV = 'local'
    DRIVE_ROOT = "data"

if not os.path.exists(DRIVE_ROOT): os.makedirs(DRIVE_ROOT, exist_ok=True)
for d in ["logs", "market_cache", "models", "db"]:
    os.makedirs(os.path.join(DRIVE_ROOT, d), exist_ok=True)

# API Keys - env vars only (no hardcoded secrets)
def require_env(name):
    """Retrieve a required environment variable or raise on critical keys."""
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
    "RAPIDAPI_KEY": os.getenv("RAPIDAPI_KEY", ""),
}
ALPACA_URL = "https://paper-api.alpaca.markets"

SETTINGS = {
    "RISK_PER_TRADE": 0.015,  # 1.5% Risk per trade
    "MAX_POSITIONS": 5,
    "STOP_MULT": 1.5,
    "TP_MULT": 3.0,
    "MIN_CONFIDENCE": 0.02, # OPTIMIZED: Proven Positive Expectancy
    "TIER_A_CONF": 0.53,
    "MAX_PORTFOLIO_HEAT": 0.06,
    "KILL_SWITCH_DD": 0.10,
    "SCAN_INTERVAL": 60,
    "ZOMBIE_MINUTES": 60,
    "PYRAMID_R": 1.5,
    "HEDGE_TICKER": "PSQ",
    "MIN_HURST": 0.38,
    "MIN_HITRATE": 0.35,
    "USE_MARKET_ORDERS": True,
    "SLIPPAGE_HAIRCUT": 0.10,

    "OPEN_BURNIN_MINUTES": 15,
    "BURNIN_CONF_ADD": 0.03,
    "BURNIN_SIZE_MULT": 0.60,

    "NEWS_LOOKBACK_HOURS": 24,
    "NEWS_HARD_SKIP_SCORE": 3,
    "NEWS_SOFT_PENALTY_SCORE": 1,
    "NEWS_PENALTY_CONF_ADD": 0.02,
    "NEWS_PENALTY_SIZE_MULT": 0.75,
    "NEWS_OVERRIDE_CONF": 0.80,

    "RATCHET_R": 1.0,
    "RATCHET_BUFFER_ATR": 0.10,
    "TRAIL_START_R": 2.0,
    "TRAIL_ATR_MULT": 1.0,
    "STOP_REPLACE_COOLDOWN_SEC": 60,
    "MARKET_OPEN_HOUR": 9,
    "MARKET_OPEN_MIN": 30,
    "MARKET_CLOSE_HOUR": 16,
    "MARKET_CLOSE_MIN": 0,
    "ENABLE_PYRAMIDING": False,
    "USE_KELLY": True,
    "KELLY_FRACTION": 0.75,
    "KELLY_MAX_RISK": 0.03,
    "KELLY_MIN_RISK": 0.003,
    "KELLY_MIN_TRADES": 60,
    "KELLY_EDGE_GATE": 0.02,
    "KELLY_CONF_BOOST": 0.50,
    "KELLY_SHRINK": 0.35,
    "STOP_MULT": 1.5,
    "TP_MULT": 3.0,

    # Phase 23: Hybrid God Mode (Dual-Tier)
    # Tier 1: "The Specialist" (Sniper Entry) - Maximize PF
    "TIER_1": {
        "NAME": "SPECIALIST",
        "MIN_PROB": 0.30,
        "MAX_HURST": 0.55,
        "MIN_ADX": 25,
        "RISK_MULT": 2.0,       # Double Size
        "RR": 2.0,              # 1:2 Risk/Reward
        "TRAIL": 1.0
    },
    # Tier 2: "The Grinder" (Standard Entry) - Maximize Volume
    "TIER_2": {
        "NAME": "GRINDER",
        "MIN_PROB": 0.20,
        "MAX_HURST": 0.45,
        "MIN_ADX": 20,
        "RISK_MULT": 1.0,       # Standard Size
        "RR": 1.5,              # 1:1.5 Risk/Reward
        "TRAIL": 1.0
    },

    "EV_GATE_R": 0.02,           # Lowered: let more candidates through initial gate
    "TIER_A_EV": 0.10,           # Lowered: 0.10R EV is still a positive-expectancy trade
    "TIER_A_CONF": 0.51,         # Lowered: direction-aware model is more calibrated
}

# --- Auto-load optimized params from backtester ---
_optimal = load_optimal_params()
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
        RotatingFileHandler(os.path.join(DRIVE_ROOT, "logs/bot.log"), maxBytes=1e7, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
warnings.filterwarnings('ignore')

# Error tracking
class ErrorTracker:
    def __init__(self):
        self.failures = defaultdict(int)
        self.last_success = defaultdict(lambda: datetime.datetime.now())

    def record_failure(self, component, error):
        self.failures[component] += 1
        logging.error(f"❌ {component} failed ({self.failures[component]}x): {error}")
        if self.failures[component] >= 5:
            logging.critical(f"🚨 {component} has failed {self.failures[component]} times!")
            send_alert(f"🚨 {component} BROKEN", f"Failed {self.failures[component]}x", "high")

    def record_success(self, component):
        if self.failures[component] > 0:
            logging.info(f"✅ {component} recovered")
        self.failures[component] = 0
        self.last_success[component] = datetime.datetime.now()

ERROR_TRACKER = ErrorTracker()

# ==============================================================================
# 2. DATABASE
# ==============================================================================
class Database_Helper:
    def __init__(self):
        db_path = os.path.join(DRIVE_ROOT, 'db/godmode.db')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()

        # FIX: Enable WAL mode and NORMAL sync for reliability under frequent writes
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self.create_tables()
        logging.info(f"💾 Database: {db_path}")

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
            # FIX: Persist pending orders for restart recovery
            c.execute('''CREATE TABLE IF NOT EXISTS pending_orders
                         (order_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty INT,
                          price REAL, sl REAL, tp REAL, atr REAL, ts TEXT)''')
            # ENHANCED: Trade outcomes for learning from mistakes (expanded schema)
            c.execute('''CREATE TABLE IF NOT EXISTS trade_outcomes
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          symbol TEXT, side TEXT,
                          entry_price REAL, exit_price REAL,
                          pnl REAL, pnl_r REAL,
                          outcome REAL,
                          rsi REAL, adx REAL, atr_pct REAL,
                          vol_rel REAL, kalman_dist REAL, hurst REAL,
                          bb_width REAL, bb_position REAL, vwap_dist REAL, hl_range REAL,
                          roc_5 REAL, roc_20 REAL,
                          vol_surge REAL, money_flow REAL,
                          volatility_rank REAL, trend_consistency REAL,
                          sa_news_count_3d INT, sa_sentiment_score REAL,
                          hour INT, day_of_week INT,
                          reason TEXT,
                          ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

            # ENHANCED: Add new feature columns to existing tables (migration)
            try:
                new_cols = [
                    'bb_width REAL', 'bb_position REAL', 'vwap_dist REAL', 'hl_range REAL',
                    'roc_5 REAL', 'roc_20 REAL', 'vol_surge REAL', 'money_flow REAL',
                    'volatility_rank REAL', 'trend_consistency REAL',
                    'sa_news_count_3d INT', 'sa_sentiment_score REAL',
                    'hour INT', 'day_of_week INT',
                    'earnings_surprise REAL', 'revenue_growth_yoy REAL',
                    'pe_ratio REAL', 'news_impact_weight REAL'
                ]
                for col in new_cols:
                    try:
                        c.execute(f'ALTER TABLE trade_outcomes ADD COLUMN {col}')
                    except:
                        pass  # Column already exists
            except Exception as e:
                logging.debug(f"Schema migration: {e}")

            # ------------------------------------------------------------------
            # SECTION 1.1: Production Expansion (State, Orders, Fills, Events)
            # ------------------------------------------------------------------
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

    # --- Production DB Methods ---
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
            # Check if exists to preserve created_ts
            exists = self.conn.execute("SELECT created_ts FROM orders WHERE client_order_id=?", (client_order_id,)).fetchone()
            created_ts = exists[0] if exists else now

            self.conn.execute("""
                INSERT OR REPLACE INTO orders
                (client_order_id, broker_order_id, symbol, side, qty, type, status, created_ts, updated_ts, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (client_order_id, broker_id, symbol, side, qty, order_type, status, created_ts, now, raw_json))
            self.conn.commit()

    def upsert_bot_state(self, last_processed_ts=None, model_ver=None, feat_hash=None):
        with self.lock:
            # Simple merge update
            current = self.conn.execute("SELECT * FROM bot_state WHERE id=1").fetchone()
            if not current:
                self.conn.execute("INSERT INTO bot_state (id) VALUES (1)")
                current = (1, None, None, None, None, None)

            # Map cols: id, last_ts, mod_ver, feat_hash, risk_hash, up_at
            # Update only provided fields
            c_ts = last_processed_ts if last_processed_ts else current[1]
            c_mv = model_ver if model_ver else current[2]
            c_fh = feat_hash if feat_hash else current[3]

            self.conn.execute("""
                UPDATE bot_state SET last_processed_ts=?, model_version=?, feature_hash=?, updated_at=? WHERE id=1
            """, (c_ts, c_mv, c_fh, datetime.datetime.now().isoformat()))
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

    def log_trade_outcome(self, symbol, side, entry_price, exit_price, pnl, pnl_r, outcome, features, reason="Signal"):
        """
        ENHANCED: Log a trade outcome with ALL 22 entry features for learning.
        outcome: R-value (e.g., +2.0 for win, -1.0 for loss)
        features: dict with all 22 features
        """
        with self.lock:
            self.conn.execute(
                """INSERT INTO trade_outcomes
                   (symbol, side, entry_price, exit_price, pnl, pnl_r, outcome,
                    rsi, adx, atr_pct, vol_rel, kalman_dist, hurst,
                    bb_width, bb_position, vwap_dist, hl_range,
                    roc_5, roc_20, vol_surge, money_flow,
                    volatility_rank, trend_consistency,
                    sa_news_count_3d, sa_sentiment_score,
                    hour, day_of_week,
                    earnings_surprise, revenue_growth_yoy, pe_ratio, news_impact_weight,
                    reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, side, entry_price, exit_price, pnl, pnl_r, outcome,
                 features.get('RSI'), features.get('ADX'), features.get('ATR_Pct'),
                 features.get('Vol_Rel'), features.get('Kalman_Dist'), features.get('Hurst'),
                 features.get('BB_Width'), features.get('BB_Position'), features.get('VWAP_Dist'), features.get('HL_Range'),
                 features.get('ROC_5'), features.get('ROC_20'), features.get('Vol_Surge'), features.get('Money_Flow'),
                 features.get('Volatility_Rank'), features.get('Trend_Consistency'),
                 features.get('sa_news_count_3d'), features.get('sa_sentiment_score'),
                 features.get('Hour'), features.get('Day_of_Week'),
                 features.get('earnings_surprise', 0), features.get('revenue_growth_yoy', 0),
                 features.get('pe_ratio', 20), features.get('news_impact_weight', 0),
                 reason)
            )
            self.conn.commit()
            logging.debug(f"📊 Logged trade outcome: {symbol} {side} PnL={pnl:.2f} R={pnl_r:.2f}")

    def get_trade_outcomes(self, days=90):
        """
        ENHANCED: Retrieve trade outcomes with ALL 22 features for training.
        Returns list of dicts with features and outcome.
        """
        with self.lock:
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
            c = self.conn.cursor()
            c.execute("""
                SELECT symbol, side, entry_price, exit_price, pnl, pnl_r, outcome,
                       rsi, adx, atr_pct, vol_rel, kalman_dist, hurst,
                       bb_width, bb_position, vwap_dist, hl_range,
                       roc_5, roc_20, vol_surge, money_flow,
                       volatility_rank, trend_consistency,
                       sa_news_count_3d, sa_sentiment_score,
                       hour, day_of_week,
                       earnings_surprise, revenue_growth_yoy, pe_ratio, news_impact_weight,
                       ts
                FROM trade_outcomes
                WHERE ts > ?
                ORDER BY ts DESC
            """, (cutoff,))
            rows = c.fetchall()

            results = []
            for row in rows:
                # Calculate days ago for recency weighting
                try:
                    ts = pd.to_datetime(row[31]) if row[31] else datetime.datetime.now()
                    days_ago = (datetime.datetime.now() - ts).days
                except:
                    days_ago = 0

                results.append({
                    'symbol': row[0], 'side': row[1],
                    'entry_price': row[2], 'exit_price': row[3],
                    'pnl': row[4], 'pnl_r': row[5], 'outcome': row[6],
                    'RSI': row[7], 'ADX': row[8], 'ATR_Pct': row[9],
                    'Vol_Rel': row[10], 'Kalman_Dist': row[11], 'Hurst': row[12],
                    'BB_Width': row[13] or 0, 'BB_Position': row[14] or 0, 'VWAP_Dist': row[15] or 0, 'HL_Range': row[16] or 0,
                    'ROC_5': row[17] or 0, 'ROC_20': row[18] or 0, 'Vol_Surge': row[19] or 0, 'Money_Flow': row[20] or 0,
                    'Volatility_Rank': row[21] or 0.5, 'Trend_Consistency': row[22] or 0.5,
                    'sa_news_count_3d': row[23] or 0, 'sa_sentiment_score': row[24] or 0,
                    'Hour': row[25] or 12, 'Day_of_Week': row[26] or 2,
                    'earnings_surprise': row[27] or 0, 'revenue_growth_yoy': row[28] or 0,
                    'pe_ratio': row[29] or 20, 'news_impact_weight': row[30] or 0,
                    'days_ago': days_ago
                })
            return results

# ==============================================================================
# 3. MARKET DATA HELPERS
# ==============================================================================
class VIX_Helper:
    """VIX with multi-source fallback"""
    def __init__(self):
        self.cache = {'value': None, 'ts': 0}
        self.cache_ttl = 60
        self.data_valid = False

    def get_vix(self):
        now = time.time()
        if self.cache['value'] is not None and now - self.cache['ts'] < self.cache_ttl:
            return self.cache['value']

        vix = self._try_yfinance() or self._try_fmp()

        if vix is not None and 5 < vix < 100:
            self.cache = {'value': vix, 'ts': now}
            self.data_valid = True
            ERROR_TRACKER.record_success("VIX")
            return vix
        else:
            ERROR_TRACKER.record_failure("VIX", f"Invalid: {vix}")
            self.data_valid = False
            return 20.0

    def _try_yfinance(self):
        try:
            tk = yf.Ticker('^VIX')
            hist = tk.history(period='2d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logging.debug(f"yfinance VIX: {e}")
        return None

    def _try_fmp(self):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/%5EVIX?apikey={KEYS['FMP']}"
            res = requests.get(url, timeout=5).json()
            if res:
                return float(res[0].get('price', 0))
        except Exception as e:
            logging.debug(f"FMP VIX: {e}")
        return None

    def get_size_multiplier(self):
        vix = self.get_vix()
        if vix > 30: return 0.5
        if vix > 25: return 0.75
        if vix < 15: return 1.2
        return 1.0

# FIX #12: Regime detection using DAILY data (not minute)
class RegimeDetector:
    """
    Multi-dimensional regime detector with continuous scores.
    Dimensions: trend, chop, correlation, volatility.
    """
    def __init__(self, vix_helper, vol_target=None, portfolio_risk=None):
        self.vix_helper = vix_helper
        self.vol_target = vol_target
        self.portfolio_risk = portfolio_risk
        self.cache = {
            'regime': 'NEUTRAL', 'trend_score': 0.0, 'chop_score': 0.5,
            'corr_regime': 0.5, 'vol_regime': 0.5, 'conf_adj': 0.0, 'ts': 0
        }
        self.cache_ttl = 300

    def get_regime(self):
        now = time.time()
        if now - self.cache['ts'] < self.cache_ttl:
            return self.cache['regime']

        try:
            tk = yf.Ticker('SPY')
            spy_df = tk.history(period='1y', interval='1d')

            if len(spy_df) < 200:
                return 'NEUTRAL'

            close = spy_df['Close']
            high = spy_df['High']
            low = spy_df['Low']
            current = close.iloc[-1]
            vix = self.vix_helper.get_vix()

            # --- Trend Score (continuous -1 to +1) ---
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]

            price_vs_50 = (current - sma50) / sma50
            price_vs_200 = (current - sma200) / sma200
            sma_cross = (sma50 - sma200) / sma200

            trend_raw = (0.4 * np.clip(price_vs_50 * 10, -1, 1) +
                         0.3 * np.clip(price_vs_200 * 5, -1, 1) +
                         0.3 * np.clip(sma_cross * 10, -1, 1))

            vix_adj = 0.0
            if vix and vix < 15: vix_adj = 0.1
            elif vix and vix > 28: vix_adj = -0.2

            trend_score = float(np.clip(trend_raw + vix_adj, -1.0, 1.0))

            # --- Chop Score (Choppiness Index, 0 to 1) ---
            chop_score = 0.5
            period = 14
            atr_series = high - low
            if len(atr_series) >= period:
                atr_sum = atr_series.rolling(period).sum().iloc[-1]
                highest_14 = high.rolling(period).max().iloc[-1]
                lowest_14 = low.rolling(period).min().iloc[-1]
                hl_range = highest_14 - lowest_14
                if hl_range > 0:
                    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
                    chop_score = float(np.clip(ci / 100.0, 0.0, 1.0))

            # --- Correlation Regime ---
            corr_regime = 0.5
            if self.portfolio_risk and getattr(self.portfolio_risk, 'correlation_matrix', None) is not None:
                try:
                    cm = self.portfolio_risk.correlation_matrix.values
                    n = cm.shape[0]
                    if n > 1:
                        upper = cm[np.triu_indices(n, k=1)]
                        corr_regime = float(np.clip(np.nanmean(upper), 0.0, 1.0))
                except Exception:
                    pass

            # --- Volatility Regime (percentile rank) ---
            vol_regime = 0.5
            returns = close.pct_change().dropna()
            if len(returns) >= 60:
                current_vol = returns.tail(20).std() * np.sqrt(252)
                if self.vol_target and self.vol_target.sigma_hat:
                    current_vol = self.vol_target.sigma_hat
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                rolling_vol = rolling_vol.dropna()
                if len(rolling_vol) > 0:
                    rank = (rolling_vol < current_vol).sum() / len(rolling_vol)
                    vol_regime = float(np.clip(rank, 0.0, 1.0))

            # --- Composite Label (backward compat) ---
            if trend_score > 0.3 and chop_score < 0.6:
                regime = 'BULL'
            elif trend_score < -0.3:
                regime = 'BEAR'
            elif chop_score > 0.65:
                regime = 'CHOP'
            else:
                regime = 'NEUTRAL'

            # --- Composite Confidence Adjustment ---
            # FIX: Scaled up coefficients so regime meaningfully affects trade gating
            # Bull trend → lower threshold (more permissive), chop → raise threshold
            conf_adj = (-0.04 * trend_score +       # Bull: -0.04, Bear: +0.04
                        0.06 * chop_score +          # Choppy: +0.06 (require more confidence)
                        0.04 * max(0, corr_regime - 0.5) +  # High correlation: +0.02
                        0.04 * max(0, vol_regime - 0.6))     # High vol rank: +0.016
            conf_adj = float(np.clip(conf_adj, -0.10, 0.12))

            self.cache = {
                'regime': regime, 'trend_score': trend_score, 'chop_score': chop_score,
                'corr_regime': corr_regime, 'vol_regime': vol_regime,
                'conf_adj': conf_adj, 'ts': now
            }

            logging.info(f"📊 Regime: {regime} | Trend={trend_score:+.2f} Chop={chop_score:.2f} "
                         f"Corr={corr_regime:.2f} Vol={vol_regime:.2f} | ConfAdj={conf_adj:+.3f}")
            ERROR_TRACKER.record_success("Regime")
            return regime

        except Exception as e:
            ERROR_TRACKER.record_failure("Regime", str(e))
            return 'NEUTRAL'

    def get_confidence_adjustment(self):
        self.get_regime()
        return self.cache.get('conf_adj', 0.0)

    def get_scores(self):
        self.get_regime()
        return {
            'trend_score': self.cache['trend_score'],
            'chop_score': self.cache['chop_score'],
            'regime': self.cache['regime'],
            'corr': self.cache.get('corr_regime', 0.5),
            'vol': self.cache.get('vol_regime', 0.5)
        }

    def get_adaptive_threshold(self, base_conf=0.48):
        """
        Dynamic Confidence Threshold based on Regime.
        - Bull/Trend: Base (0.48)
        - Choppy: +0.05
        - High Vol (VIX > 25): +0.10
        - Bear/Crash: +0.12
        """
        regime = self.get_regime()
        vix = self.vix_helper.get_vix()

        adj = 0.0

        # 1. VIX Penalty
        if vix > 30: adj += 0.10
        elif vix > 20: adj += 0.05

        # 2. Regime Penalty
        # Note: The 'regime' variable is derived from trend_score and chop_score.
        # The specific string values 'BEAR_VOLATILE' and 'CHOPPY' are not directly
        # produced by the current get_regime() method, which returns 'BULL', 'BEAR', 'CHOP', 'NEUTRAL'.
        # Assuming these are intended as more granular states or future enhancements.
        if regime == 'BEAR': adj += 0.07 # Using 'BEAR' as a proxy for 'BEAR_VOLATILE'
        elif regime == 'CHOP': adj += 0.05 # Using 'CHOP' for 'CHOPPY'
        elif regime == 'NEUTRAL': adj += 0.02

        # 3. Trend Bonus (Discount)
        # If strong trend, we can be slightly looser?
        # No, stick to base. "Don't lose money" is priority.

        return base_conf + adj

# ==============================================================================
# DIX/GEX INTEGRATION: Gamma Exposure for sizing and entry gating
# ==============================================================================
class GEX_Helper:
    """
    DIX/GEX Proxy using VIX term structure and SMA.

    NOTE: This is a SIMULATED GEX based on VIX regime.
    True GEX requires option chain gamma exposure sums (not currently implemented).
    High Simulated GEX = low VIX = allow full size.
    """

    def __init__(self):
        self.cache = {'gex': 0, 'dix': 0.45, 'ts': None}
        self.gex_history = []  # For percentile calculations

    def fetch_data(self):
        """
        Estimate GEX/DIX using VIX term structure (VIX vs VIX3M).
        - VIX/VIX3M < 0.90 → contango → positive gamma → calm, mean-reverting
        - VIX/VIX3M > 1.00 → inverted → negative gamma → volatile, trend-following
        Falls back to VIX-vs-SMA if VIX3M unavailable.
        """
        now = datetime.datetime.now()

        if self.cache['ts'] and (now - self.cache['ts']).total_seconds() < 300:
            return self.cache['gex'], self.cache['dix']

        try:
            # Primary: VIX term structure ratio (better gamma proxy)
            vix_tk = yf.Ticker('^VIX')
            vix_data = vix_tk.history(period='5d', interval='1d')

            vix3m_tk = yf.Ticker('^VIX3M')
            vix3m_data = vix3m_tk.history(period='5d', interval='1d')

            if len(vix_data) >= 1 and len(vix3m_data) >= 1:
                vix = float(vix_data['Close'].iloc[-1])
                vix3m = float(vix3m_data['Close'].iloc[-1])

                if vix3m > 0:
                    # Term structure ratio: < 1 = contango (calm), > 1 = inverted (stress)
                    ratio = vix / vix3m
                    # Map ratio to GEX proxy: contango = positive GEX, backwardation = negative
                    gex_proxy = float(np.clip((1.0 - ratio) * 100, -30, 30))
                    # DIX proxy: low VIX + contango = bullish dark pool activity
                    dix_proxy = float(np.clip(0.50 - (vix - 18) / 100 + (1.0 - ratio) * 0.1, 0.30, 0.60))

                    self.cache = {'gex': gex_proxy, 'dix': dix_proxy, 'ts': now}
                    self.gex_history.append(gex_proxy)
                    self.gex_history = self.gex_history[-60:]
                    return gex_proxy, dix_proxy

            # Fallback: VIX vs its own SMA
            vix_extended = vix_tk.history(period='1mo', interval='1d')
            if len(vix_extended) >= 20:
                vix = float(vix_extended['Close'].iloc[-1])
                vix_sma = float(vix_extended['Close'].rolling(20).mean().iloc[-1])
                gex_proxy = float(100 * (vix_sma - vix) / vix_sma)
                dix_proxy = float(np.clip(0.50 - (vix - 18) / 100, 0.30, 0.60))
                self.cache = {'gex': gex_proxy, 'dix': dix_proxy, 'ts': now}
                self.gex_history.append(gex_proxy)
                self.gex_history = self.gex_history[-60:]
                return gex_proxy, dix_proxy
        except Exception:
            pass

        return 0, 0.45

    def get_size_multiplier(self):
        """
        Return position size multiplier based on GEX:
        - High GEX (>20): 1.0-1.2x (low vol, allow full size)
        - Normal GEX (0-20): 1.0x
        - Low GEX (-10 to 0): 0.8x
        - Very low GEX (<-10): 0.5-0.7x
        """
        gex, _ = self.fetch_data()

        if gex > 20:
            return min(1.2, 1.0 + gex / 100)
        elif gex > 0:
            return 1.0
        elif gex > -10:
            return 0.8
        else:
            return max(0.5, 0.7 + gex / 50)

    def should_allow_entry(self, side, regime='NEUTRAL'):
        """
        GEX Gate (Soft): Returns True mostly, only huge red flags return False.
        Major sizing reduction happens via get_size_multiplier() instead.
        """
        # v15.9: Downgraded to soft gate (scaler). Only block extreme outliers if any.
        return True

    def get_confidence_adjustment(self):
        """Adjust confidence threshold based on GEX"""
        gex, _ = self.fetch_data()

        if gex < -10:
            return 0.05  # Require higher confidence in choppy markets
        elif gex > 20:
            return -0.03  # Slightly lower threshold in stable markets
        return 0.0

# ==============================================================================
# LIQUIDITY FILTER: Microstructure & Data Quality Checks
# ==============================================================================
class LiquidityFilter:
    SETTINGS = {
        'MAX_SPREAD_PCT': 0.02,
        'MIN_DOLLAR_VOLUME': 500000,
        'MIN_REL_VOLUME': 0.5,
        'MAX_GAP_MINUTES': 30  # v15: Max allowed gap in minutes
    }

    @staticmethod
    def check(df, lookback=20):
        """
        Returns True if ticker is liquid enough to trade AND data quality is good
        """
        if len(df) < lookback:
            return False

        try:
            recent = df.tail(lookback)

            # Spread % Proxy (FIX: Use Close-Open noise instead of High-Low range)
            # High-Low overestimates spread on volatile 15m bars.
            spread_pct = (recent['Close'] - recent['Open']).abs() / recent['Close']
            avg_spread = spread_pct.mean()

            # Dollar volume
            dollar_volume = (recent['Close'] * recent['Volume']).iloc[-1]

            # Relative volume
            vol_sma = df['Volume'].rolling(lookback).mean()
            rel_vol = df['Volume'].iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 0

            # All checks must pass
            if avg_spread > LiquidityFilter.SETTINGS['MAX_SPREAD_PCT']:
                return False
            if dollar_volume < LiquidityFilter.SETTINGS['MIN_DOLLAR_VOLUME']:
                return False
            if rel_vol < LiquidityFilter.SETTINGS['MIN_REL_VOLUME']:
                return False

            # v15: GAP DETECTION (Data Hygiene)
            # Check for missing bars or large timestamp gaps
            if len(df) > 50:
                diffs = df.index.to_series().diff().dt.total_seconds() / 60
                # Dynamic Threshold: 3x the median interval (allows for 1 missing bar in 15m)
                median_diff = diffs.median()
                expected = median_diff if median_diff > 0 else 15
                max_gap = diffs.tail(50).max()

                if max_gap > (expected * 3.0):
                     # logging.debug(f"💧 Gap detected: {max_gap:.0f} mins (Limit: {expected*3})")
                     return False

            return True
        except Exception as e:
            logging.debug(f"LiquidityFilter.check() failed: {e}")
            return False

    @staticmethod
    def get_size_haircut(df):
        """
        Calculate size reduction (haircut) based on liquidity/spread.
        Returns float between 0.05 (5%) and 0.40 (40%).
        """
        try:
            if len(df) < 20: return 0.10

            recent = df.tail(20)
            # Spread Estimate (FIX: Use Close-Open noise)
            spread_pct = ((recent['Close'] - recent['Open']).abs() / recent['Close']).mean()

            # Haircut = 20 * spread (e.g., 0.5% spread -> 10% haircut)
            haircut = spread_pct * 20

            # Clip between 5% and 40%
            return max(0.05, min(0.40, haircut))
        except Exception as e:
            logging.debug(f"LiquidityFilter.get_size_haircut() failed: {e}")
            return 0.10

# ==============================================================================
# PORTFOLIO RISK CONTROLS: Sector caps and correlation limits
# ==============================================================================
class PortfolioRisk:
    """
    Portfolio-level risk controls to prevent concentrated exposure.
    - Sector caps: max 2 positions per sector
    - Theme limits: avoid too many correlated bets
    """

    # Sector mapping for CORE tickers and common high-beta stocks
    SECTOR_MAP = {
        # Space/Aerospace
        'RKLB': 'Space', 'LUNR': 'Space', 'ASTS': 'Space', 'RDW': 'Space',
        # Semiconductors
        'NVDA': 'Semis', 'MU': 'Semis', 'AVGO': 'Semis', 'AMD': 'Semis', 'INTC': 'Semis',
        # AI/Cloud
        'PLTR': 'AI', 'CRWV': 'AI', 'NET': 'Cloud', 'SNOW': 'Cloud',
        # Defense
        'KTOS': 'Defense', 'RCAT': 'Defense', 'LMT': 'Defense',
        # Biotech/Healthcare
        'RIGL': 'Biotech', 'VKTX': 'Biotech', 'TMDX': 'Biotech', 'HIMS': 'Healthcare',
        # Consumer Tech
        'APP': 'AdTech', 'AAPL': 'Consumer', 'TSLA': 'EV', 'RIVN': 'EV',
    }

    MAX_PER_SECTOR = 2
    MAX_CORRELATED = 3  # Max positions with >0.7 correlation

    def __init__(self):
        self.holdings_history = {}  # {ticker: pd.Series of daily closes}
        self.universe_history = {}  # {ticker: pd.Series}
        self.correlation_matrix = None
        self.last_update = None
        self.last_universe_update = None

    def update_holdings_data(self, positions):
        """
        Fetch and cache 60d daily history for all open positions.
        """
        current_tickers = list(positions.keys())

        # Remove closed positions
        self.holdings_history = {
            t: data for t, data in self.holdings_history.items()
            if t in current_tickers
        }

        # Add new positions
        for t in current_tickers:
            if t not in self.holdings_history:
                try:
                    df = yf.Ticker(t).history(period='3mo', interval='1d')[-60:]
                    if not df.empty:
                        self.holdings_history[t] = df['Close']
                except:
                    pass

    def update_universe_cache(self, universe_list):
        """
        Batch download history for entire universe + computing correlation matrix
        Call this periodically (e.g. every hour) to avoid per-candidate downloads
        """
        # Update every 60 mins
        if self.last_universe_update and (datetime.datetime.now() - self.last_universe_update).total_seconds() < 3600:
            return

        logging.info("↻ Updating correlation matrix for universe...")
        try:
            # Batch download everything (yfinance handles batching internally)
            # Add HOLDINGS to the universe list so we can correlate them
            full_list = list(set(universe_list + list(self.holdings_history.keys())))

            # Download in chunks of 50 to be safe
            all_closes = {}
            chunk_size = 50
            for i in range(0, len(full_list), chunk_size):
                chunk = full_list[i:i+chunk_size]
                data = yf.download(chunk, period="3mo", interval="1d", progress=False)['Close']

                # yfinance return structure varies for single vs multiple
                if isinstance(data, pd.Series):
                    all_closes[chunk[0]] = data
                else:
                    for col in data.columns:
                        all_closes[col] = data[col]

            # Create master DF aligned on date
            self.universe_history = pd.DataFrame(all_closes).tail(60)

            # Compute correlation matrix
            # Compute correlation matrix using LOG-RETURNS (not price levels)
            # FIX: Price correlation is misleading. Returns correlation is reality.
            # Use log returns: ln(p_today / p_yesterday)
            # Safety: Drop 0/NaN columns, but allow some missingness (thresh=85%) to avoid nuking universe
            prices = self.universe_history.replace(0, np.nan)
            min_obs = int(len(prices) * 0.85) # Must have 85% of bars
            prices = prices.dropna(axis=1, thresh=min_obs)

            rets = np.log(prices).diff().dropna(how="any") # Now drop rows with any NaNs (minor data loss preferred to invalid cov)
            self.correlation_matrix = rets.corr()

            self.last_universe_update = datetime.datetime.now()
            logging.info(f"✅ Correlation matrix updated ({len(self.correlation_matrix)} tickers)")

        except Exception as e:
            logging.error(f"Correlation matrix update failed: {e}")

    def check_correlation(self, candidate, holdings_list=None, threshold=0.80):  # v15.9: Relaxed from 0.70
        """
        Returns False if candidate has > 0.8 correlation with any ticker in holdings_list.
        """
        # Default to open positions if no list provided
        if holdings_list is None:
            holdings_list = list(self.holdings_history.keys())

        if not holdings_list:
            return True

        # Fast Path: Use pre-computed matrix
        if self.correlation_matrix is not None and candidate in self.correlation_matrix.index:
            try:
                # Check against current holdings only
                holdings = holdings_list
                # Filter holdings that are in the matrix
                valid_holdings = [h for h in holdings if h in self.correlation_matrix.columns]

                if not valid_holdings:
                    return True

                corrs = self.correlation_matrix.loc[candidate, valid_holdings]
                max_corr = corrs.max()

                if max_corr > threshold:
                    # Find which ticker triggered it
                    bad_ticker = corrs.idxmax()
                    logging.debug(f"🔗 Correlation block (MATRIX): {candidate} vs {bad_ticker} ({max_corr:.2f})")
                    return False
                return True
            except Exception as e:
                logging.debug(f"Matrix corr check failed: {e}")
                # Fall through to slow check

        # Slow Path: Download individually (fallback)
        try:
            cand_df = yf.Ticker(candidate).history(period='3mo', interval='1d')[-60:]
            if cand_df.empty or len(cand_df) < 30:
                return True

            cand_close = cand_df['Close']

            for hold_ticker, hold_close in self.holdings_history.items():
                if hold_ticker == candidate:
                    continue

                # Align dates
                aligned = pd.concat([cand_close, hold_close], axis=1, join='inner')
                if len(aligned) < 30:
                    continue

                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

                if corr > threshold:
                    logging.debug(f"🔗 Correlation block (SLOW): {candidate} vs {hold_ticker} ({corr:.2f})")
                    return False

            return True
        except Exception as e:
            logging.debug(f"Correlation check error: {e}")
            return True

    @classmethod
    def get_sector(cls, ticker):
        """Get sector for a ticker, returns 'Other' if unknown"""
        return cls.SECTOR_MAP.get(ticker, 'Other')

    @classmethod
    def check_sector_cap(cls, ticker, positions):
        """
        Returns True if adding this ticker would NOT exceed sector cap
        """
        new_sector = cls.get_sector(ticker)

        # Count existing positions in same sector
        sector_count = sum(
            1 for t in positions.keys()
            if cls.get_sector(t) == new_sector
        )

        return sector_count < cls.MAX_PER_SECTOR

    @classmethod
    def check_theme_limit(cls, ticker, positions, max_same_theme=3):
        """
        Check if we're too concentrated in one theme.
        Themes: Space, Semis, AI, Defense, Biotech
        """
        theme = cls.get_sector(ticker)
        high_concentration_themes = ['Space', 'Semis', 'AI', 'Biotech', 'Defense', 'EV']

        if theme not in high_concentration_themes:
            return True

        theme_count = sum(
            1 for t in positions.keys()
            if cls.get_sector(t) == theme
        )

        return theme_count < max_same_theme

    def check_portfolio_heat(self, positions, equity):
        """
        Check if total open risk exceeds global cap (e.g. 6% of equity).
        Heat = Sum(Risk$) / Equity
        """
        heat = 0.0
        for p in positions.values():
            # Use init_risk (per share) * qty for Risk$
            r = p.get('init_risk', 0)
            q = p.get('qty', 0)
            heat += (r * q)

        heat_pct = heat / max(1.0, equity)

        # If heat > MAX (e.g. 0.06), block new entries
        max_heat = SETTINGS.get('MAX_PORTFOLIO_HEAT', 0.06)
        if heat_pct >= max_heat:
            logging.debug(f"🔥 Portfolio Heat Block: {heat_pct:.1%} >= {max_heat:.1%}")
            return False
        return True

    def should_allow_entry(self, ticker, positions, equity):
        """
        Main check: returns True if entry is allowed based on portfolio constraints
        """
        if ticker in positions:
            return False

        if not self.check_portfolio_heat(positions, equity):
            return False

        if not self.check_sector_cap(ticker, positions):
            return False

        if not self.check_theme_limit(ticker, positions):
            return False

        # 3. Correlation Check (Open + Pending)
        effective_holdings = list(positions.keys())
        if not self.check_correlation(ticker, holdings_list=effective_holdings):
            return False

        return True

# ==============================================================================
# VOLATILITY TARGETING: Dynamic sizing based on realized vol
# ==============================================================================
class VolTarget:
    """
    EWMA + simple GARCH(1,1) volatility targeting.
    sigma_hat = blend * ewma_vol + (1 - blend) * garch_vol
    w_t = clip(target_vol / sigma_hat, 0.5, w_max)
    """

    def __init__(self, target_vol=0.15, ewma_lambda=0.94, w_max=2.0,
                 garch_omega=2e-6, garch_alpha=0.06, garch_beta=0.93,
                 blend=0.70):
        self.target_vol = target_vol
        self.ewma_lambda = ewma_lambda
        self.w_max = w_max
        self.garch_omega = garch_omega
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta
        self.blend = blend
        self.scalar = 1.0
        self.last_update = None
        self.sigma_hat = None
        self._spy_cache = None      # FIX: Share SPY data, avoid redundant YF calls
        self._spy_cache_ts = 0

    def _get_spy_returns(self):
        """Cached SPY fetch shared across VolTarget and Regime (30min TTL)."""
        now = time.time()
        if self._spy_cache is not None and (now - self._spy_cache_ts) < 1800:
            return self._spy_cache
        try:
            spy = yf.Ticker("SPY").history(period="3mo", interval="1d")
            if len(spy) >= 30:
                returns = spy['Close'].pct_change().dropna()
                self._spy_cache = returns
                self._spy_cache_ts = now
                return returns
        except Exception:
            pass
        return self._spy_cache  # Return stale if fetch fails

    def _ewma_variance(self, returns):
        lam = self.ewma_lambda
        var = returns.iloc[0] ** 2
        for r in returns.iloc[1:]:
            var = lam * var + (1 - lam) * (r ** 2)
        return var

    def _garch_variance(self, returns):
        omega, alpha, beta_g = self.garch_omega, self.garch_alpha, self.garch_beta
        var = returns.var()
        # FIX: Add convergence check - stop if variance explodes or collapses
        for r in returns:
            var = omega + alpha * (r ** 2) + beta_g * var
            if not np.isfinite(var) or var > 1.0:
                return returns.var()  # Fallback to sample variance
        return var

    def update_scalar(self):
        try:
            if self.last_update and (datetime.datetime.now() - self.last_update).total_seconds() < 900:
                return self.scalar

            returns = self._get_spy_returns()
            if returns is None or len(returns) < 30:
                return self.scalar if self.scalar else 1.0

            ewma_var = self._ewma_variance(returns)
            ewma_vol = np.sqrt(max(0, ewma_var) * 252)

            garch_var = self._garch_variance(returns)
            garch_vol = np.sqrt(max(0, garch_var) * 252)

            sigma_hat = self.blend * ewma_vol + (1.0 - self.blend) * garch_vol

            if not np.isfinite(sigma_hat) or sigma_hat <= 0:
                sigma_hat = returns.tail(20).std() * np.sqrt(252)

            self.sigma_hat = sigma_hat

            if sigma_hat > 0:
                raw_scalar = self.target_vol / sigma_hat
                self.scalar = max(0.5, min(self.w_max, raw_scalar))
            else:
                self.scalar = 1.0

            self.last_update = datetime.datetime.now()
            logging.info(f"⚡ VolTarget: EWMA={ewma_vol:.1%} GARCH={garch_vol:.1%} "
                         f"Blend={sigma_hat:.1%} Target={self.target_vol:.1%} Scalar={self.scalar:.2f}")
            return self.scalar

        except Exception as e:
            logging.debug(f"VolTarget update failed: {e}")
            return self.scalar if self.scalar else 1.0

    def get_scalar(self):
        return self.scalar

    def get_sigma_hat(self):
        return self.sigma_hat if self.sigma_hat else self.target_vol

# ==============================================================================
# 4. ALERTS
# ==============================================================================
def send_alert(subject, body, priority="normal"):
    if KEYS['DISCORD']:
        try:
            color = 65280 if priority=="trade" else 16711680 if priority=="high" else 3447003
            requests.post(KEYS['DISCORD'], json={"embeds": [{"title": subject, "description": body, "color": color}]}, timeout=5)
        except: pass

# ==============================================================================
# 5. ANALYTICS (FIX #6: Direction-aware historical hit-rate)
# ==============================================================================
def get_kalman_filter(series, q_base=0.01, r_base=0.1, vol_span=20):
    """
    Adaptive Kalman filter with volatility-scaled noise parameters.
    High vol: q increases (track faster), r decreases (trust observations).
    Low vol: q decreases (smooth more), r increases (trust model).
    """
    n = len(series)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([series[0]])

    abs_returns = np.abs(np.diff(series, prepend=series[0]))

    alpha_ema = 2.0 / (vol_span + 1)
    vol_ema = np.empty(n)
    vol_ema[0] = abs_returns[0] if abs_returns[0] > 0 else 1e-8
    for i in range(1, n):
        vol_ema[i] = alpha_ema * abs_returns[i] + (1 - alpha_ema) * vol_ema[i - 1]

    # FIX: Use trimmed mean (10-90th pct) instead of median for more robust baseline
    # Median can be skewed in low-activity periods; trimmed mean resists outliers
    sorted_vol = np.sort(vol_ema)
    trim_lo = max(1, int(n * 0.10))
    trim_hi = max(trim_lo + 1, int(n * 0.90))
    vol_baseline = float(np.mean(sorted_vol[trim_lo:trim_hi]))
    if vol_baseline <= 0:
        vol_baseline = 1e-8

    x = series[0]
    p = 1.0
    estimates = np.empty(n)

    for i in range(n):
        ratio = max(0.1, min(10.0, vol_ema[i] / vol_baseline))
        q = max(0.001, min(0.1, q_base * ratio))
        r = max(0.01, min(1.0, r_base / ratio))

        z = series[i]
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        estimates[i] = x

    return estimates

def get_hurst(series):
    """
    Estimate Hurst exponent via variance of lagged differences.
    H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
    """
    try:
        if len(series) < 100: return 0.5
        lags = range(2, 20)
        # FIX: Standard formula - slope of log(std of lagged diffs) vs log(lag)
        # No sqrt wrapper, no 2x multiplier - those distorted the exponent
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        if any(t <= 0 for t in tau):
            return 0.5
        slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except:
        return 0.5

def get_historical_hitrate(df, side, lookback=100, atr_mult_sl=1.5, atr_mult_tp=3.0, atr_col='ATR', max_bars=20):
    """
    Calculate historical hit rate for bracket trades using vectorized simulation.
    Returns: (wins, losses) - raw counts for Bayesian analysis
    """
    try:
        if atr_col not in df.columns:
            return 0, 0

        if len(df) < lookback + 50:
            return 0, 0

        # Ensure ATR is valid
        df = df.dropna(subset=[atr_col])
        df = df[df[atr_col] > 0]

        if len(df) < lookback + 50:
            return 0, 0  # FIX: Always return integers

        wins = 0
        losses = 0

        for i in range(lookback, len(df) - max_bars - 1, 10):
            entry = df['Close'].iloc[i]
            atr = df[atr_col].iloc[i]

            if not np.isfinite(atr) or atr <= 0:
                continue

            # Market Hours Filter - use proper Eastern time check
            ts = df.index[i]
            if not is_regular_hours(ts):
                continue

            if side == 'LONG':
                sl = entry - atr_mult_sl * atr
                tp = entry + atr_mult_tp * atr
            else:  # SHORT
                sl = entry + atr_mult_sl * atr
                tp = entry - atr_mult_tp * atr

            # Check using HIGH/LOW for realistic hit detection
            for j in range(i + 1, min(i + max_bars + 1, len(df))):
                high = df['High'].iloc[j]
                low = df['Low'].iloc[j]

                if side == 'LONG':
                    # Check SL first (hit if low touched SL)
                    if low <= sl:
                        losses += 1
                        break
                    # Then check TP (hit if high touched TP)
                    if high >= tp:
                        wins += 1
                        break
                else:  # SHORT
                    # Check SL first (hit if high touched SL)
                    if high >= sl:
                        losses += 1
                        break
                    # Then check TP (hit if low touched TP)
                    if low <= tp:
                        wins += 1
                        break

        total = wins + losses
        # v15.8: Return RAW wins/losses for Bayesian update
        return wins, losses
    except Exception as e:
        logging.debug(f"get_historical_hitrate() failed: {e}")
        return 0, 0

def estimate_rr_net(entry, sl, tp):
    """Calculate Reward/Risk net of execution costs/slippage"""
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0: return 0.0

    rr = reward / risk

    # Haircut RR for market orders (slippage reduces effective edge)
    if SETTINGS.get("USE_MARKET_ORDERS", False):
        reduction = SETTINGS.get("SLIPPAGE_HAIRCUT", 0.10) * 0.5
        rr *= (1.0 - reduction)

    return max(0.1, rr)

def estimate_cost_in_R(df, stop_distance):
    """
    Estimate transaction costs in R-units.
    Cost_R = (Spread + Slippage) / Stop_Distance
    FIX: Use 1-min bars' close-open as noise proxy instead of 15m high-low range.
    """
    try:
        # FIX: Use close-open absolute movement as spread proxy (better than high-low)
        recent = df.tail(50)  # Use more bars for stability
        # Median absolute close-open movement as noise/spread proxy
        noise_proxy = ((recent['Close'] - recent['Open']).abs() / recent['Close']).median()

        # Cap spread estimate to reasonable bounds (0.05% to 1%)
        spread_estim = max(0.0005, min(0.01, noise_proxy))

        # Slippage assumption
        slippage = spread_estim * 0.5 if SETTINGS.get("USE_MARKET_ORDERS", False) else 0

        total_cost_pct = spread_estim + slippage
        price = df['Close'].iloc[-1]

        cost_abs = price * total_cost_pct

        if stop_distance > 0:
            return cost_abs / stop_distance
        return 0.1  # Fallback high cost
    except Exception as e:
        logging.debug(f"estimate_cost_in_R failed: {e}")
        return 0.1

def get_bayesian_p_safe(wins, losses, confidence_level=0.10):
    """
    Returns the lower bound of the win rate posterior (Safe P).
    Posterior ~ Beta(wins+1, losses+1)
    FIX: Minimum sample protection - return neutral 0.50 when data is too sparse
    to avoid penalizing tickers with no trading history.
    """
    try:
        total = wins + losses
        if total < 10:
            # Not enough data - return neutral (don't penalize or reward)
            return 0.50
        alpha = wins + 1
        beta_param = losses + 1
        p_safe = beta.ppf(confidence_level, alpha, beta_param)
        return float(np.clip(p_safe, 0.05, 0.95))
    except Exception as e:
        logging.debug(f"get_bayesian_p_safe failed: {e}")
        return 0.5

def kelly_3_outcome(pW, pL, pH, b, d=0.15, f_max=0.04):
    """
    3-Outcome Log-Utility Kelly (User Option B).
    Maximizes E[log(growth)] accounting for Wins, Losses, and Holds (Decay).
    pW: Prob win, pL: Prob loss, pH: Prob hold.
    b: Reward multiple (e.g. 2.0).
    d: Hold penalty (e.g. 0.15R).
    f_max: Cap on fraction (e.g. 0.04).
    """
    import numpy as np

    # Grid search for max log-growth
    # Safety: f must satisfy 1-f > 0, 1-f*d > 0
    def obj(f):
        if f <= 1e-6: return -1e9
        if (1 - f) <= 1e-9 or (1 - f*d) <= 1e-9 or (1 + f*b) <= 1e-9:
            return -1e9
        return (pW * np.log(1 + f * b) +
                pL * np.log(1 - f) +
                pH * np.log(1 - f * d))

    # Search in [0, f_max]
    grid = np.linspace(0, f_max, 200)
    vals = np.array([obj(f) for f in grid])
    best = grid[np.argmax(vals)]

    # If objective is negative or nan, return 0
    if np.max(vals) == -1e9 or np.isnan(np.max(vals)):
        return 0.0

    return float(best)

# Backward compatibility wrapper if needed
def aggressive_kelly_risk_pct(*args, **kwargs): return 0.01

def ev_to_size_mult(ev_r):
    """
    Map EV (in R-units) to a continuous sizing scalar [0.35 .. 1.25].
    Low-edge trades get smaller, not filtered out. Preserves frequency, improves PF.
    """
    return float(np.clip(0.35 + 1.5 * ev_r, 0.35, 1.25))


# NOTE: CrossSectionalRanker and OvernightGapModel are now imported from hedge_fund package above.

# FIX #8: Get DAILY ATR for risk sizing (more stable than minute ATR)
def get_daily_atr(ticker, period=14):
    """Use daily ATR for risk sizing - more stable than minute bars"""
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

# FIX: Cache daily ATR to prevent yfinance rate limiting (call once per 6 hours)
class DailyATRCache:
    def __init__(self, ttl_sec=6*3600):
        self.ttl = ttl_sec
        self.cache = {}  # ticker -> (timestamp, atr)

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

    def clear_expired(self):
        """Remove expired entries to prevent memory growth"""
        now = time.time()
        expired = [k for k, (ts, _) in self.cache.items() if now - ts >= self.ttl]
        for k in expired:
            del self.cache[k]

# Global ATR cache instance
ATR_CACHE = DailyATRCache(ttl_sec=6*3600)
def attach_daily_atr_to_15m(df_15m: pd.DataFrame, ticker: str, poly=None, period=14):
    """
    Attach daily ATR to 15-min bars using vectorized join (fast).
    Uses Polygon daily bars if poly provided, else yfinance.
    """
    try:
        d = pd.DataFrame()
        if poly:
            # Use Polygon for 5 years of daily data (1 call)
            d = poly.fetch_daily_data(ticker, days=1825)

        if d.empty:
            # Fallback to yfinance if Polygon fails or not provided
            d = yf.Ticker(ticker).history(period="5y", interval="1d")
            if not d.empty and d.index.tz is not None:
                d.index = d.index.tz_convert("America/New_York")

        if d.empty or len(d) < period + 2:
            return df_15m

        # True Range
        high, low, close = d["High"], d["Low"], d["Close"]
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        # FIX: Shift by 1 to prevent Daily ATR leakage into intraday bars
        d_atr = tr.rolling(period).mean().shift(1).rename("ATR_D")

        # Vectorized join on normalized date (strip tz for both sides of join)
        d_atr.index = pd.to_datetime(d_atr.index.tz_localize(None) if d_atr.index.tz is None
                                     else d_atr.index.tz_convert(None)).normalize()

        out = df_15m.copy()
        _idx = pd.to_datetime(out.index)
        if _idx.tz is not None:
            _idx = _idx.tz_convert(None)
        out["_d"] = _idx.normalize()
        out = out.join(d_atr, on="_d")
        out.drop(columns=["_d"], inplace=True)
        out["ATR_D"] = out["ATR_D"].ffill()

        # Ensure we have a column named ATR_D
        if "ATR_D" not in out.columns:
             out["ATR_D"] = np.nan

        return out
    except Exception as e:
        logging.debug(f"attach_daily_atr_to_15m failed for {ticker}: {e}")
        return df_15m

def get_daily_atr_polygon(poly, ticker: str, period=14, cache_ttl_sec=6*3600):
    """
    Get daily ATR from Polygon using optimized daily bars (no 1-min resample).
    """
    key = ("atrD", ticker, period)
    now = time.time()
    if key in poly._mem_cache and (now - poly._mem_cache[key][0] < cache_ttl_sec):
        return poly._mem_cache[key][1]

    try:
        # Use optimized daily fetch
        d = poly.fetch_daily_data(ticker, days=60)
        if d.empty or len(d) < period + 2:
            return None

        high, low, close = d["High"], d["Low"], d["Close"]
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        # FIX: Shift by 1 to prevent leakage of today's partial ATR
        atr = float(tr.rolling(period).mean().shift(1).iloc[-1])

        res = atr if np.isfinite(atr) and atr > 0 else None
        poly._mem_cache[key] = (now, res)
        return res
    except Exception as e:
        logging.debug(f"get_daily_atr_polygon({ticker}) failed: {e}")
        return None

def compute_bracket_labels(df, sl_mult=1.5, tp_mult=3.0, max_bars=20, atr_col='ATR', **_kwargs):
    """
    EV-style continuous labels with direction encoded in sign.
      +x  => LONG with expected R ~ x
      -x  => SHORT with expected R ~ |x|
       0  => HOLD / no trade (both directions negative EV)

    Timeouts use mark-to-market at horizon (continuous gradient, not flat -0.15).
    atr_col: use 'ATR_D' for daily ATR to match live bracket geometry.
    """
    n = len(df)
    labels = np.zeros(n, dtype=float)

    atr = df[atr_col].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    rr = tp_mult / sl_mult  # e.g. 3.0/1.5 = 2.0R reward

    for i in range(n - max_bars - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        entry = close[i]
        risk = sl_mult * a
        if risk <= 0:
            continue

        # --- LONG bracket ---
        long_sl = entry - risk
        long_tp = entry + tp_mult * a
        long_out = _simulate_exit(high[i+1:i+max_bars+1], low[i+1:i+max_bars+1],
                                  long_sl, long_tp, 'LONG')
        if long_out == 'win':
            long_r = rr
        elif long_out == 'loss':
            long_r = -1.0
        else:
            # Mark-to-market at horizon with small decay for dead money
            mtm = (close[i + max_bars] - entry) / risk
            long_r = float(np.clip(mtm - 0.05, -1.0, rr))

        # --- SHORT bracket ---
        short_sl = entry + risk
        short_tp = entry - tp_mult * a
        short_out = _simulate_exit(high[i+1:i+max_bars+1], low[i+1:i+max_bars+1],
                                   short_sl, short_tp, 'SHORT')
        if short_out == 'win':
            short_r = rr
        elif short_out == 'loss':
            short_r = -1.0
        else:
            mtm = (entry - close[i + max_bars]) / risk
            short_r = float(np.clip(mtm - 0.05, -1.0, rr))

        # --- Choose best POSITIVE EV only; otherwise HOLD = 0 ---
        best = max(long_r, short_r)
        if best <= 0.0:
            labels[i] = 0.0                    # Both directions negative EV => don't trade
        elif long_r >= short_r:
            labels[i] = float(best)            # + => go long
        else:
            labels[i] = float(-best)           # - => go short

    return labels

def _simulate_exit(highs, lows, sl, tp, side):
    """
    Simulate bracket exit using High/Low of future bars
    Returns 'win' if TP hit first, 'loss' if SL hit first, 'timeout' if neither
    """
    for i in range(len(highs)):
        h = highs[i]
        l = lows[i]

        if side == 'LONG':
            # SL hit if low goes below SL
            if l <= sl:
                return 'loss'
            # TP hit if high goes above TP
            if h >= tp:
                return 'win'
        else:  # SHORT
            # SL hit if high goes above SL
            if h >= sl:
                return 'loss'
            # TP hit if low goes below TP
            if l <= tp:
                return 'win'

    return 'timeout'

# ==============================================================================
# LIQUIDITY FILTER: Skip thin/illiquid trades
# ==============================================================================


# ==============================================================================
# 6. GUARDS
# ==============================================================================
class EarningsGuard:
    MAX_CACHE_SIZE = 500  # FIX: Prevent unbounded cache growth

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
                    logging.warning(f"⚠️ EARNINGS SKIP: {ticker}")
                    return False

            # FIX: Limit cache size to prevent memory leak
            if len(self.earnings_cache) >= self.MAX_CACHE_SIZE:
                # Remove oldest entries (approx half)
                keys_to_remove = list(self.earnings_cache.keys())[:self.MAX_CACHE_SIZE // 2]
                for k in keys_to_remove:
                    del self.earnings_cache[k]

            self.earnings_cache[ticker] = now + datetime.timedelta(days=1)
        except Exception as e:
            logging.debug(f"EarningsGuard.check_safe({ticker}) failed: {e}")
        return True

    def check_news(self, ticker):
        """
        v15.10: Check FMP News for catastrophic keywords.
        Returns: (is_safe, penalty_level)
           - penalty_level 0: No penalty
           - penalty_level 1: Warning (Tier Drop)
           - penalty_level 2: Hard Block
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=5&apikey={KEYS['FMP']}"
            res = requests.get(url, timeout=3).json()

            # Keywords
            block_list = ["fraud", "subpoena", "investigation", "sec ", "bankruptcy", "chapter 11"]
            warn_list = ["lawsuit", "class action", "miss", "plunge", "tumble", "crash"]

            for item in res:
                # Only check recent news (last 24h)
                pub_date = pd.to_datetime(item.get('publishedDate')).date()
                if (datetime.date.today() - pub_date).days > 1:
                    continue

                title = item.get('title', '').lower()
                text = item.get('text', '').lower()
                content = title + " " + text

                if any(x in content for x in block_list):
                    logging.warning(f"🚨 NEWS BLOCK {ticker}: {title[:50]}...")
                    return False, 2

                if any(x in content for x in warn_list):
                    return True, 1

            return True, 0
        except Exception as e:
            return True, 0

class HedgeManager:
    def __init__(self, alpaca):
        self.alpaca = alpaca
        self.is_hedged = False

    def check_hedge(self, regime):
        try:
            if regime == 'BEAR' and not self.is_hedged:
                self.deploy_hedge()
            elif regime != 'BEAR' and self.is_hedged:
                self.remove_hedge()
        except Exception as e:
            ERROR_TRACKER.record_failure("Hedge", str(e))

    def deploy_hedge(self):
        try:
            equity = float(self.alpaca.api.get_account().equity)
            tk = yf.Ticker(SETTINGS['HEDGE_TICKER'])
            hist = tk.history(period='1d')
            if hist.empty: return
            price = hist['Close'].iloc[-1]
            qty = int((equity * 0.15) / price)

            if qty > 0:
                self.alpaca.api.submit_order(
                    symbol=SETTINGS['HEDGE_TICKER'], qty=qty,
                    side='buy', type='market', time_in_force='day'
                )
                self.is_hedged = True
                send_alert("🛡️ HEDGE DEPLOYED", f"Bought {qty} {SETTINGS['HEDGE_TICKER']}", "high")
        except Exception as e:
            ERROR_TRACKER.record_failure("Hedge_Deploy", str(e))

    def remove_hedge(self):
        try:
            self.alpaca.api.close_position(SETTINGS['HEDGE_TICKER'])
            self.is_hedged = False
        except:
            self.is_hedged = False

# ==============================================================================
# 6b. COINTEGRATION PAIRS SCANNER
# ==============================================================================
try:
    from statsmodels.tsa.stattools import coint as _coint_test
    HAS_COINT = True
except ImportError:
    HAS_COINT = False


class PairsScanner:
    """
    Engle-Granger cointegration scanner.
    Finds mean-reverting pairs in the universe and generates spread-trade signals.
    Market-neutral: profits in CHOP regimes where directional strategies struggle.
    """
    def __init__(self, poly, lookback_days=60, rescan_hours=6):
        self.poly = poly
        self.lookback_days = lookback_days
        self._rescan_interval = rescan_hours * 3600
        self._pairs = []          # List of (tickerA, tickerB, hedge_ratio, half_life)
        self._last_scan = 0
        self._spread_cache = {}   # (A, B) -> (ts, z_score)
        self._lock = threading.Lock()

    def scan_pairs(self, universe, max_pairs=10):
        """
        Run Engle-Granger cointegration test on all unique pairs.
        Keeps pairs with p-value < 0.05 and half-life 2-30 days.
        """
        if not HAS_COINT:
            return []
        now = time.time()
        if now - self._last_scan < self._rescan_interval and self._pairs:
            return self._pairs

        logging.info(f"🔗 Pairs scan: testing {len(universe)} tickers...")
        # Fetch daily closes for covariance
        prices = self.poly.fetch_batch_bars(universe, days=self.lookback_days)
        if prices.empty or len(prices.columns) < 4:
            return self._pairs

        # Drop tickers with too many NaNs
        prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
        tickers = list(prices.columns)

        found = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                a, b = tickers[i], tickers[j]
                try:
                    s1 = prices[a].dropna()
                    s2 = prices[b].dropna()
                    common = s1.index.intersection(s2.index)
                    if len(common) < 40:
                        continue
                    s1, s2 = s1.loc[common].values, s2.loc[common].values

                    # Engle-Granger test
                    _, p_value, _ = _coint_test(s1, s2)
                    if p_value > 0.05:
                        continue

                    # Hedge ratio via OLS
                    hedge_ratio = float(np.polyfit(s2, s1, 1)[0])

                    # Spread and half-life
                    spread = s1 - hedge_ratio * s2
                    spread_diff = np.diff(spread)
                    if len(spread_diff) < 10 or np.std(spread) < 1e-8:
                        continue
                    # Ornstein-Uhlenbeck half-life: HL = -ln(2) / lambda
                    lag_spread = spread[:-1]
                    beta_ou = np.polyfit(lag_spread, spread_diff, 1)[0]
                    if beta_ou >= 0:
                        continue  # Not mean-reverting
                    half_life = -np.log(2) / beta_ou

                    if 2 <= half_life <= 30:
                        found.append((a, b, round(hedge_ratio, 4), round(half_life, 1), round(p_value, 4)))

                except Exception:
                    continue

        # Rank by half-life (shorter = faster reversion = better)
        found.sort(key=lambda x: x[3])
        with self._lock:
            self._pairs = found[:max_pairs]
            self._last_scan = now

        logging.info(f"🔗 Found {len(self._pairs)} cointegrated pairs (top {max_pairs})")
        for p in self._pairs:
            logging.info(f"   {p[0]}/{p[1]} HR={p[2]:.2f} HL={p[3]:.0f}d p={p[4]:.3f}")
        return self._pairs

    def get_pair_signals(self, prices_dict, z_entry=2.0, z_exit=0.5):
        """
        Generate spread-trade signals for active pairs.
        Returns list of {legA, legB, sideA, sideB, z_score, half_life} dicts.
        """
        signals = []
        with self._lock:
            pairs = list(self._pairs)
        for (a, b, hr, hl, _pv) in pairs:
            try:
                if a not in prices_dict or b not in prices_dict:
                    continue
                pa, pb = prices_dict[a], prices_dict[b]
                spread = pa - hr * pb

                # Z-score using 20-day lookback (from cache or fresh)
                cache_key = (a, b)
                now = time.time()
                if cache_key in self._spread_cache:
                    ts, hist = self._spread_cache[cache_key]
                    if now - ts < 900:  # 15-min cache
                        hist.append(spread)
                        if len(hist) > 100:
                            hist = hist[-100:]
                    else:
                        hist = [spread]
                else:
                    hist = [spread]
                self._spread_cache[cache_key] = (now, hist)

                if len(hist) < 10:
                    continue
                mu = np.mean(hist)
                sigma = np.std(hist)
                if sigma < 1e-8:
                    continue
                z = (spread - mu) / sigma

                if abs(z) >= z_entry:
                    # Spread is extended: trade mean-reversion
                    # z > +2: spread too high → short A, long B
                    # z < -2: spread too low → long A, short B
                    signals.append({
                        'legA': a, 'legB': b,
                        'sideA': 'SHORT' if z > 0 else 'LONG',
                        'sideB': 'LONG' if z > 0 else 'SHORT',
                        'hedge_ratio': hr,
                        'z_score': round(float(z), 2),
                        'half_life': hl,
                        'type': 'Pairs'
                    })
            except Exception:
                continue
        return signals


# ==============================================================================
# 7. DATA & EXECUTION
# ==============================================================================

class RateLimiter:
    """Token-bucket rate limiter. Allows short bursts while capping average rate."""
    def __init__(self, rate_per_sec=6.0, burst=10):
        self._rate = rate_per_sec
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, timeout=30.0):
        """Block until a token is available. Returns True on success."""
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


class ScanBarCache:
    """
    In-memory cache for live-scan 15-min bar data.
    Tracks which 15-minute slot each ticker was last fetched in.
    If the slot hasn't changed, returns cached bars (no API call).
    Reduces API calls by ~93% (14 of every 15 scan cycles skip fetch).
    """
    def __init__(self):
        self._data = {}        # ticker -> DataFrame
        self._last_slot = {}   # ticker -> (hour, quarter)
        self._lock = threading.Lock()

    @staticmethod
    def _current_slot():
        """Current 15-min slot as (hour, quarter). E.g. 9:37 -> (9, 2)."""
        try:
            now = datetime.datetime.now(ET)
        except Exception:
            now = datetime.datetime.now()
        return (now.hour, now.minute // 15)

    def get_if_same_slot(self, ticker):
        """Return cached df if we're in the same 15-min slot as last fetch."""
        slot = self._current_slot()
        with self._lock:
            if ticker in self._last_slot and self._last_slot[ticker] == slot:
                return self._data.get(ticker)
        return None

    def put(self, ticker, df):
        """Cache bar data and record current slot."""
        with self._lock:
            self._data[ticker] = df
            self._last_slot[ticker] = self._current_slot()

    def invalidate(self, ticker=None):
        """Clear cache for a ticker (or all)."""
        with self._lock:
            if ticker:
                self._data.pop(ticker, None)
                self._last_slot.pop(ticker, None)
            else:
                self._data.clear()
                self._last_slot.clear()


class PolygonBarStream:
    """
    WebSocket stream for Polygon aggregate minute bars.
    Detects 15-min bar closes and exposes ready tickers to the main loop.
    Hybrid: REST stays for snapshots/history, WS only triggers scan timing.
    """
    def __init__(self, api_key):
        self._api_key = api_key
        self._ws = None
        self._thread = None
        self._running = False
        self._subscribed = set()
        self._lock = threading.Lock()
        self._ready = set()          # Tickers with a fresh 15-min bar close
        self._event = threading.Event()  # Signals main loop that bars are ready
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    def start(self):
        """Start the WS listener in a background daemon thread."""
        if not HAS_WS:
            logging.warning("websocket-client not installed. WS bar stream disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PolyWS")
        self._thread.start()
        logging.info("🔌 Polygon WebSocket bar stream started")

    def _run_loop(self):
        """Reconnecting event loop."""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logging.debug(f"WS connect error: {e}")
            self._connected = False
            if self._running:
                time.sleep(5)

    def _connect(self):
        url = "wss://socket.polygon.io/stocks"

        def on_open(ws):
            ws.send(json.dumps({"action": "auth", "params": self._api_key}))

        def on_message(ws, raw):
            try:
                msgs = json.loads(raw)
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    ev = msg.get('ev')
                    if ev == 'status':
                        st = msg.get('status', '')
                        if st == 'auth_success':
                            self._connected = True
                            logging.info("🔌 Polygon WS authenticated")
                            self._send_subscriptions(ws)
                        elif st == 'auth_failed':
                            logging.error("🔌 Polygon WS auth FAILED")
                            self._running = False
                    elif ev == 'AM':
                        # Aggregate minute bar closed
                        sym = msg.get('sym', '')
                        end_ms = msg.get('e', 0)
                        if sym and end_ms:
                            try:
                                bar_end = datetime.datetime.fromtimestamp(end_ms / 1000, tz=ET)
                                # 15-min boundary: bar ending at :00, :15, :30, :45
                                if bar_end.minute % 15 == 0:
                                    with self._lock:
                                        self._ready.add(sym)
                                    self._event.set()  # Wake up main loop
                            except Exception:
                                pass
            except Exception:
                pass

        def on_error(ws, error):
            logging.debug(f"WS error: {error}")
            self._connected = False

        def on_close(ws, code, msg):
            self._connected = False
            logging.info(f"🔌 WS closed ({code})")

        self._ws = ws_lib.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _send_subscriptions(self, ws):
        with self._lock:
            tickers = list(self._subscribed)
        if tickers:
            params = ",".join(f"AM.{t}" for t in tickers)
            ws.send(json.dumps({"action": "subscribe", "params": params}))
            logging.info(f"🔌 Subscribed to AM.* for {len(tickers)} tickers")

    def subscribe(self, tickers):
        """Update subscription list. Safe to call anytime."""
        with self._lock:
            new = set(tickers) - self._subscribed
            self._subscribed = set(tickers)
        if new and self._ws and self._connected:
            try:
                params = ",".join(f"AM.{t}" for t in new)
                self._ws.send(json.dumps({"action": "subscribe", "params": params}))
            except Exception:
                pass

    def get_ready_tickers(self):
        """Return tickers with fresh 15-min bars and clear the set."""
        with self._lock:
            ready = set(self._ready)
            self._ready.clear()
        self._event.clear()
        return ready

    def wait_for_bars(self, timeout=60.0):
        """Block until bar-close event or timeout. Returns True if bars ready."""
        return self._event.wait(timeout=timeout)

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass


class Polygon_Helper:
    def __init__(self):
        self.sess = requests.Session()
        self.base = "https://api.polygon.io"
        self.last_429 = 0
        self._lock = threading.Lock()  # FIX: Thread-safe for training
        self._mem_cache = {}  # In-memory cache for snapshots and daily ATR

        # Rate Limit - token bucket (paid plan: ~100 req/min, burst OK)
        self._rate_limiter = RateLimiter(rate_per_sec=6.0, burst=10)

    def _throttle(self):
        self._rate_limiter.acquire()

    def fetch_snapshot_prices(self, tickers, ttl=5):
        """
        Polygon Snapshot: 1 call returns latest price & day stats for many tickers.
        Returns dict: { "AAPL": {"price": 123.4, "dayVol": 123, ...}, ... }
        """
        if not tickers:
            return {}

        # Cache by sorted tickers
        key = ("snap", tuple(sorted(tickers)))
        now = time.time()
        if key in self._mem_cache and (now - self._mem_cache[key][0] < ttl):
            return self._mem_cache[key][1]

        out = {}
        chunk_size = 200  # Polygon limit
        chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

        def fetch_chunk(chunk):
            try:
                url = f"{self.base}/v2/snapshot/locale/us/markets/stocks/tickers"
                params = {"tickers": ",".join(chunk), "apiKey": KEYS["POLY"]}

                self._throttle()
                r = self.sess.get(url, params=params, timeout=10)

                if r.status_code == 429:
                    with self._lock:
                        self.last_429 = time.time()
                    return []
                if r.status_code != 200:
                    return []

                return r.json().get("tickers", [])
            except Exception as e:
                logging.debug(f"Snapshot fetch failed: {e}")
                return []

        # Optimized: Use high concurrency for I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
            futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                for item in data:
                    sym = item.get("ticker")
                    if not sym: continue

                    last_trade = item.get("lastTrade") or {}
                    day = item.get("day") or {}

                    # Price: prefer lastTrade, fallback to day close
                    price = last_trade.get("p") or day.get("c")
                    if price is None: continue

                    out[sym] = {
                        "price": float(price),
                        "dayVol": float(day.get("v") or 0),
                        "dayVWAP": float(day.get("vw") or 0),
                        "dayOpen": float(day.get("o") or 0),
                        "dayHigh": float(day.get("h") or 0),
                        "dayLow": float(day.get("l") or 0),
                        "dayClose": float(day.get("c") or 0),
                    }

        self._mem_cache[key] = (now, out)
        return out

    def fetch_data(self, t, days=5, mult=1):
        """
        Fetch Polygon aggregate bars with pagination.
        Optimization: Single request + next_url pagination instead of chunking.
        For 15-min bars, 4 years is usually < 50k bars = 1 request.
        """
        with self._lock:
            if time.time() - self.last_429 < 60:
                time.sleep(60 - (time.time() - self.last_429))

        cache = os.path.join(DRIVE_ROOT, f"market_cache/{t}_{mult}min_{days}d.parquet")
        if os.path.exists(cache):
            age = time.time() - os.path.getmtime(cache)
            # FIX: Relax cache strictness. If huge history, allow 30 days age.
            max_age = 60 if days <= 5 else 2592000 # 30 days
            if age < max_age:
                try:
                    df = pd.read_parquet(cache)
                    if len(df) > 0:
                        logging.info(f"📂 Cached {t}: {len(df)} bars (Age: {age/3600:.1f}h)")
                        return df
                except Exception as e:
                    logging.debug(f"Cache read failed for {t}: {e}")

        end = datetime.datetime.now(datetime.timezone.utc)
        start = end - datetime.timedelta(days=days)

        # Single request with pagination (not chunking)
        url = (
            f"{self.base}/v2/aggs/ticker/{t}/range/{mult}/minute/"
            f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
            f"?adjusted=true&limit=50000&sort=asc&apiKey={KEYS['POLY']}"
        )

        all_rows = []
        # Infinite Patience Mode: If 429, wait and retry. Don't give up easily.
        max_retries = 12

        while url:
            retries = 0
            while retries < max_retries:
                try:
                    self._throttle()
                    r = self.sess.get(url, timeout=15)
                    if r.status_code == 200:
                        js = r.json()
                        all_rows.extend(js.get("results", []) or [])
                        # Follow pagination
                        url = js.get("next_url")
                        if url and "apiKey=" not in url:
                            url = url + ("&" if "?" in url else "?") + f"apiKey={KEYS['POLY']}"
                        break
                    elif r.status_code == 429:
                        retries += 1
                        with self._lock:
                            self.last_429 = time.time()
                        # Exponential backoff for patience (60s, 65s, ...)
                        wait_time = 60 + (retries * 5)
                        logging.warning(f"⚠️ Polygon 429 {t} (Attempt {retries}/{max_retries}) -> Sleeping {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.warning(f"Polygon {t} status {r.status_code}: {r.text[:200] if r.text else 'no body'}")
                        url = None
                        break
                except Exception as e:
                    retries += 1
                    logging.debug(f"Polygon error {t}: {e}")
                    time.sleep(3)

            if retries >= max_retries:
                url = None

        if not all_rows:
            # FAILOVER 1: Try FMP (Financial Modeling Prep)
            # FMP paid plans support from/to params for full historical 15min data
            try:
                logging.warning(f"⚠️ Polygon {t}: No data. Failing over to FMP...")
                fmp_from = start.strftime('%Y-%m-%d')
                fmp_to = end.strftime('%Y-%m-%d')
                fmp_url = (
                    f"https://financialmodelingprep.com/api/v3/historical-chart/15min/{t}"
                    f"?from={fmp_from}&to={fmp_to}&apikey={KEYS['FMP']}"
                )
                r_fmp = self.sess.get(fmp_url, timeout=15)
                if r_fmp.status_code == 200:
                    data_fmp = r_fmp.json()
                    if data_fmp and isinstance(data_fmp, list):
                        df_fmp = pd.DataFrame(data_fmp)
                        # FMP Cols: date, open, low, high, close, volume
                        if 'date' in df_fmp.columns:
                            df_fmp = df_fmp.rename(columns={
                                'date': 'Datetime', 'open': 'Open', 'high': 'High',
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                            })
                            df_fmp['Datetime'] = pd.to_datetime(df_fmp['Datetime'])
                            df_fmp = df_fmp.set_index('Datetime').sort_index()
                            if len(df_fmp) > 100:
                                logging.info(f"✅ FMP {t}: Fetched {len(df_fmp)} bars ({fmp_from} to {fmp_to})")
                                try:
                                    df_fmp.to_parquet(cache)
                                except Exception:
                                    pass
                                return df_fmp
                    elif r_fmp.status_code == 403:
                        logging.warning(f"⚠️ FMP {t}: 403 Forbidden - check FMP plan/key")
                    else:
                        logging.debug(f"FMP {t}: status {r_fmp.status_code}")
            except Exception as ef:
                logging.warning(f"FMP fallback failed for {t}: {ef}")

            # FAILOVER 2: Try YFinance if Polygon fails (e.g. Expired Key / Rate Limit)
            try:
                logging.warning(f"⚠️ Polygon {t}: No data. Failing over to Yahoo Finance (60d max)...")
                # YFinance 15m limit is ~60 days
                df_yf = yf.download(t, period="59d", interval="15m", progress=False, auto_adjust=True)
                if not df_yf.empty:
                    # Flat columns if MultiIndex
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.get_level_values(0)

                    df_yf = df_yf.reset_index()
                    # Rename to internal standard
                    # YF: Datetime/Date, Open, High, Low, Close, Volume
                    # Clean column names
                    df_yf.columns = [c.lower() for c in df_yf.columns]
                    rename_map = {
                        'date': 'Datetime', 'datetime': 'Datetime',
                        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                    }
                    df_yf = df_yf.rename(columns=rename_map)

                    # Ensure Datetime
                    if 'Datetime' in df_yf.columns:
                        df_yf = df_yf.set_index('Datetime')

                    # Convert to Eastern for consistency with Polygon data
                    if df_yf.index.tz is not None:
                        df_yf.index = df_yf.index.tz_convert("America/New_York")

                    df_yf = df_yf.sort_index()
                    df_yf = df_yf[~df_yf.index.duplicated()]
                    logging.info(f"✅ YFinance {t}: Fetched {len(df_yf)} bars")
                    return df_yf
            except Exception as ey:
                logging.debug(f"YFinance fallback failed: {ey}")

            ERROR_TRACKER.record_failure(f"Data_{t}", "No data (Poly+YF)")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).rename(columns={
            't': 'Datetime', 'c': 'Close', 'o': 'Open',
            'h': 'High', 'l': 'Low', 'v': 'Volume'
        })
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True).dt.tz_convert("America/New_York")
        df = df.set_index('Datetime').sort_index()
        df = df[~df.index.duplicated()]

        try:
            df.to_parquet(cache)
        except Exception as e:
            logging.debug(f"Cache write failed for {t}: {e}")

        return df

    def fetch_daily_data(self, t, days=365):
        """
        Fetch daily bars directly from Polygon. Optimized for daily ATR/Metrics.
        """
        key = ("daily", t, days)
        now = time.time()
        with self._lock:
            if key in self._mem_cache and (now - self._mem_cache[key][0] < 3600): # 1h TTL
                 return self._mem_cache[key][1]

        end = datetime.datetime.now(datetime.timezone.utc).date()
        start = end - datetime.timedelta(days=days)

        url = f"{self.base}/v2/aggs/ticker/{t}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={KEYS['POLY']}"
        try:
            self._throttle()
            r = self.sess.get(url, timeout=15)
            if r.status_code != 200:
                return pd.DataFrame()

            rows = r.json().get("results", [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).rename(columns={'t': 'Datetime', 'c': 'Close', 'o': 'Open', 'h': 'High', 'l': 'Low', 'v': 'Volume'})
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True).dt.tz_convert("America/New_York")
            df = df.set_index('Datetime').sort_index()

            with self._lock:
                self._mem_cache[key] = (now, df)
            return df
        except Exception as e:
            logging.debug(f"fetch_daily_data({t}) failed: {e}")
            return pd.DataFrame()

    def fetch_batch_bars(self, tickers, days=30):
        """
        Fetch daily close prices for multiple tickers and return as a single DataFrame.
        Columns = ticker symbols, Index = dates, Values = Close prices.
        Used by the portfolio optimizer for covariance estimation.
        """
        frames = {}
        for t in tickers:
            try:
                df = self.fetch_daily_data(t, days=days)
                if not df.empty and 'Close' in df.columns:
                    frames[t] = df['Close']
            except Exception as e:
                logging.debug(f"fetch_batch_bars({t}) failed: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result = result.sort_index().ffill().dropna(how='all')
        return result

class FMP_Helper:
    def __init__(self):
        self.news_cache = {}  # (ticker, hours) -> (ts, score)

    def get_dynamic_universe(self, exclude=None, max_tickers=30):
        """
        Supertrend Gold Screener — finds high-momentum trending stocks.

        Hard filters (FMP screener):
          Market cap > $2B, Avg volume > 1M, Beta > 1.5, Price > $5
          Sectors: Technology, Industrials, Healthcare

        Technical filters (computed from 250 daily bars):
          Price above SMA(50) and SMA(200)
          SMA(20) above SMA(50) — bullish stack
          RSI(14) between 50 and 70 — bullish, not overbought
          ADX(14) > 25 — strong trend, no chop
          YTD performance > 20%

        Returns top max_tickers ranked by YTD performance, no duplicates.
        Results cached for 30 min to avoid redundant API calls on retrains.
        """
        exclude = exclude or set()

        # --- Cache: reuse within 30 min ---
        now = time.time()
        if hasattr(self, '_stg_cache') and (now - self._stg_cache[0] < 1800):
            cached = [s for s in self._stg_cache[1] if s not in exclude]
            logging.info(f"Supertrend Gold: returning {len(cached[:max_tickers])} from cache")
            return cached[:max_tickers]

        try:
            # --- Step 1: FMP hard filters per target sector ---
            TARGET_SECTORS = ['Technology', 'Industrials', 'Healthcare']
            raw = []
            for sector in TARGET_SECTORS:
                try:
                    url = (
                        f"https://financialmodelingprep.com/api/v3/stock-screener?"
                        f"marketCapMoreThan=2000000000&volumeMoreThan=1000000"
                        f"&betaMoreThan=1.5&priceMoreThan=5"
                        f"&sector={sector}&limit=100"
                        f"&apikey={KEYS['FMP']}"
                    )
                    resp = requests.get(url, timeout=10).json()
                    if isinstance(resp, list):
                        raw.extend(resp)
                except Exception as e:
                    logging.warning(f"Supertrend Gold screener failed for {sector}: {e}")

            # Deduplicate, NASDAQ/NYSE only, no ETFs
            seen = set()
            candidates = []
            for x in raw:
                sym = x.get('symbol', '')
                if (sym and sym not in seen
                        and x.get('exchangeShortName') in ('NASDAQ', 'NYSE')
                        and not x.get('isEtf', False)):
                    seen.add(sym)
                    candidates.append(x)

            logging.info(f"Supertrend Gold: {len(candidates)} passed hard filters")

            # Sort by volume, take top 60 for technical screening
            candidates.sort(key=lambda x: x.get('volume', 0), reverse=True)
            to_check = [c['symbol'] for c in candidates[:60]]

            if not to_check:
                self._stg_cache = (now, [])
                return []

            # --- Step 2: Technical filters on 250 daily bars ---
            qualified = []
            for sym in to_check:
                try:
                    url = (
                        f"https://financialmodelingprep.com/api/v3/"
                        f"historical-price-full/{sym}?timeseries=250"
                        f"&apikey={KEYS['FMP']}"
                    )
                    data = requests.get(url, timeout=8).json()
                    bars = data.get('historical', [])
                    if len(bars) < 200:
                        continue

                    # FMP returns newest-first; reverse for chronological
                    bars = list(reversed(bars))
                    closes = pd.Series([b['close'] for b in bars], dtype=float)
                    highs  = pd.Series([b['high'] for b in bars], dtype=float)
                    lows   = pd.Series([b['low'] for b in bars], dtype=float)
                    price  = closes.iloc[-1]

                    # -- Trend: Bullish SMA stack --
                    sma20  = closes.rolling(20).mean().iloc[-1]
                    sma50  = closes.rolling(50).mean().iloc[-1]
                    sma200 = closes.rolling(200).mean().iloc[-1]
                    if not (price > sma50 and price > sma200 and sma20 > sma50):
                        continue

                    # -- Momentum: RSI(14) between 50-70 --
                    delta = closes.diff()
                    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
                    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
                    rs = gain / loss.replace(0, 1e-10)
                    rsi_val = float((100 - 100 / (1 + rs)).iloc[-1])
                    if not (50 <= rsi_val <= 70):
                        continue

                    # -- Trend strength: ADX(14) > 25 --
                    tr = pd.concat([
                        highs - lows,
                        (highs - closes.shift(1)).abs(),
                        (lows - closes.shift(1)).abs()
                    ], axis=1).max(axis=1)
                    up   = highs.diff()
                    down = -lows.diff()
                    plus_dm = pd.Series(
                        np.where((up > down) & (up > 0), up, 0.0),
                        index=closes.index)
                    minus_dm = pd.Series(
                        np.where((down > up) & (down > 0), down, 0.0),
                        index=closes.index)
                    atr14    = tr.ewm(alpha=1/14, min_periods=14).mean()
                    plus_di  = 100 * plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14
                    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14
                    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
                    adx_val  = float(dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1])
                    if adx_val < 25:
                        continue

                    # -- Performance: YTD > 20% --
                    year_str = str(pd.Timestamp.now().year)
                    dates = [b['date'] for b in bars]
                    ytd_idx = None
                    for i, d in enumerate(dates):
                        if d.startswith(year_str):
                            ytd_idx = i
                            break
                    if ytd_idx is None:
                        continue
                    ytd_perf = (price / closes.iloc[ytd_idx] - 1) * 100
                    if ytd_perf < 20:
                        continue

                    qualified.append((sym, ytd_perf, rsi_val, adx_val))
                except Exception:
                    continue

            logging.info(
                f"Supertrend Gold: {len(qualified)}/{len(to_check)} passed "
                f"technical filters"
            )

            # Rank by YTD performance descending
            qualified.sort(key=lambda x: x[1], reverse=True)
            result = [x[0] for x in qualified]

            # Cache full result before exclusion
            self._stg_cache = (now, result)

            filtered = [s for s in result if s not in exclude]
            return filtered[:max_tickers]

        except Exception as e:
            logging.error(f"Supertrend Gold screener failed: {e}")
            return []

    def news_score(self, ticker, lookback_hours=24, limit=30):
        """
        Returns an integer news severity score:
          0 = neutral/none
          1 = mild negative
          2 = strong negative
          3+ = severe negative (hard skip unless overridden)
        """
        key = (ticker, lookback_hours)
        now = time.time()

        # cache for 5 minutes
        if key in self.news_cache and (now - self.news_cache[key][0] < 300):
            return self.news_cache[key][1]

        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={KEYS['FMP']}"
            items = requests.get(url, timeout=5).json() or []
        except:
            items = []

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=lookback_hours)

        severe = ["fraud", "subpoena", "sec", "doj", "indict", "bankrupt", "restatement", "delist"]
        strong = ["lawsuit", "investigation", "probe", "recall", "halt", "crash", "explosion", "guidance cut"]
        mild   = ["downgrade", "miss", "cuts", "warns", "weak", "slows", "delay"]

        score = 0
        for it in items:
            title = (it.get("title") or "").lower()
            dt_str = it.get("publishedDate") or it.get("published_date") or ""
            try:
                dt = pd.to_datetime(dt_str, utc=True)
            except:
                dt = None
            if dt is not None and dt < cutoff:
                continue

            if any(w in title for w in severe):
                score = max(score, 3)
            elif any(w in title for w in strong):
                score = max(score, 2)
            elif any(w in title for w in mild):
                score = max(score, 1)

        self.news_cache[key] = (now, score)
        return score

    def news_scores_batch(self, tickers, lookback_hours=24, limit=30, cache_ttl=300, chunk_size=20):
        """
        Batch news scoring for multiple tickers. 1 FMP call per chunk.
        Returns {ticker: score_int}.
        Use this for finalists only to reduce API calls by 10-50x.
        """
        tickers = list(dict.fromkeys([t for t in tickers if t]))  # Unique
        if not tickers:
            return {}

        key = ("news_batch", tuple(sorted(tickers)), lookback_hours, limit)
        now = time.time()

        if key in self.news_cache and (now - self.news_cache[key][0] < cache_ttl):
            return self.news_cache[key][1]

        severe = ["fraud", "subpoena", "sec", "doj", "indict", "bankrupt", "restatement", "delist"]
        strong = ["lawsuit", "investigation", "probe", "recall", "halt", "crash", "explosion", "guidance cut"]
        mild   = ["downgrade", "miss", "cuts", "warns", "weak", "slows", "delay"]

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=lookback_hours)
        scores = {t: 0 for t in tickers}

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            try:
                url = f"https://financialmodelingprep.com/api/v3/stock_news"
                params = {"tickers": ",".join(chunk), "limit": limit, "apikey": KEYS["FMP"]}
                items = requests.get(url, params=params, timeout=5).json() or []
            except:
                items = []

            for it in items:
                tkr = (it.get("symbol") or it.get("ticker") or "").upper()
                if tkr not in scores:
                    continue

                dt_str = it.get("publishedDate") or it.get("published_date") or ""
                try:
                    dt = pd.to_datetime(dt_str, utc=True)
                except:
                    dt = None
                if dt is not None and dt < cutoff:
                    continue

                title = (it.get("title") or "").lower()
                txt = (it.get("text") or "").lower()
                content = title + " " + txt

                if any(w in content for w in severe):
                    scores[tkr] = max(scores[tkr], 3)
                elif any(w in content for w in strong):
                    scores[tkr] = max(scores[tkr], 2)
                elif any(w in content for w in mild):
                    scores[tkr] = max(scores[tkr], 1)

        self.news_cache[key] = (now, scores)
        return scores

    def get_ratios(self, ticker):
        """Fetch Key TTM Ratios (P/E, Debt/Eq)"""
        try:
             url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={KEYS['FMP']}"
             res = requests.get(url, timeout=5).json()
             if res and isinstance(res, list):
                 return res[0]
        except: pass
        return {}

    def get_growth(self, ticker):
        """Fetch Financial Growth (Revenue, Income)"""
        try:
             url = f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?limit=1&apikey={KEYS['FMP']}"
             res = requests.get(url, timeout=5).json()
             if res and isinstance(res, list):
                 return res[0]
        except: pass
        return {}

    def get_fundamental_features(self, ticker):
        """
        Fetch fundamental features for ML: earnings_surprise, revenue_growth_yoy, pe_ratio.
        Cached for 24h per ticker. Returns dict with defaults on failure.
        """
        cache_key = ("fundamentals", ticker)
        now = time.time()
        if cache_key in self.news_cache and (now - self.news_cache[cache_key][0] < 86400):
            return self.news_cache[cache_key][1]

        result = {'earnings_surprise': 0.0, 'revenue_growth_yoy': 0.0, 'pe_ratio': 20.0, 'news_impact_weight': 0.0}
        try:
            # PE ratio from TTM ratios
            ratios = self.get_ratios(ticker)
            pe = float(ratios.get('peRatioTTM', 0) or 0)
            result['pe_ratio'] = max(-100, min(500, pe)) if pe else 20.0

            # Revenue growth YoY
            growth = self.get_growth(ticker)
            rev_g = float(growth.get('revenueGrowth', 0) or 0)
            result['revenue_growth_yoy'] = max(-1.0, min(5.0, rev_g))

            # Earnings surprise from earnings calendar
            url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={KEYS['FMP']}"
            res = requests.get(url, timeout=5).json()
            if res and isinstance(res, list) and len(res) > 0:
                latest = res[0]
                actual = float(latest.get('actualEarningResult', 0) or 0)
                estimated = float(latest.get('estimatedEarning', 0) or 0)
                if estimated != 0:
                    result['earnings_surprise'] = max(-5.0, min(5.0, (actual - estimated) / abs(estimated)))
        except Exception as e:
            logging.debug(f"Fundamental features {ticker}: {e}")

        self.news_cache[cache_key] = (now, result)
        return result

class FundamentalGuard:
    """
    Filter out companies with toxic fundamentals (Debt spiral, collapsing revenue).
    """
    def __init__(self, fmp_helper):
        self.fmp = fmp_helper
        self.cache = {} # ticker -> (expiry, is_safe)

    def check_healthy(self, ticker):
        now = datetime.datetime.now()
        if ticker in self.cache:
            exp, safe = self.cache[ticker]
            if now < exp:
                return safe

        try:
            # 1. Check Ratios (Debt)
            ratios = self.fmp.get_ratios(ticker)
            debt_eq = float(ratios.get('debtEquityRatio', 0) or 0)

            # 2. Check Growth
            growth = self.fmp.get_growth(ticker)
            rev_growth = float(growth.get('revenueGrowth', 0) or 0)

            is_safe = True

            # TOXIC RULES
            # 1. Massive Debt (> 5.0 D/E)
            if debt_eq > 5.0:
                is_safe = False
                # logging.debug(f"Fundamentally Toxic {ticker}: High Debt ({debt_eq:.1f})")

            # 2. Collapsing Revenue (< -20% YoY)
            if rev_growth < -0.20:
                is_safe = False
                # logging.debug(f"Fundamentally Toxic {ticker}: Collapsing Revenue ({rev_growth:.1%})")

            # Cache for 24 hours
            self.cache[ticker] = (now + datetime.timedelta(hours=24), is_safe)
            return is_safe

        except Exception as e:
            # Fail safe (allow if data missing)
            self.cache[ticker] = (now + datetime.timedelta(hours=6), True)
            return True

class SeekingAlpha_Helper:
    """
    RapidAPI Seeking Alpha wrapper for robust news/sentiment features.
    Docs: https://rapidapi.com/seeking-alpha-seeking-alpha-default/api/seeking-alpha-finance
    """
    def __init__(self):
        self.host = "seeking-alpha-finance.p.rapidapi.com"
        self.key = KEYS.get('RAPIDAPI_KEY')
        self.base = f"https://{self.host}"
        self.sess = requests.Session()
        self.cache_dir = os.path.join(DRIVE_ROOT, 'market_cache', 'sa_cache')
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)

    def _get(self, path, params=None, cache_key=None):
        if not self.key: return {}

        # Check cache
        if cache_key:
            cp = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cp):
                try:
                    with open(cp, 'r') as f: return json.load(f)
                except Exception as e:
                    logging.debug(f"SA Cache Read Error: {e}")

        headers = {
            "x-rapidapi-key": self.key,
            "x-rapidapi-host": self.host
        }
        try:
             url = f"{self.base}{path}"
             time.sleep(0.25) # Polite rate limit handling
             r = self.sess.get(url, headers=headers, params=params, timeout=10)
             if r.status_code == 200:
                 data = r.json()
                 if cache_key:
                     with open(cp, 'w') as f: json.dump(data, f)
                 return data
             else:
                 logging.debug(f"SA {path} error: {r.status_code}")
                 return {}
        except Exception as e:
            logging.debug(f"SA request failed: {e}")
            return {}

    def get_news_features(self, ticker):
        """
        Return dict of features:
          - sa_news_count_3d/7d
          - sa_sentiment_score (Text Sentiment)
        """
        if not self.key:
            return {'sa_news_count_3d': 0, 'sa_news_count_7d': 0, 'sa_sentiment_score': 0}

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        data = self._get("/v1/symbols/news",
                         {"ticker_slug": ticker.lower()},
                         cache_key=f"news_{ticker}_{today}")

        now = datetime.datetime.now(datetime.timezone.utc)
        count_3d = 0
        count_7d = 0
        sentiment_score = 0.0

        # Simple Lexicon
        pos_words = {'beat', 'jump', 'surge', 'buy', 'strong', 'upgrade', 'record', 'growth', 'bull', 'top', 'winner'}
        neg_words = {'miss', 'fall', 'drop', 'sell', 'weak', 'downgrade', 'loss', 'bear', 'crash', 'risk', 'warn'}

        items = data.get('data', [])
        scored_items = 0

        for item in items:
            try:
                attr = item.get('attributes', {})
                pub_str = attr.get('publishOn')
                if not pub_str: continue

                pub = pd.to_datetime(pub_str, utc=True)
                age = (now - pub).days

                if age <= 7:
                    if age <= 3: count_3d += 1
                    count_7d += 1

                    # Sentiment Analysis (Title)
                    title = (attr.get('title') or "").lower()
                    title_words = set(title.split())

                    s_val = 0
                    if title_words & pos_words: s_val += 1
                    if title_words & neg_words: s_val -= 1

                    sentiment_score += s_val
                    scored_items += 1
            except: pass

        avg_sentiment = (sentiment_score / scored_items) if scored_items > 0 else 0

        return {
            'sa_news_count_3d': count_3d,
            'sa_news_count_7d': count_7d,
            'sa_sentiment_score': avg_sentiment
        }

    def get_ratings(self, ticker):
        """
        Fetch Quant and Analyst Ratings.
        Returns: {'sa_quant_rating': 1-5, 'sa_analyst_rating': 1-5}
        """
        if not self.key: return {'sa_quant_rating': 3, 'sa_analyst_rating': 3}

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        # Try 'summary' endpoint for ratings
        data = self._get("/v1/symbols/summary",
                         {"ticker_slug": ticker.lower()},
                         cache_key=f"summary_{ticker}_{today}")

        try:
            # Structure varies, looking for top-level rating fields
            # or data.attributes.ratings
            attr = data.get('data', {}).get('attributes', {})

            # Map labels to 1-5 scale (Strong Sell=1, Strong Buy=5)
            def map_rating(val):
                if not val: return 3
                val = float(val)
                return max(1, min(5, val))

            # hypothetical fields - API specific
            quant = map_rating(attr.get('quantRating', 3))
            authors = map_rating(attr.get('authorsRating', 3))

            return {
                'sa_quant_rating': quant,
                'sa_analyst_rating': authors
            }
        except:
            return {'sa_quant_rating': 3, 'sa_analyst_rating': 3}

class Alpaca_Helper:
    def __init__(self, db, mc_governor=None, perf_monitor=None):
        self.api = tradeapi.REST(KEYS['ALPACA_KEY'], KEYS['ALPACA_SEC'], ALPACA_URL, 'v2')
        self.db = db
        self.mc_governor = mc_governor
        self.perf_monitor = perf_monitor  # ENHANCED: Performance tracking for drift detection
        self.pos_cache = {}
        self._pos_lock = threading.RLock()  # FIX: Thread safety for pos_cache
        self._stop_replace_last = defaultdict(float)  # v15.10: Rate limit stop updates per symbol

        # FIX: Load persisted pending orders from DB on startup
        self.pending_orders = self.db.load_pending_orders()
        if self.pending_orders:
            logging.info(f"🔄 Recovered {len(self.pending_orders)} pending orders from DB")
        self._init_equity()

        # FIX: Prime Governor with history so it works on restart
        if self.mc_governor:
            self._prime_governor()

        self.sync_positions()

    def _init_equity(self):
        try:
            account = self.api.get_account()
            self.equity = float(account.equity)
            self.peak_equity = self.equity
            logging.info(f"💰 Equity: ${self.equity:,.2f}")
        except Exception as e:
            logging.error(f"Equity fetch failed: {e}")
            self.equity = 50000.0
            self.peak_equity = 50000.0

    def refresh_equity(self):
        """FIX: Refresh equity every loop for accurate sizing"""
        try:
            account = self.api.get_account()
            self.equity = float(account.equity)
            self.peak_equity = max(self.peak_equity, self.equity)
            return self.equity
        except Exception as e:
            ERROR_TRACKER.record_failure("EquityRefresh", str(e))
            return self.equity

    def _prime_governor(self):
        """Load historical trade outcomes to prime the MC Governor"""
        try:
            # Fetch last 180 days of trades
            trades = self.db.get_trade_outcomes(days=180)
            if not trades:
                return

            # Sort Oldest -> Newest (DB returns DESC)
            trades.reverse()

            count = 0
            for t in trades:
                pnl = t.get('pnl', 0) or 0
                r = t.get('pnl_r', 0) or 0
                days_ago = t.get('days_ago', 0)

                # Reconstruct Approx Timestamp
                ts = datetime.datetime.now() - datetime.timedelta(days=days_ago)

                # We need pnl and R to exist and be valid
                if abs(r) > 0.01 and abs(pnl) > 0:
                    # Reverse engineer Risk Dollars: R = PnL / Risk$  => Risk$ = PnL / R
                    risk_dollars = abs(pnl / r)

                    # Feed Governor
                    # Note: We rely on DB PnL being net of fees? Usu. yes.
                    self.mc_governor.add_trade(pnl, risk_dollars, side=t.get('side'), timestamp=ts)
                    count += 1

            logging.info(f"🎲 Primed Monte Carlo with {count} historical trades")
        except Exception as e:
            logging.warning(f"Failed to prime Governor: {e}")

    def sync_positions(self):
        """FIX: Sync with ACTUAL broker positions + rebuild pending from broker open orders"""
        try:
            # Detect closed positions first (before updating cache)
            if self.mc_governor and self.pos_cache:
                current_symbols = {p.symbol for p in self.api.list_positions()}
                cached_symbols = set(self.pos_cache.keys())
                closed_symbols = cached_symbols - current_symbols

                for s in closed_symbols:
                    try:
                        # Position closed! Calculate Realized PnL and Risk$
                        cached = self.pos_cache[s]
                        entry = cached.get('entry', 0)
                        sl = cached.get('sl', 0)
                        qty = cached.get('qty', 0)
                        side = cached.get('side')

                        # Fix 1: Use cached init_risk (per share) -> Total Risk Dollars
                        init_risk_per_share = cached.get("init_risk") or abs(entry - sl)
                        risk_dollars = init_risk_per_share * qty if qty > 0 else 0

                        # Fetch Actual PnL from Activities
                        acts = self.api.get_activities(activity_types='FILL', direction='desc', limit=50)

                        # Fix 2: Correct closing side mapping
                        want_side = "sell" if side == "LONG" else "buy"

                        exit_price = entry # Fallback
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
                            break # Found the closing fill (most recent)

                        if found_fill and filled_qty > 0:
                            pnl = (exit_price - entry) * filled_qty if side == 'LONG' else (entry - exit_price) * filled_qty
                            pnl_r = pnl / risk_dollars if risk_dollars > 0 else 0.0
                            logging.info(f"🏁 Trade Closed {s}: PnL=${pnl:.2f} R={pnl_r:+.2f} Risk=${risk_dollars:.2f}")
                            self.mc_governor.add_trade(pnl, risk_dollars, side=side)

                            # Track outcome for drift detection
                            if self.perf_monitor:
                                self.perf_monitor.add_trade_result(pnl > 0)

                            # FIX: Log trade outcome WITH entry_features for retraining
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
                            logging.info(f"🏁 Trade Closed {s} (Fill not found in recent acts)")

                    except Exception as e:
                        logging.warning(f"Failed to process closed trade {s}: {e}")

            # Get real positions from broker
            broker_positions = {p.symbol: p for p in self.api.list_positions()}

            # FIX: Rebuild pending_orders from broker's open orders (handles restarts)
            # Only track BRACKET PARENT orders, not child SL/TP legs
            try:
                # FIX: Use nested=True to get leg info
                open_orders = self.api.list_orders(status='open', nested=True)
                broker_order_ids = set()

                for order in open_orders:
                    # FIX: Child orders have parent_order_id; parents don't
                    parent_id = getattr(order, 'parent_order_id', None)
                    if parent_id is not None:
                        continue  # This is a child leg, skip

                    # Also check order_class for bracket parents
                    order_class = getattr(order, 'order_class', None)
                    legs = getattr(order, 'legs', None)
                    order_type = getattr(order, 'type', None)

                    # Parent entry is: order_class='bracket' OR 'oco' OR 'oto'
                    is_bracket_parent = (
                        order_class in ['bracket', 'oco', 'oto'] and
                        order_type in ['market', 'limit', None]
                    )

                    # FIX: Filter non-bracket parents
                    if not is_bracket_parent:
                        continue

                    broker_order_ids.add(order.id)

                    if order.id not in self.pending_orders:
                        # FIX: Extract SL/TP from bracket legs
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
                            except:
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

                        # FIX: Persist recovered orders to DB
                        self.db.save_pending_order(order.id, order.symbol, side, int(order.qty), price, sl, tp, atr)
                        logging.info(f"🔄 Recovered bracket order: {order.symbol} SL=${sl:.2f} TP=${tp:.2f}")

                # Remove pending orders that no longer exist at broker
                for order_id in list(self.pending_orders.keys()):
                    if order_id not in broker_order_ids:
                        # Order no longer open - check if it filled or canceled
                        try:
                            order = self.api.get_order(order_id)
                            if order.status in ['filled', 'canceled', 'expired', 'rejected']:
                                logging.info(f"🔄 Order {order.status}: {self.pending_orders[order_id]['symbol']}")
                                self.db.delete_pending_order(order_id)
                                del self.pending_orders[order_id]
                        except:
                            # Order doesn't exist - remove
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
                        # Order filled - now it's a real position
                        if symbol in broker_positions:
                            p = broker_positions[symbol]
                            entry = float(p.avg_entry_price)
                            qty = int(abs(float(p.qty)))
                            side = order_info['side']
                            # FIX: Use STORED SL/TP from order time (not recomputed)
                            sl = order_info.get('sl', 0)
                            tp = order_info.get('tp', 0)
                            atr = order_info.get('atr', 0) or get_daily_atr(symbol) or entry * 0.02

                            # If SL/TP were not stored, compute them
                            if sl == 0 or tp == 0:
                                atr = get_daily_atr(symbol) or entry * 0.02
                                sl = entry - 1.5*atr if side=='LONG' else entry + 1.5*atr
                                tp = entry + 3.0*atr if side=='LONG' else entry - 3.0*atr


                            # init_risk for R calc
                            init_risk = abs(entry - sl) if sl else abs(entry * 0.02)

                            self.pos_cache[symbol] = {
                                'entry': entry, 'qty': qty, 'side': side,
                                'ts': datetime.datetime.now().isoformat(),
                                'atr': atr, 'pyramided': False,
                                'sl': sl, 'tp': tp,  # Store for reference
                                'init_sl': sl,       # v15.10: Original Stop
                                'init_risk': init_risk, # v15.10: Original Risk (R)
                                'ratcheted': False,  # v15.10: Track if SL moved to Breakeven
                                'be_moved': False,   # Legacy support
                                'entry_features': order_info.get('entry_features', {})  # NEW: For learning
                            }
                            self.db.update_position(symbol, entry, qty, side, sl, tp, atr, False)
                            self.db.log_trade(symbol, side, qty, entry, 0, "Filled")
                            logging.info(f"✅ Order filled: {side} {symbol} x{qty} @ ${entry:.2f}")

                        self.db.delete_pending_order(order_id)
                        del self.pending_orders[order_id]

                    elif order.status in ['canceled', 'expired', 'rejected']:
                        logging.warning(f"⚠️ Order {order.status}: {symbol}")
                        self.db.delete_pending_order(order_id)
                        del self.pending_orders[order_id]

                except Exception as e:
                    logging.debug(f"Order check failed: {e}")

            # Sync pos_cache with actual broker positions
            # Remove positions from cache that don't exist at broker
            for symbol in list(self.pos_cache.keys()):
                if symbol not in broker_positions:
                    del self.pos_cache[symbol]
                    self.db.delete_position(symbol)

            # Add positions from broker that aren't in cache (e.g. manual trades)
            for symbol, p in broker_positions.items():
                if symbol not in self.pos_cache:
                    entry = float(p.avg_entry_price)
                    qty = int(abs(float(p.qty)))
                    side = 'LONG' if float(p.qty) > 0 else 'SHORT'
                    atr = get_daily_atr(symbol) or entry * 0.02
                    # Assume 1.5 ATR stop for manual/unknown trades
                    sl = entry - 1.5*atr if side=='LONG' else entry + 1.5*atr
                    tp = entry + 3.0*atr if side=='LONG' else entry - 3.0*atr

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
                    logging.info(f"🔄 Position synced from broker: {side} {symbol}")

            self.refresh_equity()
            logging.info(f"✅ Sync: {len(self.pos_cache)} positions, {len(self.pending_orders)} pending")
        except Exception as e:
            ERROR_TRACKER.record_failure("SyncPositions", str(e))

    def _find_open_stop_order_id(self, symbol):
        """
        v15.10: Find the open STOP leg for a symbol (from a filled bracket).
        Returns order_id or None.
        """
        try:
            orders = self.api.list_orders(status='open', nested=True)
            for o in orders:
                if getattr(o, "symbol", None) != symbol:
                    continue
                otype = getattr(o, "type", None)
                # stop or stop_limit are the stop-loss legs
                if otype in ["stop", "stop_limit"]:
                    return o.id
        except Exception as e:
            logging.debug(f"_find_open_stop_order_id failed for {symbol}: {e}")
        return None

    def replace_stop(self, symbol, new_stop):
        """
        v15.10: Replace the open stop-loss leg price for an existing bracket.
        Rate-limited to avoid spamming.
        FIX: Handle both stop and stop_limit order types.
        """
        now = time.time()
        if now - self._stop_replace_last[symbol] < SETTINGS.get("STOP_REPLACE_COOLDOWN_SEC", 60):
            return False

        stop_id = self._find_open_stop_order_id(symbol)
        if not stop_id:
            logging.debug(f"No open stop leg found for {symbol}")
            return False

        try:
            # FIX: Get order type to handle stop_limit properly
            order = self.api.get_order(stop_id)
            order_type = getattr(order, 'type', None)

            payload = {'stop_price': round(float(new_stop), 2)}

            # FIX: If stop_limit, also update limit_price maintaining offset
            if order_type == 'stop_limit':
                old_stop = float(getattr(order, 'stop_price', 0) or 0)
                old_limit = float(getattr(order, 'limit_price', 0) or 0)
                if old_limit and old_stop:
                    # Maintain the same offset between stop and limit
                    offset = old_limit - old_stop
                    payload['limit_price'] = round(float(new_stop) + offset, 2)
                else:
                    # Fallback: set limit = stop (tight)
                    payload['limit_price'] = payload['stop_price']

            self.api.replace_order(stop_id, **payload)
            self._stop_replace_last[symbol] = now
            logging.info(f"🛡️ SL Replaced {symbol} -> ${new_stop:.2f}")
            return True
        except Exception as e:
            logging.warning(f"replace_stop failed for {symbol}: {e}")
            return False

    def replace_stop_loss(self, symbol, new_stop):
        """Alias for legacy calls"""
        return self.replace_stop(symbol, new_stop)

    def calculate_position_size(self, entry_price, stop_price, vix_mult=1.0):
        """Risk-based sizing: qty = (equity * risk_pct) / stop_distance
        FIX: Apply slippage haircut for market orders
        """
        risk_per_trade = self.equity * SETTINGS['RISK_PER_TRADE']
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            return 0

        qty = int((risk_per_trade * vix_mult) / stop_distance)
        max_qty = int(self.equity * 0.20 / entry_price)
        qty = min(max(qty, 0), max_qty)

        # FIX: Apply slippage haircut for market orders
        if SETTINGS.get('USE_MARKET_ORDERS', False):
            haircut = SETTINGS.get('SLIPPAGE_HAIRCUT', 0.10)
            qty = int(qty * (1 - haircut))

        return qty

    def submit_bracket(self, t, side, qty, current_price, sl, tp, atr_override=None, entry_features=None):
        """
        FIXED: Robust error handling with Idempotency check.
        Tracks as pending order; adds to DB orders table.
        """
        global KILL_TRIGGERED
        if KILL_TRIGGERED or qty <= 0: return False

        # 0. Fast Guard: Don't stack pending or open positions
        with self._pos_lock:
            if t in self.pos_cache:
                logging.debug(f"Already have open position for {t}")
                return False
            for oi in self.pending_orders.values():
                if oi['symbol'] == t:
                    logging.debug(f"Already have pending order for {t}")
                    return False

        # Phase 21: Idempotency Token (deterministic per symbol/side/minute)
        ts_token = int(time.time() // 60)
        client_oid = f"gm_v14_{t}_{side}_{ts_token}"

        # FIX: Check broker open orders for duplicate client_order_id before submitting
        try:
            open_orders = self.api.list_orders(status='open', limit=50)
            if any(getattr(o, 'client_order_id', '') == client_oid for o in open_orders):
                logging.debug(f"Duplicate CID already open at broker: {client_oid}")
                return False
        except Exception:
            pass  # Proceed anyway if check fails

        logging.info(f"🚀 Submitting {side} {t} x{qty} (CID: {client_oid})")

        # 1. Log Intent
        self.db.upsert_order(client_oid, t, side, qty, "BRACKET", "INTENT_DECLARED",
                             raw_json=json.dumps({'sl': sl, 'tp': tp, 'price': current_price}))

        try:
            order_side = 'buy' if side == 'LONG' else 'sell'

            # 2. Submit API
            if SETTINGS['USE_MARKET_ORDERS']:
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

            # 3. Log Success
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

            logging.info(f"✅ Bracket Sent: {t} {side} (ID: {order.id})")
            return True

        except Exception as e:
            # Handle duplicates gracefully
            msg = str(e).lower()
            if "client_order_id" in msg and ("already exists" in msg or "duplicate" in msg):
                 logging.warning(f"⚠️ Duplicate CID; assuming sync will recover: {client_oid}")
                 self.db.upsert_order(client_oid, t, side, int(qty), "BRACKET", "DUPLICATE_CID", raw_json=str(e))
                 return False

            # Log actual failures
            ERROR_TRACKER.record_failure(f"Order_{t}", str(e))
            self.db.upsert_order(client_oid, t, side, int(qty), "BRACKET", "FAILED", raw_json=str(e))
            logging.error(f"❌ Bracket Failed {t}: {e}")
            return False

    def close(self, symbol, reason="Manual"):
        """Close a position, calculate P&L, and log outcome for learning"""
        try:
            # FIX: Get current price for accurate P&L before closing
            current_price = None
            try:
                quote = self.api.get_latest_trade(symbol)
                current_price = float(quote.price)
            except Exception as e:
                logging.debug(f"Could not get latest price for {symbol}: {e}")

            self.api.close_position(symbol)

            if symbol in self.pos_cache:
                with self._pos_lock:  # FIX: Thread safety
                    pos = self.pos_cache[symbol]
                    entry = pos['entry']
                    qty = pos['qty']
                    side = pos['side']
                    atr = pos.get('atr', entry * 0.02)  # Fallback ATR

                    # FIX: Calculate actual realized P&L
                    if current_price:
                        if side == 'LONG':
                            pnl = (current_price - entry) * qty
                        else:  # SHORT
                            pnl = (entry - current_price) * qty
                    else:
                        pnl = 0  # Unknown if we couldn't get price

                    # NEW: Calculate P&L in R-multiples for learning
                    init_risk = pos.get('init_risk')
                    if init_risk and init_risk > 0 and qty > 0:
                        # R = PnL / (RiskPerShare * Qty)
                        # RiskPerShare is what init_risk stores?
                        # pos_cache init_risk assignment (line 1924) is abs(entry - sl). That is per share.
                        # Wait, line 1916: init_risk = abs(entry - sl). Yes per share.
                        # So Risk$ = init_risk * qty.
                        pnl_r = pnl / (init_risk * qty)
                    else:
                        # Fallback
                        risk_per_share = atr * SETTINGS.get('STOP_MULT', 1.5) if atr else entry * 0.03
                        pnl_r = (pnl / qty / risk_per_share) if risk_per_share and qty else 0

                    # Regression outcome MUST be realized R-multiple
                    outcome = float(pnl_r) if pnl_r is not None else 0.0

                    # NEW: Log outcome with entry features for learning
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
                    logging.info(f"🛑 Closed {symbol} ({reason}) P&L: {pnl_str} {r_str}")
            else:
                logging.info(f"🛑 Closed {symbol} ({reason})")
        except Exception as e:
            logging.error(f"Close failed {symbol}: {e}")

    def close_position(self, symbol, reason="Manual"):
        """Alias for close() to fix bug in time-stop logic"""
        return self.close(symbol, reason)

    def check_kill(self, ):
        global KILL_TRIGGERED
        if KILL_TRIGGERED:
            return True
        try:
            self.equity = float(self.api.get_account().equity)
            self.peak_equity = max(self.peak_equity, self.equity)
            drawdown = 1 - (self.equity / self.peak_equity)

            if drawdown >= SETTINGS['KILL_SWITCH_DD']:
                send_alert("🛑 KILL SWITCH", f"DD: {drawdown:.1%}", "high")
                logging.critical(f"🛑 KILL: DD={drawdown:.1%}")
                self.api.cancel_all_orders()
                self.api.close_all_positions()
                KILL_TRIGGERED = True
                return True
        except Exception as e:
            ERROR_TRACKER.record_failure("KillCheck", str(e))
        return False

# ==============================================================================
# ==============================================================================
# INSTITUTIONAL-GRADE LOSS FUNCTIONS & MICROSTRUCTURE FEATURES
# ==============================================================================

def asymmetric_loss_objective(y_true, y_pred):
    """
    Custom XGBoost objective: Asymmetric Loss for High Win Rate
    Penalizes false signals (wrong direction) 10x more than missed opportunities
    This forces the model to only predict when highly confident

    Loss = {
        10.0 * (y_true - y_pred)^2  if sign(y_true) != sign(y_pred)  (WRONG DIRECTION)
        1.0 * (y_true - y_pred)^2   if sign(y_true) == sign(y_pred)  (RIGHT DIRECTION)
    }

    Returns: (gradient, hessian)
    """
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)

    residual = y_true - y_pred
    wrong_direction = np.sign(y_true) != np.sign(y_pred)

    # Asymmetric penalty: 10x for wrong direction
    penalty = np.where(wrong_direction, 10.0, 1.0)

    # Gradient: d/dy_pred [ penalty * (y_true - y_pred)^2 ] = -2 * penalty * (y_true - y_pred)
    grad = -2.0 * penalty * residual

    # Hessian: d^2/dy_pred^2 = 2 * penalty
    hess = 2.0 * penalty

    return grad, hess


def profit_factor_objective(y_true, y_pred):
    """
    🎯 INSTITUTIONAL-GRADE: Differentiable Profit Factor Objective

    Directly optimizes for Profit Factor = Gross_Wins / Gross_Losses
    This is THE KEY to achieving >1.5 PF systematically.

    Mathematical Strategy:
    1. Classify predictions as "winners" or "losers" using soft sigmoid
    2. Calculate Gross Profit (sum of winning R-values)
    3. Calculate Gross Loss (sum of losing R-values)
    4. Minimize -log(PF) = -log(GP) + log(GL)

    The sigmoid smoothing makes the objective differentiable while
    maintaining the economic interpretation of profit factor.

    Returns: (gradient, hessian) for XGBoost
    """
    epsilon = 1e-6

    # Agreement score: y_true * y_pred
    # Positive = correct direction, negative = wrong direction
    agreement = y_true * y_pred

    # Soft classification using sigmoid for differentiability
    # sigmoid(agreement/temp) = probability this trade is a "winner"
    temp = 2.0  # Temperature (lower = sharper, higher = smoother)
    soft_win_prob = 1.0 / (1.0 + np.exp(-agreement / temp))
    soft_loss_prob = 1.0 - soft_win_prob

    # Weighted returns (use absolute value of y_true as magnitude)
    win_contribution = soft_win_prob * np.abs(y_true)
    loss_contribution = soft_loss_prob * np.abs(y_true)

    # Batch-level gross profit and gross loss
    gross_profit = np.sum(win_contribution) + epsilon
    gross_loss = np.sum(loss_contribution) + epsilon

    # Profit Factor
    pf = gross_profit / gross_loss

    # We MAXIMIZE PF by MINIMIZING -log(PF)
    # Gradient calculation:
    # d(-log(PF))/d(y_pred) = d(-log(GP) + log(GL))/d(y_pred)

    # Sigmoid derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    sigmoid_derivative = soft_win_prob * (1.0 - soft_win_prob)

    # Chain rule:
    # d(win_prob)/d(y_pred) = sigmoid'(agreement/temp) * y_true / temp
    d_win_prob = sigmoid_derivative * y_true / temp
    d_loss_prob = -d_win_prob  # Loss prob decreases when win prob increases

    # Gradient of loss w.r.t. y_pred
    grad = -(1.0 / gross_profit) * d_win_prob * np.abs(y_true) + \
           (1.0 / gross_loss) * d_loss_prob * np.abs(y_true)

    # Hessian approximation (constant diagonal for stability)
    # Use sigmoid curvature as proxy
    hess = sigmoid_derivative * (1.0 / (temp**2)) * np.abs(y_true) + epsilon

    return grad, hess


def calculate_vpin(df, volume_bucket_size=None, window=50):
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading)

    VPIN measures order flow toxicity - high VPIN = informed traders present = avoid trading

    Args:
        df: DataFrame with ['Close', 'Volume', 'High', 'Low']
        volume_bucket_size: Volume per bucket (default: daily_vol / 50)
        window: Number of buckets for rolling VPIN

    Returns:
        Series of VPIN values (0-1, higher = more toxic)
    """
    if df is None or len(df) < window:
        return pd.Series(0.0, index=df.index if df is not None else [])

    # Auto-determine bucket size if not provided
    if volume_bucket_size is None:
        avg_daily_volume = df['Volume'].rolling(20).mean().median()
        volume_bucket_size = max(1, int(avg_daily_volume / 50))

    # Classify buy/sell volume using Bulk Volume Classification (BVC)
    # When price goes up, volume is buy-side; when down, sell-side
    price_change = df['Close'].diff()
    price_std = price_change.rolling(20).std()

    # Standardized price change → probability via CDF
    z_score = price_change / (price_std + 1e-9)
    buy_prob = norm.cdf(z_score)  # Probability this bar was buy-initiated

    df['Buy_Volume'] = df['Volume'] * buy_prob
    df['Sell_Volume'] = df['Volume'] * (1 - buy_prob)

    # Create volume buckets
    cumulative_volume = df['Volume'].cumsum()
    bucket_id = (cumulative_volume / volume_bucket_size).astype(int)

    # Aggregate by bucket
    bucket_imbalance = df.groupby(bucket_id).apply(
        lambda x: abs(x['Buy_Volume'].sum() - x['Sell_Volume'].sum())
    )
    bucket_volume = df.groupby(bucket_id)['Volume'].sum()

    # VPIN = Rolling average of |Buy - Sell| / Total Volume
    vpin_buckets = bucket_imbalance / (bucket_volume + 1)
    vpin_rolling = vpin_buckets.rolling(window, min_periods=10).mean()

    # Map back to original index
    bucket_to_vpin = vpin_rolling.to_dict()
    df['VPIN'] = df.groupby(bucket_id).ngroup().map(bucket_to_vpin).fillna(0)

    return df['VPIN'].clip(0, 1)  # Constrain to [0, 1]


def calculate_enhanced_vwap_features(df):
    """
    Enhanced VWAP features for institutional-grade mean reversion signals

    Returns dict of features:
        - VWAP_ZScore: Standardized distance from VWAP
        - VWAP_Slope: Rate of VWAP change (trending vs ranging)
        - VWAP_Volume_Ratio: Current volume vs VWAP period volume
    """
    features = {}

    # Standard VWAP (20-period)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

    # 1. VWAP Z-Score (standardized distance)
    vwap_dist = df['Close'] - vwap
    vwap_std = vwap_dist.rolling(20).std()
    features['VWAP_ZScore'] = vwap_dist / (vwap_std + 1e-9)

    # 2. VWAP Slope (momentum of VWAP itself)
    vwap_roc = vwap.pct_change(5)
    features['VWAP_Slope'] = vwap_roc.rolling(10).mean()

    # 3. Volume Confirmation (is current volume supporting the move?)
    avg_vwap_volume = df['Volume'].rolling(20).mean()
    features['VWAP_Volume_Ratio'] = df['Volume'] / (avg_vwap_volume + 1)

    return features


def calculate_volatility_regime(df):
    """
    GEX Proxy: Volatility Regime Detection

    Since real GEX requires options data, we proxy it via realized volatility patterns:
    - Low RV + Contracting ATR = Positive GEX Regime (mean reversion)
    - High RV + Expanding ATR = Negative GEX Regime (trending/breakouts)

    Returns:
        - Regime_GEX_Proxy: -1 (negative GEX), 0 (neutral), +1 (positive GEX)
        - Volatility_Regime: 'LOW', 'MEDIUM', 'HIGH'
    """
    # Calculate realized volatility (20-period rolling std of returns)
    returns = df['Close'].pct_change()
    realized_vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized

    # Calculate ATR expansion/contraction
    df['ATR_20'] = df.ta.atr(20)
    atr_change = df['ATR_20'].pct_change(5)

    # Regime classification
    vol_percentile = realized_vol.rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    # GEX Proxy Logic
    regime = pd.Series(0, index=df.index)  # Default neutral
    regime[(vol_percentile < 0.3) & (atr_change < 0)] = 1   # Positive GEX (low vol + contracting)
    regime[(vol_percentile > 0.7) & (atr_change > 0)] = -1  # Negative GEX (high vol + expanding)

    # Volatility Regime Labels
    vol_regime = pd.Series('MEDIUM', index=df.index)
    vol_regime[vol_percentile < 0.3] = 'LOW'
    vol_regime[vol_percentile > 0.7] = 'HIGH'

    return regime, vol_regime


def calculate_amihud_illiquidity(df, window=20):
    """
    Amihud Illiquidity Ratio: Measures price impact per dollar traded

    Illiquidity = |Return| / (Volume * Price)

    High values = low liquidity = high slippage risk = avoid trading
    """
    returns = df['Close'].pct_change().abs()
    dollar_volume = df['Volume'] * df['Close']

    illiquidity = returns / (dollar_volume + 1e-9)
    illiquidity_ratio = illiquidity.rolling(window).mean()

    # Normalize to percentile rank
    illiquidity_rank = illiquidity_ratio.rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    return illiquidity_rank


def calculate_real_relative_strength(stock_df, spy_df=None):
    """
    🔥 CRITICAL ALPHA FEATURE: Real Relative Strength (RRS)

    This is THE feature that separates institutional desks from retail.

    Logic:
    - If SPY drops 0.5% and NVDA is flat → Institutions are buying NVDA
    - When SPY bounces, NVDA will outperform massively

    Calculation:
    RRS = (Stock_Return - Beta * SPY_Return)

    A positive RRS means the stock is STRONGER than expected given market movement.
    This predicts future outperformance.

    Args:
        stock_df: DataFrame with 'Close' for the stock
        spy_df: DataFrame with 'Close' for SPY (market index)

    Returns:
        Series with RRS values
    """
    if spy_df is None or len(spy_df) < 50:
        # Fallback: use stock's own momentum if no SPY data
        return stock_df['Close'].pct_change(5)

    # Align indices
    stock_df = stock_df.copy()
    spy_df = spy_df.copy()

    # Calculate returns
    stock_returns = stock_df['Close'].pct_change()
    spy_returns = spy_df['Close'].pct_change()

    # Align by reindexing
    spy_returns_aligned = spy_returns.reindex(stock_df.index, method='ffill')

    # Calculate rolling beta (50-period)
    covariance = stock_returns.rolling(50).cov(spy_returns_aligned)
    spy_variance = spy_returns_aligned.rolling(50).var()
    beta = covariance / (spy_variance + 1e-9)
    beta = beta.fillna(1.0).clip(-3, 3)  # Constrain beta to reasonable range

    # Real Relative Strength = Actual Return - Expected Return (Beta * Market Return)
    expected_return = beta * spy_returns_aligned
    rrs = stock_returns - expected_return

    # Cumulative RRS over 5 periods (capture persistent strength)
    rrs_cumulative = rrs.rolling(5).sum()

    return rrs_cumulative.fillna(0.0)


def calculate_liquidity_sweep(df, lookback=16):
    """
    🎯 HIGH PROBABILITY REVERSAL: Liquidity Sweep Detection

    "The Fakeout" - Institutions push price beyond a key level to trigger
    retail stop losses, then reverse for easy profit.

    Logic:
    1. Price breaks above recent high (triggers retail breakout buyers)
    2. Price immediately closes BACK INSIDE the range (trap!)
    3. Volume spike confirms institutional participation

    This is a HIGH WIN RATE mean reversion signal.

    Args:
        df: DataFrame with OHLCV data
        lookback: Periods to look back for high/low (default 16 = 4 hours on 15m)

    Returns:
        Series with signals:
        +1 = Bullish sweep (fake breakdown → reversal up)
        -1 = Bearish sweep (fake breakout → reversal down)
         0 = No sweep
    """
    signals = pd.Series(0, index=df.index)

    # Calculate rolling high/low
    rolling_high = df['High'].rolling(lookback).max()
    rolling_low = df['Low'].rolling(lookback).min()

    # Volume threshold (must be above average for institutional involvement)
    avg_volume = df['Volume'].rolling(20).mean()
    volume_surge = df['Volume'] > (avg_volume * 1.5)

    for i in range(lookback + 1, len(df)):
        current_bar = df.iloc[i]
        prev_high = rolling_high.iloc[i-1]
        prev_low = rolling_low.iloc[i-1]

        # Bearish Sweep (Fake Breakout)
        # High breaks above recent high BUT close is back inside
        if (current_bar['High'] > prev_high and
            current_bar['Close'] < prev_high and
            volume_surge.iloc[i]):
            signals.iloc[i] = -1  # Expect reversal DOWN

        # Bullish Sweep (Fake Breakdown)
        # Low breaks below recent low BUT close is back inside
        elif (current_bar['Low'] < prev_low and
              current_bar['Close'] > prev_low and
              volume_surge.iloc[i]):
            signals.iloc[i] = 1  # Expect reversal UP

    return signals


def enhance_triple_barrier_labels(df, sl_mult=1.5, tp_mult=3.0, max_bars=12, atr_col='ATR'):
    """
    🎯 ENHANCED: Triple Barrier Labeling with Time Penalty

    Standard bracket labeling misses a critical fact: TIME HAS COST.
    A trade that goes nowhere for 3 hours is a FAILED trade even if
    it eventually breaks even.

    This enhancement:
    1. Labels trades that hit TP as WINNERS (positive R-value)
    2. Labels trades that hit SL as LOSERS (negative R-value)
    3. Labels trades that TIME OUT as LOSERS (small negative for opportunity cost)

    This forces the model to find MOMENTUM, not drift.

    Args:
        df: DataFrame with OHLC and ATR
        sl_mult: Stop loss multiplier of ATR
        tp_mult: Take profit multiplier of ATR
        max_bars: Maximum bars before timeout (vertical barrier)
        atr_col: Column name for ATR

    Returns:
        Series of R-values (positive = win, negative = loss, near-zero = timeout)
    """
    labels = pd.Series(0.0, index=df.index)

    for i in range(len(df) - max_bars):
        entry_price = df['Close'].iloc[i]
        atr = df[atr_col].iloc[i]

        if pd.isna(atr) or atr <= 0:
            continue

        # Define barriers
        sl_long = entry_price - sl_mult * atr
        tp_long = entry_price + tp_mult * atr
        sl_short = entry_price + sl_mult * atr
        tp_short = entry_price - tp_mult * atr

        # Look forward up to max_bars
        future_highs = df['High'].iloc[i+1:i+max_bars+1]
        future_lows = df['Low'].iloc[i+1:i+max_bars+1]

        # Check LONG trade outcome
        hit_tp_long = (future_highs >= tp_long).any()
        hit_sl_long = (future_lows <= sl_long).any()

        if hit_tp_long and hit_sl_long:
            # Both hit - check which came first
            tp_bar = (future_highs >= tp_long).idxmax() if hit_tp_long else max_bars
            sl_bar = (future_lows <= sl_long).idxmax() if hit_sl_long else max_bars

            if future_highs.index.get_loc(tp_bar) < future_lows.index.get_loc(sl_bar):
                labels.iloc[i] = tp_mult  # Winner: TP hit first
            else:
                labels.iloc[i] = -sl_mult  # Loser: SL hit first
        elif hit_tp_long:
            labels.iloc[i] = tp_mult  # Winner
        elif hit_sl_long:
            labels.iloc[i] = -sl_mult  # Loser
        else:
            # TIMEOUT (neither barrier hit)
            # Penalize with small negative (opportunity cost)
            # This forces model to avoid sideways chop
            final_price = df['Close'].iloc[min(i + max_bars, len(df) - 1)]
            unrealized_r = (final_price - entry_price) / atr

            # If unrealized is positive but didn't hit TP, penalize for slow momentum
            # If unrealized is negative, treat as partial loss
            labels.iloc[i] = min(unrealized_r * 0.3, -0.2)  # Cap at -0.2R penalty

    return labels


# ==============================================================================
# 8. AI MODEL (FIX #9: True per-ticker walk-forward with embargo)
# ==============================================================================
class WalkForwardAI:
    def __init__(self, use_ensemble=True):
        self.model = None  # Main/fallback model
        self.use_ensemble = use_ensemble

        # ENHANCED: Ensemble of specialist models
        self.trend_model = None  # Specialist for trending markets (ADX > 25)
        self.mean_reversion_model = None  # Specialist for mean-reversion (Hurst < 0.4)
        self.volatility_model = None  # Specialist for volatility breakouts (ATR spikes)
        self.meta_weights = {'trend': 0.33, 'mean_rev': 0.33, 'vol': 0.33}  # Default equal weights

        # Quantile models for prediction intervals (q10/q90)
        self.model_q10 = None
        self.model_q90 = None

        # Model stacking: diverse learners + meta-learner
        self.lgb_model = None       # LightGBM base learner
        self.ridge_model = None     # Ridge regression base learner
        self.stack_meta = None      # Meta-learner blending base predictions

        # SHAP feature monitoring
        self.shap_importances = {}  # {feature: importance} from last training
        self.shap_history = []      # List of (timestamp, {feature: importance}) for drift detection

        # ENHANCED: Expanded feature set from 6 to 30+ features (INSTITUTIONAL-GRADE)
        self.cols = [
            # Original features
            'RSI', 'ADX', 'ATR_Pct', 'Vol_Rel', 'Kalman_Dist', 'Hurst',
            # Price action features
            'BB_Width', 'BB_Position', 'VWAP_Dist', 'HL_Range',
            # Momentum features
            'ROC_5', 'ROC_20',
            # Volume features
            'Vol_Surge', 'Money_Flow',
            # Regime features
            'Volatility_Rank', 'Trend_Consistency',
            # ENHANCED: Sentiment features (SA API)
            'sa_news_count_3d', 'sa_sentiment_score',
            # ENHANCED: Time features (intraday patterns)
            'Hour', 'Day_of_Week',
            # Fundamental features
            'earnings_surprise', 'revenue_growth_yoy', 'pe_ratio',
            # Dynamic news weighting
            'news_impact_weight',
            # Market context features (cross-asset signals)
            'SPY_ROC_5', 'SPY_ROC_20', 'VIX_Level', 'VIX_ROC',
            # Cross-sectional factor ranks
            'momentum_rank', 'volume_rank', 'value_rank', 'composite_rank',
            # INSTITUTIONAL MICROSTRUCTURE FEATURES
            'VPIN', 'VWAP_ZScore', 'VWAP_Slope', 'VWAP_Volume_Ratio',
            'Regime_GEX_Proxy', 'Amihud_Illiquidity',
            'Volatility_Regime_Score',
            # CRITICAL ALPHA FEATURES (Institutional Edge)
            'RRS_Cumulative', 'Liquidity_Sweep', 'Beta_To_SPY'
        ]
        self.active_features = list(self.cols)  # v15: Track active features after IC check

        # ENHANCED: Optuna State - WEEKLY tuning to prevent overfitting
        self.best_params = None
        self.last_tune = datetime.datetime.now() - datetime.timedelta(days=8) # Force tune on start
        self.tune_lock = threading.Lock()

        # FIX: Save to DRIVE_ROOT/models, not CWD (persists across restarts)
        self.params_path = os.path.join(DRIVE_ROOT, "models", "best_params.json")
        self._load_best_params()

    def _load_best_params(self):
        try:
            if os.path.exists(self.params_path):
                with open(self.params_path, 'r') as f:
                    self.best_params = json.load(f)
                logging.info(f"🧠 Loaded persisted Best Params: {self.best_params}")
        except Exception as e:
            logging.warning(f"Failed to load best_params: {e}")

    def _save_best_params(self):
        try:
            with open(self.params_path, 'w') as f:
                json.dump(self.best_params, f)
            logging.info("🧠 Saved Best Params to disk")
        except Exception as e:
            logging.error(f"Failed to save best_params: {e}")

    def tune(self, X, y):
        """
        v15.12: Daily Optuna Tuning
        Objective: Maximize Expectancy * sqrt(N_Trades) using TimeSeriesSplit (Walk-Forward)
        """
        with self.tune_lock:
            # Only tune if enough data (e.g. > 1000 samples)
            if len(X) < 1000:
                logging.info("🧠 Not enough data for Optuna tuning.")
                return

            logging.info("🧠 Starting Optuna Tuning (20 trials, TimeSeriesSplit)...")

            def objective(trial):
                # Suggest params
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'verbosity': 0,
                    'n_jobs': 1
                }

                # ENHANCED: Valid Walk-Forward CV with Embargo (Regression Mode)
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                embargo = 130  # FIX: 1-week embargo (Phase 19) to prevent lookahead

                for train_idx, val_idx in tscv.split(X):
                    train_end = train_idx[-1]
                    # Apply embargo: skip samples immediately after train
                    val_idx_embargoed = val_idx[val_idx > (train_end + embargo)]

                    if len(val_idx_embargoed) < 50:
                        continue

                    train_X, val_X = X.iloc[train_idx], X.iloc[val_idx_embargoed]
                    train_y, val_y = y.iloc[train_idx], y.iloc[val_idx_embargoed]

                    # ENHANCED: Use XGBRegressor for R-value prediction
                    model = XGBRegressor(**params, eval_metric='mae')
                    model.fit(train_X, train_y)

                    # Predict R-values
                    preds = model.predict(val_X)

                    # Filter to significant predictions (abs(R) > 0.10, equivalent to trade threshold)
                    trade_mask = np.abs(preds) > 0.10

                    if trade_mask.sum() < 5:
                        scores.append(-1.0)  # Penalize inactivity
                        continue

                    # Calculate actual performance on traded samples
                    trades_preds = preds[trade_mask]
                    trades_actuals = val_y.values[trade_mask]

                    # Correlation between predicted and actual R-values
                    if len(trades_preds) > 5:
                        corr = np.corrcoef(trades_preds, trades_actuals)[0, 1]
                        corr = 0.0 if np.isnan(corr) else corr
                    else:
                        corr = 0.0

                    # FIX: With signed labels, use correlation * sqrt(N) as score
                    # High correlation = model predicts direction + magnitude correctly
                    # sqrt(N) rewards models that trade enough (not too selective)
                    n = len(trades_preds)
                    score = corr * (n ** 0.5)
                    scores.append(score)

                if not scores:
                    return -1.0

                return float(np.mean(scores))

            study = optuna.create_study(direction='maximize')
            # ENHANCED: 100 trials with 30min timeout (was 20 trials, 10min)
            study.optimize(objective, n_trials=100, timeout=1800, show_progress_bar=False)

            self.best_params = study.best_params
            self._save_best_params()
            self.last_tune = datetime.datetime.now()
            logging.info(f"🧠 Best Params (100 trials): {self.best_params}")

    def train(self, tickers, poly, db=None, progress_callback=None, core_tickers=None):
        """Train model on historical data + real trade outcomes if available"""
        with MODEL_LOCK:
            logging.info("🧠 AI Training (15-min bars, BRACKET LABELS, per-ticker split)...")
            all_train = []
            all_calib = []

            # Market context: fetch SPY 15-min + VIX daily ONCE for all tickers
            try:
                _spy_df = self._fetch_15min_data('SPY', poly)
                if len(_spy_df) > 20:
                    _spy_df['SPY_ROC_5'] = _spy_df['Close'].pct_change(5)
                    _spy_df['SPY_ROC_20'] = _spy_df['Close'].pct_change(20)
                    spy_ctx = _spy_df[['SPY_ROC_5', 'SPY_ROC_20']].copy()
                else:
                    spy_ctx = None
            except Exception:
                spy_ctx = None

            try:
                _vix_hist = yf.Ticker('^VIX').history(period='6mo')
                if not _vix_hist.empty:
                    _vix_hist.index = _vix_hist.index.tz_convert('America/New_York') if _vix_hist.index.tzinfo else _vix_hist.index
                    _vix_by_date = _vix_hist.groupby(_vix_hist.index.date).last()
                    vix_level_map = _vix_by_date['Close'].to_dict()
                    vix_roc_map = _vix_by_date['Close'].pct_change(5).to_dict()
                else:
                    vix_level_map, vix_roc_map = {}, {}
            except Exception:
                vix_level_map, vix_roc_map = {}, {}

            total_tickers = len(tickers)

            def process(t_idx, t):
                try:
                    if progress_callback:
                         progress_callback(f"Training {t} ({t_idx+1}/{total_tickers})")

                    # FIX: Use 15-min bars for training
                    # CORE tickers get 4 years, dynamic universe gets 2 years
                    fetch_days = 1460 if (core_tickers and t in core_tickers) else 730
                    df = self._fetch_15min_data(t, poly, days=fetch_days)
                    if df.empty or len(df) < 200:
                        return None, None

                    # ENHANCED FEATURES: Original + 10 new features
                    # Original features
                    df['RSI'] = df.ta.rsi(14)
                    df['ADX'] = df.ta.adx(14)['ADX_14']
                    df['ATR'] = df.ta.atr(14)
                    df['ATR_Pct'] = df['ATR'] / df['Close']
                    v = df['Volume'].rolling(20)
                    df['Vol_Rel'] = df['Volume'] / v.mean()
                    df['Kalman'] = get_kalman_filter(df['Close'].values)
                    df['Kalman_Dist'] = (df['Close'] - df['Kalman']) / df['Close']
                    # Hurst: compute every 10 bars and forward-fill (perf optimization)
                    df['Hurst'] = np.nan
                    _close_vals = df['Close'].values
                    for _hi in range(50, len(df), 10):
                        _window = _close_vals[_hi - 50:_hi]
                        df.iloc[_hi, df.columns.get_loc('Hurst')] = get_hurst(_window)
                    df['Hurst'] = df['Hurst'].ffill().fillna(0.5)

                    # NEW: Bollinger Bands features
                    bb = df.ta.bbands(20, 2)
                    if bb is not None and len(bb.columns) >= 3:
                        df['BB_Upper'] = bb.iloc[:, 0]
                        df['BB_Mid'] = bb.iloc[:, 1]
                        df['BB_Lower'] = bb.iloc[:, 2]
                        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
                        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
                    else:
                        df['BB_Width'] = 0.02
                        df['BB_Position'] = 0.5

                    # NEW: VWAP feature
                    try:
                        df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
                        df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['Close']
                    except:
                        df['VWAP_Dist'] = 0.0

                    # NEW: Microstructure features
                    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
                    df['Money_Flow'] = (df['Close'] * df['Volume']).rolling(10).sum()
                    df['Money_Flow'] = df['Money_Flow'] / df['Money_Flow'].rolling(50).mean()  # Normalize

                    # NEW: Momentum features
                    df['ROC_5'] = df['Close'].pct_change(5)
                    df['ROC_20'] = df['Close'].pct_change(20)

                    # NEW: Volume features
                    df['Vol_Surge'] = df['Volume'] / df['Volume'].rolling(5).mean()

                    # NEW: Regime features
                    df['Volatility_Rank'] = df['ATR_Pct'].rolling(100).apply(
                        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5, raw=False
                    )
                    _ret = df['Close'].pct_change()
                    df['Trend_Consistency'] = _ret.rolling(20).apply(
                        lambda s: (s > 0).mean(), raw=False
                    )

                    # ENHANCED: Time features (intraday patterns)
                    df['Hour'] = df.index.hour
                    df['Day_of_Week'] = df.index.dayofweek  # 0=Monday, 4=Friday

                    # ENHANCED: Sentiment features (will be 0 during training, populated during live scanning)
                    df['sa_news_count_3d'] = 0  # Placeholder for training
                    df['sa_sentiment_score'] = 0  # Populated live from SA API

                    # Fundamental features (populated live from FMP, default 0 for training)
                    df['earnings_surprise'] = 0.0
                    df['revenue_growth_yoy'] = 0.0
                    df['pe_ratio'] = 20.0

                    # Dynamic news weighting (populated live, default 0 for training)
                    df['news_impact_weight'] = 0.0

                    # Market context: SPY momentum + VIX regime
                    if spy_ctx is not None:
                        df = df.join(spy_ctx, how='left')
                        df['SPY_ROC_5'] = df['SPY_ROC_5'].ffill().fillna(0.0)
                        df['SPY_ROC_20'] = df['SPY_ROC_20'].ffill().fillna(0.0)
                    else:
                        df['SPY_ROC_5'] = 0.0
                        df['SPY_ROC_20'] = 0.0

                    if vix_level_map:
                        df['VIX_Level'] = [vix_level_map.get(d, 20.0) for d in df.index.date]
                        df['VIX_ROC'] = [vix_roc_map.get(d, 0.0) for d in df.index.date]
                        df['VIX_Level'] = df['VIX_Level'].fillna(20.0)
                        df['VIX_ROC'] = df['VIX_ROC'].fillna(0.0)
                    else:
                        df['VIX_Level'] = 20.0
                        df['VIX_ROC'] = 0.0

                    # Cross-sectional ranks PROXY (Self-Relative Percentiles for Training)
                    # "Is this stock exhibiting top-tier behavior relative to its recent history?"
                    df['momentum_rank'] = df['ROC_20'].rolling(100).rank(pct=True).fillna(0.5)

                    # Volume Rank: Relative Volume Strength
                    df['volume_rank'] = (df['Volume'] / df['Volume'].rolling(20).mean()).rolling(100).rank(pct=True).fillna(0.5)

                    # Value Rank: Low RSI = 'Value' (oversold) in this context, or PE if available
                    # We use RSI Rank as a proxy for "Value/Mean-Reversion Potential"
                    df['value_rank'] = (100 - df['RSI']).rolling(100).rank(pct=True).fillna(0.5)

                    # Composite = Average of Momentum and Volume
                    df['composite_rank'] = ((df['momentum_rank'] + df['volume_rank']) / 2).fillna(0.5)

                    # ==============================================================================
                    # INSTITUTIONAL MICROSTRUCTURE FEATURES (HIGH-CONFIDENCE TRADING)
                    # ==============================================================================

                    # 1. VPIN (Order Flow Toxicity) - Filter out toxic market conditions
                    try:
                        df['VPIN'] = calculate_vpin(df, window=50)
                        logging.debug(f"   ✅ VPIN calculated (mean={df['VPIN'].mean():.3f})")
                    except Exception as e:
                        logging.debug(f"   VPIN calc failed: {e}")
                        df['VPIN'] = 0.0

                    # 2. Enhanced VWAP Features (Institutional Mean Reversion Signals)
                    try:
                        vwap_features = calculate_enhanced_vwap_features(df)
                        df['VWAP_ZScore'] = vwap_features['VWAP_ZScore'].fillna(0.0)
                        df['VWAP_Slope'] = vwap_features['VWAP_Slope'].fillna(0.0)
                        df['VWAP_Volume_Ratio'] = vwap_features['VWAP_Volume_Ratio'].fillna(1.0)
                        logging.debug(f"   ✅ Enhanced VWAP features calculated")
                    except Exception as e:
                        logging.debug(f"   VWAP features failed: {e}")
                        df['VWAP_ZScore'] = 0.0
                        df['VWAP_Slope'] = 0.0
                        df['VWAP_Volume_Ratio'] = 1.0

                    # 3. GEX Proxy (Volatility Regime Detection)
                    try:
                        regime_gex, vol_regime_label = calculate_volatility_regime(df)
                        df['Regime_GEX_Proxy'] = regime_gex.fillna(0)
                        # Convert regime label to score: LOW=-1, MEDIUM=0, HIGH=1
                        regime_score_map = {'LOW': -1, 'MEDIUM': 0, 'HIGH': 1}
                        df['Volatility_Regime_Score'] = vol_regime_label.map(regime_score_map).fillna(0)
                        logging.debug(f"   ✅ GEX Proxy calculated")
                    except Exception as e:
                        logging.debug(f"   GEX Proxy failed: {e}")
                        df['Regime_GEX_Proxy'] = 0
                        df['Volatility_Regime_Score'] = 0

                    # 4. Amihud Illiquidity (Slippage Risk)
                    try:
                        df['Amihud_Illiquidity'] = calculate_amihud_illiquidity(df, window=20)
                        logging.debug(f"   ✅ Amihud Illiquidity calculated (mean={df['Amihud_Illiquidity'].mean():.3f})")
                    except Exception as e:
                        logging.debug(f"   Amihud calc failed: {e}")
                        df['Amihud_Illiquidity'] = 0.5

                    # 5. 🔥 CRITICAL: Real Relative Strength (RRS) vs SPY
                    try:
                        # Fetch SPY data for beta calculation
                        spy_df_for_rrs = None
                        if spy_ctx is not None and 'SPY_Close' in spy_ctx.columns:
                            spy_df_for_rrs = pd.DataFrame({'Close': spy_ctx['SPY_Close']})

                        rrs = calculate_real_relative_strength(df, spy_df_for_rrs)
                        df['RRS_Cumulative'] = rrs.fillna(0.0)

                        # Also store Beta for analysis
                        if spy_df_for_rrs is not None:
                            stock_returns = df['Close'].pct_change()
                            spy_returns = spy_df_for_rrs['Close'].pct_change().reindex(df.index, method='ffill')
                            covariance = stock_returns.rolling(50).cov(spy_returns)
                            spy_variance = spy_returns.rolling(50).var()
                            df['Beta_To_SPY'] = (covariance / (spy_variance + 1e-9)).fillna(1.0).clip(-3, 3)
                        else:
                            df['Beta_To_SPY'] = 1.0

                        logging.debug(f"   ✅ RRS calculated (mean={df['RRS_Cumulative'].mean():.4f})")
                    except Exception as e:
                        logging.debug(f"   RRS calc failed: {e}")
                        df['RRS_Cumulative'] = 0.0
                        df['Beta_To_SPY'] = 1.0

                    # 6. 🎯 HIGH WIN RATE: Liquidity Sweep Detection
                    try:
                        df['Liquidity_Sweep'] = calculate_liquidity_sweep(df, lookback=16)
                        sweep_count = (df['Liquidity_Sweep'] != 0).sum()
                        logging.debug(f"   ✅ Liquidity Sweeps detected: {sweep_count}")
                    except Exception as e:
                        logging.debug(f"   Liquidity Sweep calc failed: {e}")
                        df['Liquidity_Sweep'] = 0

                    # ==============================================================================

                    # FIX: Attach daily ATR for consistent train/live stop geometry
                    # Uses Polygon daily bars via vectorized join (fast)
                    df = attach_daily_atr_to_15m(df, t, poly=poly)

                    df = df.dropna()

                    # BRACKET-AWARE LABELING: Simulate whether TP or SL would hit first
                    # ENHANCED: Use regression mode to predict R-value directly
                    # FIX: Use ATR_D (daily ATR) for labels to match live trading
                    df['Target'] = compute_bracket_labels(
                        df,
                        sl_mult=1.5,  # Same as SETTINGS SL
                        tp_mult=3.0,  # Same as SETTINGS TP
                        max_bars=12,  # FIX: ~3 hours at 15-min bars (matches time-stop)
                        atr_col='ATR_D' if 'ATR_D' in df.columns else 'ATR',  # Use daily ATR if available
                        mode='regression'  # ENHANCED: Predict R-values instead of classes
                    )

                    if len(df) < 100:
                        return None, None

                    # FIX: True per-ticker percentile split (80/20) + bar embargo
                    # 5 trading days = ~5*26 = 130 15-min bars
                    embargo_bars = 130
                    split_i = int(len(df) * 0.8)

                    # 80% Train, 20% Calibration
                    train_df = df.iloc[:max(0, split_i - embargo_bars)][self.cols + ['Target']]
                    calib_df = df.iloc[min(len(df), split_i + embargo_bars):][self.cols + ['Target']]

                    return train_df, calib_df
                except Exception as e:
                    logging.debug(f"Train process {t}: {e}")
                    return None, None

            # Optimized: Use CPU_WORKERS for Training (CPU-bound)
            if not tickers: return {}

        # Optimized: Use high concurrency for I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
            future_to_ticker = {executor.submit(process, i, t): t for i, t in enumerate(tickers)}
            for future in concurrent.futures.as_completed(future_to_ticker):
                    train_df, calib_df = future.result()
                    if train_df is not None and len(train_df) > 0:
                        all_train.append(train_df)
                    if calib_df is not None and len(calib_df) > 0:
                        all_calib.append(calib_df)

            # NEW: Blend real trade outcomes if available (Learning from Mistakes)
            if db:
                try:
                    real_outcomes = db.get_trade_outcomes(days=90)
                    if len(real_outcomes) >= 50:  # Only start blending after enough data
                        logging.info(f"🧠 Blending {len(real_outcomes)} real trade outcomes into training")
                        real_df = pd.DataFrame(real_outcomes)

                        # Ensure columns exist
                        missing_cols = [c for c in self.cols if c not in real_df.columns]
                        if not missing_cols:
                            real_df['Target'] = real_df['pnl_r']  # Use R-multiple directly (regression target)

                            # Weight recent trades higher (exponential decay)
                            # Recent trades get ~2x-3x weight compared to old ones
                            # This isn't directly supported by standard XGB fit unless we pass sample_weight
                            # For simplicity validation, we'll just duplicate recent mistake rows
                            # better: just append them.

                            real_train = real_df[self.cols + ['Target']]
                            # Append to CALIBRATION set (all_calib) to ground the model in reality
                            all_calib.append(real_train)
                except Exception as e:
                    logging.warning(f"Failed to blend real outcomes: {e}")

            if not all_train:
                logging.error("No training data")
                return

            train = pd.concat(all_train)
            # FIX: Use all_calib for the calibration set (previously undefined all_test)
            test = pd.concat(all_calib) if all_calib else train.sample(frac=0.2)

            # v15: FEATURE SELECTION (IC Tracking + Stability Check)
            # Pass if: |aggregate_ic| > 0.02 AND stability > 0.3
            # Stability = 1 - std(IC_k) / mean(|IC_k|) across quarterly windows
            valid_cols = []
            ic_only_cols = []  # fallback: IC-only (no stability requirement)

            # Split training data into 4 quarterly windows for stability check
            n_rows = len(train)
            window_size = n_rows // 4
            use_stability = window_size >= 50  # need enough rows per window

            for col in self.cols:
                try:
                    agg_ic = train[col].corr(train['Target'], method='spearman')
                    if np.isnan(agg_ic):
                        continue

                    passes_ic = abs(agg_ic) > 0.02
                    if passes_ic:
                        ic_only_cols.append(col)

                    if passes_ic and use_stability:
                        # Compute IC in each quarterly window
                        window_ics = []
                        for k in range(4):
                            start = k * window_size
                            end = start + window_size if k < 3 else n_rows
                            window = train.iloc[start:end]
                            w_ic = window[col].corr(window['Target'], method='spearman')
                            if not np.isnan(w_ic):
                                window_ics.append(w_ic)

                        if len(window_ics) >= 3:
                            ic_std = np.std(window_ics)
                            ic_mean_abs = np.mean(np.abs(window_ics))
                            stability = 1.0 - (ic_std / ic_mean_abs) if ic_mean_abs > 1e-9 else 0.0
                            if stability > 0.3:
                                valid_cols.append(col)
                                logging.debug(f"✅ {col}: IC={agg_ic:.3f}, stability={stability:.2f}")
                            else:
                                logging.debug(f"🗑️ Dropping {col} (IC={agg_ic:.3f}, stability={stability:.2f} < 0.3)")
                        else:
                            # Not enough valid windows, accept on IC alone
                            valid_cols.append(col)
                    elif passes_ic and not use_stability:
                        # Too few rows for stability check, accept on IC alone
                        valid_cols.append(col)
                    else:
                        logging.debug(f"🗑️ Dropping {col} (IC={agg_ic:.3f})")
                except Exception:
                    pass

            # Fallback chain: stability → IC-only → all defaults
            if len(valid_cols) < 6 and ic_only_cols:
                logging.warning(f"⚠️ Stability pruning left {len(valid_cols)} features, falling back to IC-only ({len(ic_only_cols)} features)")
                valid_cols = ic_only_cols
            if not valid_cols:
                logging.warning("⚠️ All features dropped by IC check! Using all defaults.")
                valid_cols = self.cols

            self.active_features = valid_cols  # v15: Persist for prediction
            logging.info(f"✨ Active Features: {valid_cols}")

            X_train = train[valid_cols]
            y_train = train['Target']
            X_test = test[valid_cols]
            y_test = test['Target']

            # ENHANCED: Optuna Tuning WEEKLY (not daily) to prevent overfitting
            # Tune if it's been more than 7 days since last tune
            if (datetime.datetime.now() - self.last_tune).total_seconds() > 604800:
                logging.info("🧠 Starting weekly Optuna tuning...")
                self.tune(X_train, y_train)

            # Use best params if available, otherwise default
            xgb_params = self.best_params if self.best_params else {
                'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05
            }

            # ENHANCED: Switch to XGBRegressor with CUSTOM ASYMMETRIC LOSS for high win rate
            # FIX: Proper GPU config with gpu_hist and gpu_predictor
            try:
                # Optimized: Force GPU for Colab Pro+ (A100/H100)
                # XGBoost 2.0+ uses device="cuda"
                # CRITICAL: Use custom objective for institutional-grade optimization
                self.model = XGBRegressor(
                    **xgb_params,
                    device="cuda",
                    n_jobs=CPU_WORKERS,
                    objective=profit_factor_objective,  # PROFIT FACTOR OPTIMIZATION
                    verbosity=0
                )
            except Exception as e:
                try:
                    # Legacy Fallback
                    self.model = XGBRegressor(
                        **xgb_params,
                        tree_method='gpu_hist',
                        predictor='gpu_predictor',
                        n_jobs=CPU_WORKERS,
                        objective=profit_factor_objective,  # PROFIT FACTOR OPTIMIZATION
                        verbosity=0
                    )
                except Exception as e2:
                    logging.warning(f"⚠️ GPU init failed ({e2}). Falling back to CPU.")
                    self.model = XGBRegressor(
                        **xgb_params,
                        tree_method='hist',
                        n_jobs=CPU_WORKERS,
                        objective=profit_factor_objective,  # PROFIT FACTOR OPTIMIZATION
                        verbosity=0
                    )

            self.model.fit(X_train, y_train)

            # Feature importance pruning: log and auto-prune weak features
            try:
                importances = self.model.feature_importances_
                feat_imp = sorted(zip(valid_cols, importances), key=lambda x: x[1], reverse=True)
                logging.info("📊 Feature Importances (top 10):")
                for fname, fimp in feat_imp[:10]:
                    logging.info(f"   {fname}: {fimp:.4f}")

                # Auto-prune features with near-zero importance AND low IC
                pruned = []
                for fname, fimp in feat_imp:
                    if fimp < 0.005:  # Near-zero tree importance
                        try:
                            ic = abs(train[fname].corr(train['Target'], method='spearman'))
                        except:
                            ic = 0
                        if ic < 0.02:
                            logging.info(f"   🗑️ Pruning {fname} (imp={fimp:.4f}, IC={ic:.3f})")
                            continue
                    pruned.append(fname)

                if len(pruned) >= 6:  # Keep minimum viable feature set
                    self.active_features = pruned
                    logging.info(f"✨ Post-prune features ({len(pruned)}): {pruned}")
                    # Re-fit with pruned features for cleaner model
                    X_train = train[pruned]
                    X_test = test[pruned]
                    self.model.fit(X_train, y_train)
            except Exception as e:
                logging.debug(f"Feature pruning failed: {e}")

            # ── MODEL STACKING: LightGBM + Ridge + Meta-learner ──
            try:
                # Base learner 2: LightGBM (different inductive bias from XGB)
                if HAS_LGB:
                    self.lgb_model = lgb.LGBMRegressor(
                        n_estimators=xgb_params.get('n_estimators', 100),
                        max_depth=xgb_params.get('max_depth', 4),
                        learning_rate=xgb_params.get('learning_rate', 0.05),
                        verbosity=-1, n_jobs=CPU_WORKERS
                    )
                    self.lgb_model.fit(X_train, y_train)
                    logging.info(f"   LightGBM trained (R²={self.lgb_model.score(X_test, y_test):.3f})")

                # Base learner 3: Ridge regression (linear, catches what trees miss)
                self.ridge_model = Ridge(alpha=1.0)
                self.ridge_model.fit(X_train, y_train)
                logging.info(f"   Ridge trained (R²={self.ridge_model.score(X_test, y_test):.3f})")

                # Meta-learner: blend base predictions via Ridge (blending on test set)
                xgb_test_pred = self.model.predict(X_test)
                lgb_test_pred = self.lgb_model.predict(X_test) if self.lgb_model else xgb_test_pred
                ridge_test_pred = self.ridge_model.predict(X_test)
                meta_X = np.column_stack([xgb_test_pred, lgb_test_pred, ridge_test_pred])
                self.stack_meta = Ridge(alpha=0.5)
                self.stack_meta.fit(meta_X, y_test)
                meta_r2 = self.stack_meta.score(meta_X, y_test)
                logging.info(f"✅ Stacking meta-learner trained (meta R²={meta_r2:.3f})")
            except Exception as e:
                logging.warning(f"Model stacking failed (XGB-only fallback): {e}")
                self.lgb_model = None
                self.ridge_model = None
                self.stack_meta = None

            # Train quantile models for prediction intervals (q10, q90)
            try:
                q10_params = dict(xgb_params)
                # FIX: Use 'reg:quantileerror' for compatibility with installed XGBoost version
                q10_params['objective'] = 'reg:quantileerror'
                q10_params['quantile_alpha'] = 0.10
                q10_params.pop('eval_metric', None)
                self.model_q10 = XGBRegressor(**q10_params, device="cuda", verbosity=0)
                self.model_q10.fit(X_train, y_train)

                q90_params = dict(xgb_params)
                q90_params['objective'] = 'reg:quantileerror'
                q90_params['quantile_alpha'] = 0.90
                q90_params.pop('eval_metric', None)
                self.model_q90 = XGBRegressor(**q90_params, device="cuda", verbosity=0)
                self.model_q90.fit(X_train, y_train)

                logging.info("✅ Quantile models (q10, q90) trained on GPU")
            except Exception as e:
                logging.warning(f"Quantile model GPU training failed: {e}. Falling back to CPU with reg:quantileerror.")
                # CPU Fallback code...
                q10_params = dict(xgb_params)
                q10_params['objective'] = 'reg:quantileerror'
                q10_params['quantile_alpha'] = 0.10
                q10_params.pop('eval_metric', None)
                self.model_q10 = XGBRegressor(**q10_params, tree_method='hist', verbosity=0)
                self.model_q10.fit(X_train, y_train)

                q90_params = dict(xgb_params)
                q90_params['objective'] = 'reg:quantileerror'
                q90_params['quantile_alpha'] = 0.90
                q90_params.pop('eval_metric', None)
                self.model_q90 = XGBRegressor(**q90_params, tree_method='hist', verbosity=0)
                self.model_q90.fit(X_train, y_train)

            # ENHANCED: Train Ensemble Specialist Models
            if self.use_ensemble and len(train) > 500:
                logging.info("🎯 Training Ensemble Specialists...")
                try:
                    # 1. Trend Follower (ADX > 25)
                    trend_mask_train = train['ADX'] > 25
                    trend_mask_test = test['ADX'] > 25
                    if trend_mask_train.sum() > 100:
                        logging.info(f"   Trend Specialist: {trend_mask_train.sum()} samples")
                        self.trend_model = XGBRegressor(**xgb_params, tree_method='hist', eval_metric='mae', verbosity=0)
                        self.trend_model.fit(train.loc[trend_mask_train, valid_cols], train.loc[trend_mask_train, 'Target'])

                    # 2. Mean Reversion (Hurst < 0.4)
                    mr_mask_train = train['Hurst'] < 0.4
                    mr_mask_test = test['Hurst'] < 0.4
                    if mr_mask_train.sum() > 100:
                        logging.info(f"   Mean-Rev Specialist: {mr_mask_train.sum()} samples")
                        self.mean_reversion_model = XGBRegressor(**xgb_params, tree_method='hist', eval_metric='mae', verbosity=0)
                        self.mean_reversion_model.fit(train.loc[mr_mask_train, valid_cols], train.loc[mr_mask_train, 'Target'])

                    # 3. Volatility Breakout (ATR_Pct in top 30%)
                    vol_threshold = train['ATR_Pct'].quantile(0.70)
                    vol_mask_train = train['ATR_Pct'] > vol_threshold
                    vol_mask_test = test['ATR_Pct'] > vol_threshold
                    if vol_mask_train.sum() > 100:
                        logging.info(f"   Volatility Specialist: {vol_mask_train.sum()} samples")
                        self.volatility_model = XGBRegressor(**xgb_params, tree_method='hist', eval_metric='mae', verbosity=0)
                        self.volatility_model.fit(train.loc[vol_mask_train, valid_cols], train.loc[vol_mask_train, 'Target'])

                    # 4. Calculate meta-weights based on test performance
                    scores = {}
                    if self.trend_model and trend_mask_test.sum() > 20:
                        scores['trend'] = self.trend_model.score(test.loc[trend_mask_test, valid_cols], test.loc[trend_mask_test, 'Target'])
                    if self.mean_reversion_model and mr_mask_test.sum() > 20:
                        scores['mean_rev'] = self.mean_reversion_model.score(test.loc[mr_mask_test, valid_cols], test.loc[mr_mask_test, 'Target'])
                    if self.volatility_model and vol_mask_test.sum() > 20:
                        scores['vol'] = self.volatility_model.score(test.loc[vol_mask_test, valid_cols], test.loc[vol_mask_test, 'Target'])

                    # Normalize scores to weights
                    if scores:
                        total = sum(max(0, s) for s in scores.values())
                        if total > 0:
                            self.meta_weights = {k: max(0, v) / total for k, v in scores.items()}
                        logging.info(f"   Meta-weights: {self.meta_weights}")

                except Exception as e:
                    logging.warning(f"Ensemble training failed: {e}")

            # ENHANCED: For regression, calibration is replaced by R² scoring
            # No calibrators needed - model predicts R-values directly

            train_r2 = self.model.score(X_train, y_train)  # R² score for regression
            test_r2 = self.model.score(X_test, y_test)

            logging.info(f"✅ AI: {len(train)} train, {len(test)} test")
            logging.info(f"   Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")

            # ENHANCED: Overfitting detection for regression (R² gap)
            if train_r2 - test_r2 > 0.15:  # R² gap indicates overfitting
                logging.warning("⚠️ Overfitting detected! Applying 10% confidence dampener.")
                self.confidence_dampener = 0.90  # Reduce predictions by 10%
            else:
                self.confidence_dampener = 1.0

            self.last_train = datetime.datetime.now()

            # ── SHAP Feature Importance Monitoring ──
            try:
                if HAS_SHAP and len(X_test) > 50:
                    sample = X_test.sample(min(200, len(X_test)), random_state=42)
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(sample)
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    self.shap_importances = dict(zip(self.active_features, mean_abs_shap))

                    # Log top features by SHAP
                    sorted_shap = sorted(self.shap_importances.items(), key=lambda x: x[1], reverse=True)
                    logging.info("📊 SHAP Feature Importance (top 10):")
                    for fname, simp in sorted_shap[:10]:
                        logging.info(f"   {fname}: {simp:.4f}")

                    # Drift detection: compare to previous SHAP snapshot
                    self.shap_history.append((datetime.datetime.now(), dict(self.shap_importances)))
                    if len(self.shap_history) >= 2:
                        prev = self.shap_history[-2][1]
                        curr = self.shap_importances
                        common = set(prev.keys()) & set(curr.keys())
                        if common:
                            prev_vec = np.array([prev[k] for k in common])
                            curr_vec = np.array([curr[k] for k in common])
                            # Normalize
                            if prev_vec.sum() > 0:
                                prev_vec = prev_vec / prev_vec.sum()
                            if curr_vec.sum() > 0:
                                curr_vec = curr_vec / curr_vec.sum()
                            drift = float(np.sqrt(np.mean((prev_vec - curr_vec) ** 2)))
                            if drift > 0.15:
                                logging.warning(f"⚠️ SHAP DRIFT detected (RMSE={drift:.3f}). "
                                                "Feature importance shifted significantly since last training.")
                            else:
                                logging.info(f"   SHAP drift: {drift:.3f} (stable)")
                    # Keep only last 10 snapshots
                    if len(self.shap_history) > 10:
                        self.shap_history = self.shap_history[-10:]
            except Exception as e_shap:
                logging.debug(f"SHAP monitoring skipped: {e_shap}")

    def _fetch_15min_data(self, ticker, poly, days=1460):
        """
        Fetch 15-min bars directly from Polygon (no resampling).
        CORE tickers: 4 years (1460 days). Dynamic universe: 2 years (730 days).
        """
        try:
            df_15m = poly.fetch_data(ticker, days=days, mult=15)
            return df_15m if df_15m is not None else pd.DataFrame()
        except Exception as e:
            logging.warning(f"_fetch_15min_data({ticker}) failed: {e}")
            return pd.DataFrame()

    def predict(self, row, return_uncertainty=False):
        """
        ENHANCED: Predict expected R-value using ensemble of specialist models.
        Returns a single float representing expected R-multiple (or tuple if return_uncertainty=True).
        Positive = profitable, negative = unprofitable
        """
        with MODEL_LOCK:
            if not self.model:
                logging.warning("⚠️ Model not fitted yet - returning neutral R-value")
                return (0.0, 1.0) if return_uncertainty else 0.0
            try:
                # 1. Structure Features
                features = pd.DataFrame([row])[self.active_features]

                # 2. Ensemble Prediction (if enabled and specialists available)
                if self.use_ensemble:
                    predictions = []
                    weights = []

                    # Check regime and get specialist predictions
                    adx = row.get('ADX', 0)
                    hurst = row.get('Hurst', 0.5)
                    atr_pct = row.get('ATR_Pct', 0)

                    # Trend Specialist (ADX > 25)
                    if self.trend_model and adx > 25:
                        pred = float(self.trend_model.predict(features)[0])
                        predictions.append(pred)
                        weights.append(self.meta_weights.get('trend', 0.33))

                    # Mean Reversion Specialist (Hurst < 0.4)
                    if self.mean_reversion_model and hurst < 0.4:
                        pred = float(self.mean_reversion_model.predict(features)[0])
                        predictions.append(pred)
                        weights.append(self.meta_weights.get('mean_rev', 0.33))

                    # Volatility Specialist (High ATR)
                    if self.volatility_model and atr_pct > 0.02:  # Top 30% threshold
                        pred = float(self.volatility_model.predict(features)[0])
                        predictions.append(pred)
                        weights.append(self.meta_weights.get('vol', 0.33))

                    # Blend predictions if any specialists apply
                    if predictions:
                        total_weight = sum(weights)
                        predicted_r = sum(p * w for p, w in zip(predictions, weights)) / total_weight

                        # ENHANCED: Uncertainty from ensemble disagreement
                        if len(predictions) > 1:
                            uncertainty = np.std(predictions)  # High std = high uncertainty
                        else:
                            uncertainty = 0.5  # Single specialist = moderate uncertainty
                    else:
                        # Fallback: use stacked prediction if available
                        xgb_pred = float(self.model.predict(features)[0])
                        if self.stack_meta and self.lgb_model and self.ridge_model:
                            lgb_pred = float(self.lgb_model.predict(features)[0])
                            ridge_pred = float(self.ridge_model.predict(features)[0])
                            meta_X = np.array([[xgb_pred, lgb_pred, ridge_pred]])
                            predicted_r = float(self.stack_meta.predict(meta_X)[0])
                            uncertainty = float(np.std([xgb_pred, lgb_pred, ridge_pred]))
                        else:
                            predicted_r = xgb_pred
                            uncertainty = 0.75
                else:
                    # Stacked model prediction (XGB + LGB + Ridge → meta)
                    xgb_pred = float(self.model.predict(features)[0])
                    if self.stack_meta and self.lgb_model and self.ridge_model:
                        lgb_pred = float(self.lgb_model.predict(features)[0])
                        ridge_pred = float(self.ridge_model.predict(features)[0])
                        meta_X = np.array([[xgb_pred, lgb_pred, ridge_pred]])
                        predicted_r = float(self.stack_meta.predict(meta_X)[0])
                        uncertainty = float(np.std([xgb_pred, lgb_pred, ridge_pred]))
                    else:
                        predicted_r = xgb_pred
                        uncertainty = 0.5

                # 3. Apply dampener (reduce predictions if overfitting detected)
                damp = getattr(self, "confidence_dampener", 1.0)
                if damp < 1.0:
                    predicted_r = predicted_r * damp

                if return_uncertainty:
                    return predicted_r, uncertainty
                else:
                    return predicted_r

            except Exception as e:
                logging.error(f"Predict failed: {e}")
                return (0.0, 1.0) if return_uncertainty else 0.0

    def predict_with_intervals(self, row):
        """
        Predict R-value with quantile-based prediction intervals.
        Returns: (predicted_r, p_win, p_hold, p_loss, uncertainty)
        """
        predicted_r, unc = self.predict(row, return_uncertainty=True)

        # Default: fallback linear mapping
        raw_conf = min(0.90, max(0.30, 0.50 + abs(predicted_r) * 0.15))
        p_win = raw_conf if predicted_r > 0 else (1.0 - raw_conf)
        p_loss = (1.0 - raw_conf) if predicted_r > 0 else raw_conf
        p_hold = 0.05
        uncertainty = unc

        # Override with quantile models if available
        if self.model_q10 and self.model_q90:
            try:
                with MODEL_LOCK:
                    features = pd.DataFrame([row])[self.active_features]
                    q10 = float(self.model_q10.predict(features)[0])
                    q90 = float(self.model_q90.predict(features)[0])

                interval_width = q90 - q10
                uncertainty = max(0.01, interval_width)

                if interval_width > 1e-6 and q90 > q10:
                    frac_above = np.clip((q90 - 0.0) / interval_width, 0, 1)
                    frac_below = 1.0 - frac_above

                    if predicted_r > 0:
                        p_win = np.clip(frac_above, 0.30, 0.90)
                        p_loss = np.clip(frac_below, 0.05, 0.60)
                    else:
                        p_win = np.clip(frac_below, 0.30, 0.90)
                        p_loss = np.clip(frac_above, 0.05, 0.60)

                    p_hold = max(0.05, 1.0 - p_win - p_loss)

                    # Renormalize
                    total = p_win + p_hold + p_loss
                    p_win /= total
                    p_hold /= total
                    p_loss /= total

            except Exception as e:
                logging.debug(f"Quantile prediction failed, using fallback: {e}")

        return predicted_r, float(p_win), float(p_hold), float(p_loss), float(uncertainty)

# ==============================================================================
# WALK-FORWARD BACKTEST ENGINE
# ==============================================================================
class BracketBacktest:
    """
    Vectorized walk-forward backtest with realistic bracket-order simulation.
    Replays the model's predictions on historical data with slippage + costs.
    """
    def __init__(self, ai, poly, sl_mult=1.5, tp_mult=3.0, max_bars=12,
                 cost_bps=5.0, risk_per_trade=0.01):
        self.ai = ai
        self.poly = poly
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.max_bars = max_bars
        self.cost_bps = cost_bps  # Round-trip cost in basis points
        self.risk_per_trade = risk_per_trade

    def run(self, tickers, test_days=120, initial_equity=100000.0):
        """
        Run vectorized backtest on last `test_days` of data.
        Returns dict with equity_curve, trades, metrics.
        """
        logging.info(f"📊 Running backtest: {len(tickers)} tickers, {test_days} days...")
        equity = initial_equity
        equity_curve = [equity]
        all_trades = []
        daily_returns = []
        prev_equity = equity

        for t in tickers:
            try:
                df = self.poly.fetch_data(t, days=test_days + 60, mult=15)
                if df is None or len(df) < 200:
                    continue

                # Compute features (same pipeline as training)
                df['RSI'] = df.ta.rsi(14)
                df['ADX'] = df.ta.adx(14)['ADX_14']
                df['ATR'] = df.ta.atr(14)
                df['ATR_Pct'] = df['ATR'] / df['Close']
                v = df['Volume'].rolling(20)
                df['Vol_Rel'] = df['Volume'] / v.mean()
                df['Kalman'] = get_kalman_filter(df['Close'].values)
                df['Kalman_Dist'] = (df['Close'] - df['Kalman']) / df['Close']
                df['Hurst'] = 0.5
                _cv = df['Close'].values
                for _hi in range(50, len(df), 10):
                    df.iloc[_hi, df.columns.get_loc('Hurst')] = get_hurst(_cv[_hi-50:_hi])
                df['Hurst'] = df['Hurst'].ffill().fillna(0.5)

                bb = df.ta.bbands(20, 2)
                if bb is not None and len(bb.columns) >= 3:
                    df['BB_Width'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / df['Close']
                    df['BB_Position'] = (df['Close'] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2] + 1e-10)
                else:
                    df['BB_Width'] = 0.02; df['BB_Position'] = 0.5

                try:
                    df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
                    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['Close']
                except Exception:
                    df['VWAP_Dist'] = 0.0

                df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
                df['Money_Flow'] = (df['Close'] * df['Volume']).rolling(10).sum()
                df['Money_Flow'] = df['Money_Flow'] / df['Money_Flow'].rolling(50).mean()
                df['ROC_5'] = df['Close'].pct_change(5)
                df['ROC_20'] = df['Close'].pct_change(20)
                df['Vol_Surge'] = df['Volume'] / df['Volume'].rolling(5).mean()
                df['Volatility_Rank'] = df['ATR_Pct'].rolling(100).apply(
                    lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5, raw=False)
                _ret = df['Close'].pct_change()
                df['Trend_Consistency'] = _ret.rolling(20).apply(lambda s: (s > 0).mean(), raw=False)
                df['Hour'] = df.index.hour
                df['Day_of_Week'] = df.index.dayofweek

                # ==============================================================================
                # INSTITUTIONAL MICROSTRUCTURE FEATURES (BACKTEST)
                # ==============================================================================
                try:
                    df['VPIN'] = calculate_vpin(df, window=50)
                except:
                    df['VPIN'] = 0.0

                try:
                    vwap_features = calculate_enhanced_vwap_features(df)
                    df['VWAP_ZScore'] = vwap_features['VWAP_ZScore'].fillna(0.0)
                    df['VWAP_Slope'] = vwap_features['VWAP_Slope'].fillna(0.0)
                    df['VWAP_Volume_Ratio'] = vwap_features['VWAP_Volume_Ratio'].fillna(1.0)
                except:
                    df['VWAP_ZScore'] = 0.0
                    df['VWAP_Slope'] = 0.0
                    df['VWAP_Volume_Ratio'] = 1.0

                try:
                    regime_gex, vol_regime_label = calculate_volatility_regime(df)
                    df['Regime_GEX_Proxy'] = regime_gex.fillna(0)
                    regime_score_map = {'LOW': -1, 'MEDIUM': 0, 'HIGH': 1}
                    df['Volatility_Regime_Score'] = vol_regime_label.map(regime_score_map).fillna(0)
                except:
                    df['Regime_GEX_Proxy'] = 0
                    df['Volatility_Regime_Score'] = 0

                try:
                    df['Amihud_Illiquidity'] = calculate_amihud_illiquidity(df, window=20)
                except:
                    df['Amihud_Illiquidity'] = 0.5

                # RRS and Liquidity Sweep (Critical Alpha Features)
                try:
                    # RRS (no SPY data in backtest, use momentum proxy)
                    df['RRS_Cumulative'] = df['Close'].pct_change(5).rolling(5).sum().fillna(0.0)
                    df['Beta_To_SPY'] = 1.0  # Placeholder for backtest
                except:
                    df['RRS_Cumulative'] = 0.0
                    df['Beta_To_SPY'] = 1.0

                try:
                    df['Liquidity_Sweep'] = calculate_liquidity_sweep(df, lookback=16)
                except:
                    df['Liquidity_Sweep'] = 0
                # ==============================================================================

                # Fill placeholders for features not available in backtest
                for col in self.ai.active_features:
                    if col not in df.columns:
                        df[col] = 0.0

                df = df.dropna(subset=self.ai.active_features)
                if len(df) < 100:
                    continue

                # Walk-forward: only test on last test_days portion
                cutoff = df.index[-1] - pd.Timedelta(days=test_days)
                test_mask = df.index >= cutoff
                test_indices = df.index[test_mask]

                close = df['Close'].values
                high = df['High'].values
                low = df['Low'].values
                atr = df['ATR'].values

                i = 0
                while i < len(test_indices):
                    idx = df.index.get_loc(test_indices[i])
                    if idx + self.max_bars >= len(df):
                        break

                    row = df.iloc[idx].to_dict()
                    predicted_r, uncertainty = self.ai.predict(row, return_uncertainty=True)

                    # ==============================================================================
                    # INSTITUTIONAL FILTERS (MATCH LIVE TRADING LOGIC)
                    # ==============================================================================

                    # 1. Base prediction threshold
                    if abs(predicted_r) < 0.10:
                        i += 1
                        continue

                    # 2. Uncertainty filter
                    if uncertainty > 1.5:
                        i += 1
                        continue

                    # 3. VPIN filter (toxic order flow)
                    if row.get('VPIN', 0.0) > 0.85:
                        i += 1
                        continue

                    # 4. Amihud filter (illiquidity)
                    if row.get('Amihud_Illiquidity', 0.5) > 0.90:
                        i += 1
                        continue

                    # 5. VWAP extreme filter
                    vwap_z = row.get('VWAP_ZScore', 0.0)
                    if abs(vwap_z) > 2.5:
                        # Fade extreme VWAP deviations
                        predicted_side = 'LONG' if predicted_r > 0 else 'SHORT'
                        if vwap_z > 2.5 and predicted_side == 'LONG':
                            i += 1
                            continue
                        if vwap_z < -2.5 and predicted_side == 'SHORT':
                            i += 1
                            continue

                    # ==============================================================================

                    side = 'LONG' if predicted_r > 0 else 'SHORT'
                    entry = close[idx]
                    a = atr[idx]
                    if not np.isfinite(a) or a <= 0:
                        i += 1
                        continue

                    sl_dist = self.sl_mult * a
                    tp_dist = self.tp_mult * a
                    sl = entry - sl_dist if side == 'LONG' else entry + sl_dist
                    tp = entry + tp_dist if side == 'LONG' else entry - tp_dist

                    # Simulate bracket
                    outcome = _simulate_exit(
                        high[idx+1:idx+self.max_bars+1],
                        low[idx+1:idx+self.max_bars+1],
                        sl, tp, side
                    )

                    # Calculate PnL
                    cost = entry * (self.cost_bps / 10000)
                    if outcome == 'win':
                        pnl_per_share = tp_dist - cost
                    elif outcome == 'loss':
                        pnl_per_share = -(sl_dist + cost)
                    else:
                        exit_price = close[min(idx + self.max_bars, len(close) - 1)]
                        raw = (exit_price - entry) if side == 'LONG' else (entry - exit_price)
                        pnl_per_share = raw - cost

                    # Size based on risk
                    qty = max(1, int((equity * self.risk_per_trade) / sl_dist))
                    trade_pnl = pnl_per_share * qty
                    equity += trade_pnl

                    all_trades.append({
                        'ticker': t, 'side': side, 'entry': entry,
                        'outcome': outcome, 'pnl': trade_pnl,
                        'r_multiple': pnl_per_share / sl_dist if sl_dist > 0 else 0,
                        'date': test_indices[i]
                    })
                    equity_curve.append(equity)

                    # Skip forward by max_bars to avoid overlapping trades
                    i += self.max_bars
                    continue

                # Daily return tracking
                if equity != prev_equity:
                    daily_returns.append((equity - prev_equity) / prev_equity)
                    prev_equity = equity

            except Exception as e:
                logging.debug(f"Backtest {t}: {e}")
                continue

        # Compute metrics
        metrics = self._compute_metrics(all_trades, equity_curve, initial_equity)
        logging.info(f"📊 Backtest complete: {len(all_trades)} trades")
        logging.info(f"   Final equity: ${equity:,.0f} ({(equity/initial_equity - 1)*100:+.1f}%)")
        logging.info(f"   Win rate: {metrics['win_rate']:.1%} | Profit Factor: {metrics['profit_factor']:.2f}")
        logging.info(f"   Sharpe: {metrics['sharpe']:.2f} | Max DD: {metrics['max_drawdown']:.1%}")
        logging.info(f"   Avg R: {metrics['avg_r']:.2f} | Expectancy: {metrics['expectancy']:.3f}")
        return {'equity_curve': equity_curve, 'trades': all_trades, 'metrics': metrics}

    @staticmethod
    def _compute_metrics(trades, equity_curve, initial_equity):
        if not trades:
            return {'win_rate': 0, 'profit_factor': 0, 'sharpe': 0,
                    'max_drawdown': 0, 'avg_r': 0, 'expectancy': 0, 'n_trades': 0}

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1

        # Max drawdown from equity curve
        peak = initial_equity
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        r_values = [t['r_multiple'] for t in trades]
        pnls = [t['pnl'] for t in trades]

        # Daily Sharpe approximation
        if len(pnls) > 1:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if np.std(pnls) > 0 else 0
        else:
            sharpe = 0

        return {
            'n_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'avg_r': float(np.mean(r_values)) if r_values else 0,
            'expectancy': float(np.mean(pnls)) if pnls else 0,
        }


# ==============================================================================
# PERFORMANCE MONITORING & ADAPTIVE RETRAINING
# ==============================================================================
class PerformanceMonitor:
    """
    Monitors live trading performance and detects model drift.
    Triggers retraining when performance degrades significantly.
    """
    def __init__(self, baseline_winrate=0.50):
        self.recent_trades = deque(maxlen=30)  # Last 30 trades
        self.baseline_winrate = baseline_winrate
        self.last_check = datetime.datetime.now()

    def add_trade_result(self, won):
        """Add a trade outcome (True = win, False = loss)"""
        self.recent_trades.append(1 if won else 0)

    def should_retrain(self):
        """Check if model drift is detected"""
        if len(self.recent_trades) < 20:
            return False

        # Only check every hour
        if (datetime.datetime.now() - self.last_check).total_seconds() < 3600:
            return False

        self.last_check = datetime.datetime.now()

        recent_wr = sum(self.recent_trades) / len(self.recent_trades)

        # Statistical significance test (binomial) - SciPy compat
        try:
            try:
                from scipy.stats import binomtest as _binomtest
                p_value = _binomtest(sum(self.recent_trades), len(self.recent_trades), self.baseline_winrate).pvalue
            except ImportError:
                from scipy.stats import binom_test as _binom_test
                p_value = _binom_test(sum(self.recent_trades), len(self.recent_trades), self.baseline_winrate)

            # If statistically significant AND worse than expected
            if p_value < 0.05 and recent_wr < self.baseline_winrate - 0.10:
                logging.warning(f"⚠️ MODEL DRIFT DETECTED! WR={recent_wr:.1%} (expected {self.baseline_winrate:.1%}, p={p_value:.3f})")
                return True
        except Exception:
            # Fallback: simple threshold check
            if recent_wr < self.baseline_winrate - 0.15:
                logging.warning(f"⚠️ Performance degraded! WR={recent_wr:.1%} (expected {self.baseline_winrate:.1%})")
                return True

        return False

    def update_baseline(self, new_baseline):
        """Update baseline after retraining"""
        self.baseline_winrate = new_baseline
        logging.info(f"📊 Baseline WR updated to {new_baseline:.1%}")

# ==============================================================================
# 10. DASHBOARD & UI
# ==============================================================================
class Dashboard:
    def __init__(self):
        self.rich_available = False
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.layout import Layout
            from rich import box
            from rich.live import Live
            self.console = Console()
            self.rich_available = True
        except ImportError:
            pass

    def render(self, state):
        """
        Render the full dashboard state.
        state = {
            'equity': float,
            'vix': float,
            'regime': str,
            'universe_size': int,
            'positions': list of dicts,
            'candidates': list of dicts,
            'hedged': bool,
            'orders': int,
            'pnl_day': float
        }
        """
        if self.rich_available:
            self._render_rich(state)
        else:
            self._render_ascii(state)

    def _render_rich(self, state):
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text

        # Header Metrics
        equity_color = "green" if state.get('pnl_day', 0) >= 0 else "red"
        regime_icon = "🐮" if state['regime'] == "BULL" else "🐻" if state['regime'] == "BEAR" else "🦀"

        metrics = [
            f"[bold cyan]Equity:[/bold cyan] [bold {equity_color}]${state['equity']:,.2f}[/]",
            f"[bold yellow]VIX:[/bold yellow] {state['vix']:.2f}",
            f"[bold magenta]Regime:[/bold magenta] {regime_icon} {state['regime']}",
            f"[bold blue]Univ:[/bold blue] {state['universe_size']}",
            f"[bold white]Hedged:[/bold white] {'✅' if state['hedged'] else '❌'}"
        ]

        self.console.clear()
        self.console.rule("[bold gold1]GOD MODE v14.3[/bold gold1]")
        self.console.print(Panel(Columns([Text.from_markup(m) for m in metrics]), box=box.ROUNDED, expand=True))

        # Positions Table
        pos_table = Table(title="[bold green]ACTIVE POSITIONS[/bold green]", box=box.SIMPLE_HEAD, expand=True)
        pos_table.add_column("Sym", style="cyan")
        pos_table.add_column("Side", style="white")
        pos_table.add_column("Size", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Curr", justify="right")
        pos_table.add_column("PnL (R)", justify="right")
        pos_table.add_column("Stop", justify="right")

        for p in state['positions']:
            pnl_color = "green" if p['pnl_r'] > 0 else "red"
            pos_table.add_row(
                p['symbol'], p['side'], str(p['qty']),
                f"{p['entry']:.2f}", f"{p['curr']:.2f}",
                f"[{pnl_color}]{p['pnl_r']:.2f}R[/]",
                f"{p['sl']:.2f}"
            )

        if not state['positions']:
            pos_table.add_row("-", "-", "-", "-", "-", "-", "-")

        # Candidates Table
        cand_table = Table(title="[bold yellow]TOP OPPORTUNITIES[/bold yellow]", box=box.SIMPLE_HEAD, expand=True)
        cand_table.add_column("Sym", style="cyan")
        cand_table.add_column("Score", justify="right")
        cand_table.add_column("Win%", justify="right")
        cand_table.add_column("EV", justify="right")
        cand_table.add_column("Tier", style="magenta")
        cand_table.add_column("Type", style="white")

        for c in state['candidates'][:5]: # Show top 5
             cand_table.add_row(
                 c['symbol'],
                 f"{c['score']:.2f}",
                 f"{c['p_win']:.0%}",
                 f"{c['ev']:.2f}",
                 f"{c['tier_mult']:.1f}x",
                 c['type']
             )

        if not state['candidates']:
             cand_table.add_row("-", "-", "-", "-", "-", "-")

        self.console.print(Columns([pos_table, cand_table], expand=True))

        if state.get('logs'):
            self.console.rule("[bold dim]Events[/bold dim]")
            for log in state['logs'][-5:]:
                self.console.print(f"[dim]{log}[/dim]")

    def _render_ascii(self, state):
        print("\n" + "="*60)
        print(f"GOD MODE v14.3 | ${state['equity']:,.2f} | {state['regime']} | VIX: {state['vix']:.2f}")
        print("-" * 60)
        print("POSITIONS:")
        print(f"{'SYM':<6} {'SIDE':<5} {'QTY':<5} {'ENTRY':<8} {'CURR':<8} {'PNL(R)':<6}")
        for p in state['positions']:
            print(f"{p['symbol']:<6} {p['side']:<5} {p['qty']:<5} {p['entry']:<8.2f} {p['curr']:<8.2f} {p['pnl_r']:<6.2f}")
        print("-" * 60)
        print("TOP CANDIDATES:")
        print(f"{'SYM':<6} {'SCR':<5} {'WIN%':<5} {'EV':<5} {'TYPE'}")
        for c in state['candidates'][:5]:
            print(f"{c['symbol']:<6} {c['score']:<5.2f} {c['p_win']:<5.2f} {c['ev']:<5.2f} {c['type']}")
        print("=" * 60)

    def render_loading(self, message):
        """Show a loading/progress message (Scrolling log style)"""
        # Optimized: Don't clear screen, just print progress line-by-line
        # This prevents flickering when multiple threads report status
        if "Training" in message:
            print(f"🧠 {message}")
        else:
            # For major phases, maybe clear or box
            print(f"⏳ {message}")

# ==============================================================================
# 11. MAIN LOOP
# ==============================================================================
def run_god_mode():
    print("=" * 60)
    print("🔥 GOD MODE v14.2 PRODUCTION")
    print("   Multi-Signal Scanner + ATR-Bracket Trades")
    print("=" * 60)

    db = Database_Helper()
    poly = Polygon_Helper()
    bar_stream = PolygonBarStream(KEYS['POLY'])
    fmp = FMP_Helper()
    mc_governor = MonteCarloGovernor(SETTINGS) # Monte Carlo Risk Governor
    perf_monitor = PerformanceMonitor(baseline_winrate=0.50)  # ENHANCED: Adaptive retraining
    alpaca = Alpaca_Helper(db, mc_governor, perf_monitor)  # Pass perf_monitor
    vix_helper = VIX_Helper()
    gex_helper = GEX_Helper()  # DIX/GEX for sizing and entry gating
    portfolio_risk = PortfolioRisk()  # v15: Instance-based for correlation caching
    vol_target = VolTarget(target_vol=0.15)  # v15: Volatility Targeting
    regime_detector = RegimeDetector(vix_helper, vol_target=vol_target, portfolio_risk=portfolio_risk)
    ai = WalkForwardAI()
    earnings = EarningsGuard()
    hedger = HedgeManager(alpaca)
    fundamentals = FundamentalGuard(fmp) # Phase 15: Fundamental Guard
    sa_helper = SeekingAlpha_Helper()    # Phase 16: SA Data
    optimizer = PortfolioOptimizer(target_vol=0.25) # Aggressive Optimization
    pairs_scanner = PairsScanner(poly, lookback_days=60, rescan_hours=6)
    gap_model = OvernightGapModel(earnings_guard=earnings)
    cs_ranker = CrossSectionalRanker()
    dashboard = Dashboard()

    # v15.10: Safe Import for ZoneInfo
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None  # Fallback logic if missing


    CORE = ['RKLB', 'ASTS', 'MU', 'PLTR', 'LUNR', 'NVDA', 'NET', 'KTOS',
            'AVGO', 'RIGL', 'RCAT', 'CRWV', 'HIMS', 'APP', 'VKTX', 'TMDX']

    try:
        universe = list(set(CORE + fmp.get_dynamic_universe(exclude=set(CORE))))
    except:
        universe = list(CORE)

    dashboard.render_loading("Starting AI Training...")
    ai.train(universe, poly, db=db, progress_callback=lambda msg: dashboard.render_loading(msg), core_tickers=set(CORE))

    # Run backtest after initial training to validate strategy
    try:
        dashboard.render_loading("Running walk-forward backtest...")
        backtester = BracketBacktest(ai, poly, sl_mult=SETTINGS['STOP_MULT'],
                                     tp_mult=SETTINGS['TP_MULT'], max_bars=12)
        bt_result = backtester.run(list(CORE)[:10], test_days=90)
        if bt_result['metrics']['win_rate'] < 0.30:
            logging.warning("⚠️ Backtest win rate < 30% — model may be undertrained")
    except Exception as e_bt:
        logging.warning(f"Backtest skipped: {e_bt}")

    last_refresh = datetime.datetime.now()
    last_equity_log = datetime.datetime.now()
    start_equity = alpaca.equity if alpaca.equity > 0 else 100000.0

    order_mode = "MARKET" if SETTINGS['USE_MARKET_ORDERS'] else "LIMIT"

    prices = {}
    scored_candidates = []
    scan_cache = ScanBarCache()  # New-bar gating: skip tickers with no new 15-min bar

    # Start WebSocket bar stream (hybrid: WS triggers scan, REST fetches data)
    bar_stream.start()
    bar_stream.subscribe(universe)

    while True:
        try:
            if alpaca.check_kill():
                time.sleep(60)
                continue

            # FIX: Refresh equity and sync positions EVERY loop
            alpaca.refresh_equity()
            alpaca.sync_positions()

            # Apply Monte Carlo Risk Adjustments
            if mc_governor:
                mc_governor.apply_adjustments()

            # v15: Update holdings data for correlation check
            portfolio_risk.update_holdings_data(alpaca.pos_cache)
            # FIX: Update Universe Correlation Matrix (Batch)
            portfolio_risk.update_universe_cache(universe)

            # ENHANCED: Adaptive retraining based on performance drift
            # Check for drift first
            if perf_monitor.should_retrain():
                logging.warning("🔄 DRIFT DETECTED - Forcing immediate retraining...")
                universe = list(set(CORE + fmp.get_dynamic_universe(exclude=set(CORE))))
                ai.train(universe, poly, db=db, core_tickers=set(CORE))
                gc.collect()
                scan_cache.invalidate()  # Force fresh bars after retrain
                bar_stream.subscribe(universe)  # Update WS subscriptions
                last_refresh = datetime.datetime.now()
                # Update baseline after retraining
                if len(perf_monitor.recent_trades) >= 20:
                    new_baseline = sum(perf_monitor.recent_trades) / len(perf_monitor.recent_trades)
                    perf_monitor.update_baseline(max(0.45, new_baseline))  # Don't go below 45%

            # Regular refresh every 45 mins (if no drift detected)
            elif (datetime.datetime.now() - last_refresh).total_seconds() > 2700:
                universe = list(set(CORE + fmp.get_dynamic_universe(exclude=set(CORE))))
                ai.train(universe, poly, db=db, core_tickers=set(CORE))
                gc.collect()
                scan_cache.invalidate()  # Force fresh bars after retrain
                bar_stream.subscribe(universe)  # Update WS subscriptions
                last_refresh = datetime.datetime.now()

            vix = vix_helper.get_vix()
            vix_mult = vix_helper.get_size_multiplier()
            regime = regime_detector.get_regime()

            # FIX: Use Alpaca Clock for accurate market hours (handles holidays/early close)
            try:
                clock = alpaca.api.get_clock()
                is_market_hours = clock.is_open
                # Optional: Sleep until open if close
                if not is_market_hours:
                    next_open = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
                    now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
                    sleep_sec = min(300, max(60, next_open - now_ts))

                    # Dashboard Update for Sleep
                    dashboard.render({
                        'equity': alpaca.equity,
                        'vix': vix,
                        'regime': regime,
                        'universe_size': len(universe),
                        'positions': [],
                        'candidates': [],
                        'hedged': hedger.is_hedged,
                        'orders': 0,
                        'pnl_day': alpaca.equity - start_equity,
                        'logs': [f"💤 Market closed. Sleeping {sleep_sec:.0f}s..."]
                    })

                    time.sleep(sleep_sec)
                    continue
            except Exception as e:
                # FIX: Fallback to proper ET time if API fails
                try:
                    from zoneinfo import ZoneInfo
                    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
                except ImportError:
                    # Last resort: assume local time (not ideal)
                    now_et = datetime.datetime.now()
                    logging.warning("ZoneInfo not available - using local time for market hours")

                # FIX: Proper market hours check (9:30 AM - 4:00 PM ET, weekdays)
                is_market_hours = (
                    now_et.weekday() < 5 and
                    ((now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 30)) and now_et.hour < 16)
                )
                if not is_market_hours:
                    time.sleep(60)
                    continue
            hedger.check_hedge(regime)

            # GEX-adjusted confidence threshold
            gex_mult = gex_helper.get_size_multiplier()
            gex_conf_adj = gex_helper.get_confidence_adjustment()
            threshold = SETTINGS['MIN_CONFIDENCE'] + regime_detector.get_confidence_adjustment() + gex_conf_adj

            # v15: Volatility Targeting Scalar
            vol_scalar = vol_target.update_scalar()

            # FIX: Show pending orders in log
            pending_count = len(alpaca.pending_orders)

            # FIX: Prepare Dashboard State for Rendering
            # Calculate PnL for active positions
            db_positions = []
            for t, p in alpaca.pos_cache.items():
                curr = prices.get(t, p['entry'])
                init_risk = p.get('init_risk', 1) or 1
                diff = (curr - p['entry']) if p['side'] == 'LONG' else (p['entry'] - curr)
                pnl_r = diff / init_risk
                db_positions.append({
                    'symbol': t, 'side': p['side'], 'qty': p['qty'],
                    'entry': p['entry'], 'curr': curr, 'pnl_r': pnl_r,
                    'sl': p.get('sl', 0)
                })

            dashboard.render({
                'equity': alpaca.equity,
                'vix': vix,
                'regime': regime,
                'universe_size': len(universe),
                'positions': db_positions,
                'candidates': scored_candidates if 'scored_candidates' in locals() else [],
                'hedged': hedger.is_hedged,
                'orders': pending_count,
                'pnl_day': alpaca.equity - start_equity
            })



            # FIX: Fetch all prices in 1 API call using Polygon Snapshot
            snap = poly.fetch_snapshot_prices(universe, ttl=5)
            prices = {t: snap[t]["price"] for t in snap.keys()}

            # FIX: Pre-filter by day dollar volume before expensive bar fetches
            candidates = []
            for t in universe:
                if t not in snap:
                    continue
                day_vol_usd = snap[t]["dayVol"] * prices[t]
                if day_vol_usd < 1_000_000:  # Skip illiquid names
                    continue
                candidates.append(t)

            # FIX: Batch news fetching to avoid per-ticker latency in loop
            all_news_scores = fmp.news_scores_batch(candidates, lookback_hours=SETTINGS.get("NEWS_LOOKBACK_HOURS", 24))

            # Cross-sectional factor ranking (once per scan cycle)
            try:
                cs_ranker.update(snap, fmp, universe)
            except Exception:
                pass

            # Market context: compute SPY ROC + VIX ROC once per scan cycle
            try:
                _spy_scan = scan_cache.get_if_same_slot('SPY')
                if _spy_scan is None:
                    _spy_scan = poly.fetch_data('SPY', days=5, mult=15)
                    if len(_spy_scan) > 0:
                        scan_cache.put('SPY', _spy_scan)
                live_spy_roc_5 = float(_spy_scan['Close'].pct_change(5).iloc[-1]) if len(_spy_scan) > 5 else 0.0
                live_spy_roc_20 = float(_spy_scan['Close'].pct_change(20).iloc[-1]) if len(_spy_scan) > 20 else 0.0
            except Exception:
                live_spy_roc_5, live_spy_roc_20 = 0.0, 0.0

            live_vix_level = vix  # Already fetched above
            try:
                _vix_h = yf.Ticker('^VIX').history(period='10d')
                live_vix_roc = float(_vix_h['Close'].pct_change(5).iloc[-1]) if len(_vix_h) > 5 else 0.0
            except Exception:
                live_vix_roc = 0.0

            # Phase 23b: Candidate Collection for Top-K Selection
            scored_candidates = []

            # PHASE 23b: Parallelized Candidate Scanning
            def check_candidate(t):
                try:
                    # New-bar gating: reuse cached bars if still in same 15-min slot
                    cached_df = scan_cache.get_if_same_slot(t)
                    if cached_df is not None:
                        df_15m = cached_df
                    else:
                        # FIX: Fetch 12 days (approx 300 bars) to support EMA200
                        df_15m = poly.fetch_data(t, days=12, mult=15)
                        if len(df_15m) < 200: return None
                        scan_cache.put(t, df_15m)

                    if len(df_15m) < 200: return None

                    # FIX: Attach Daily ATR
                    df_15m = attach_daily_atr_to_15m(df_15m, t, poly=poly)

                    # News Guard & Penalty
                    news_score = all_news_scores.get(t, 0)
                    news_penalty = (news_score >= SETTINGS.get("NEWS_SOFT_PENALTY_SCORE", 1))

                    # Severe News Skip
                    if news_score >= SETTINGS.get("NEWS_HARD_SKIP_SCORE", 3):
                        return None

                    # Morning Burn-In
                    burnin_conf_add = 0.0
                    burnin_size_mult = 1.0
                    if ZoneInfo: now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
                    else: now_et = datetime.datetime.now()

                    if now_et.hour == 9 and now_et.minute < (30 + SETTINGS.get("OPEN_BURNIN_MINUTES", 15)):
                            burnin_conf_add = SETTINGS.get("BURNIN_CONF_ADD", 0.03)
                            burnin_size_mult = SETTINGS.get("BURNIN_SIZE_MULT", 0.60)

                    # Guards
                    if not earnings.check_safe(t): return None
                    if not fundamentals.check_healthy(t): return None
                    if not LiquidityFilter.check(df_15m): return None

                    # Portfolio Check (Initial)
                    combined_exposure = alpaca.pos_cache.copy()
                    for o in alpaca.pending_orders.values():
                        risk_per_share = abs(float(o.get('price', 0)) - float(o.get('sl', 0)))
                        combined_exposure[o['symbol']] = {'qty': int(o.get('qty', 0)), 'init_risk': risk_per_share, 'pending': True}

                    if not portfolio_risk.should_allow_entry(t, combined_exposure, alpaca.equity):
                        return None

                    # Feature Eng
                    df_15m['RSI'] = df_15m.ta.rsi(14)
                    df_15m['ADX'] = df_15m.ta.adx(14)['ADX_14']
                    df_15m['ATR'] = df_15m.ta.atr(14)
                    df_15m['ATR_Pct'] = df_15m['ATR'] / df_15m['Close']
                    # EMA 200 (Trend Filter)
                    df_15m['EMA_200'] = df_15m.ta.ema(200)

                    v = df_15m['Volume'].rolling(20)
                    df_15m['Vol_Rel'] = df_15m['Volume'] / v.mean()
                    df_15m['Kalman'] = get_kalman_filter(df_15m['Close'].values)
                    df_15m['Kalman_Dist'] = (df_15m['Close'] - df_15m['Kalman']) / df_15m['Close']
                    hurst = get_hurst(df_15m['Close'].values)
                    df_15m['Hurst'] = df_15m['Close'].rolling(50).apply(lambda x: get_hurst(x.values) if len(x) >= 20 else 0.5, raw=False).fillna(hurst)

                    # --- FEATURES MUST MATCH TRAINING (was missing, causing silent predict failures) ---
                    # Bollinger Bands
                    bb = df_15m.ta.bbands(20, 2)
                    if bb is not None and len(bb.columns) >= 3:
                        df_15m['BB_Upper'] = bb.iloc[:, 0]
                        df_15m['BB_Mid'] = bb.iloc[:, 1]
                        df_15m['BB_Lower'] = bb.iloc[:, 2]
                        df_15m['BB_Width'] = (df_15m['BB_Upper'] - df_15m['BB_Lower']) / df_15m['Close']
                        df_15m['BB_Position'] = (df_15m['Close'] - df_15m['BB_Lower']) / (df_15m['BB_Upper'] - df_15m['BB_Lower'] + 1e-10)
                    else:
                        df_15m['BB_Width'] = 0.02
                        df_15m['BB_Position'] = 0.5

                    # VWAP
                    try:
                        df_15m['VWAP'] = (df_15m['Close'] * df_15m['Volume']).rolling(20).sum() / df_15m['Volume'].rolling(20).sum()
                        df_15m['VWAP_Dist'] = (df_15m['Close'] - df_15m['VWAP']) / df_15m['Close']
                    except:
                        df_15m['VWAP_Dist'] = 0.0

                    # Microstructure
                    df_15m['HL_Range'] = (df_15m['High'] - df_15m['Low']) / df_15m['Close']
                    df_15m['Money_Flow'] = (df_15m['Close'] * df_15m['Volume']).rolling(10).sum()
                    mf_mean = df_15m['Money_Flow'].rolling(50).mean()
                    df_15m['Money_Flow'] = df_15m['Money_Flow'] / mf_mean

                    # Momentum
                    df_15m['ROC_5'] = df_15m['Close'].pct_change(5)
                    df_15m['ROC_20'] = df_15m['Close'].pct_change(20)

                    # Volume surge
                    df_15m['Vol_Surge'] = df_15m['Volume'] / df_15m['Volume'].rolling(5).mean()

                    # Regime features
                    df_15m['Volatility_Rank'] = df_15m['ATR_Pct'].rolling(100, min_periods=20).apply(
                        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5, raw=False
                    )
                    _ret_live = df_15m['Close'].pct_change()
                    df_15m['Trend_Consistency'] = _ret_live.rolling(20, min_periods=5).apply(
                        lambda s: (s > 0).mean(), raw=False
                    )

                    # Time features
                    df_15m['Hour'] = df_15m.index.hour
                    df_15m['Day_of_Week'] = df_15m.index.dayofweek

                    # Market context features (computed once per scan cycle above)
                    df_15m['SPY_ROC_5'] = live_spy_roc_5
                    df_15m['SPY_ROC_20'] = live_spy_roc_20
                    df_15m['VIX_Level'] = live_vix_level
                    df_15m['VIX_ROC'] = live_vix_roc

                    # Cross-sectional factor ranks (computed once per scan cycle)
                    _ranks = cs_ranker.get_ranks(t)
                    df_15m['momentum_rank'] = _ranks['momentum_rank']
                    df_15m['volume_rank'] = _ranks['volume_rank']
                    df_15m['value_rank'] = _ranks['value_rank']
                    df_15m['composite_rank'] = _ranks['composite_rank']
                    # --- END FEATURE PARITY FIX ---

                    row = df_15m.iloc[-2].to_dict()

                    # Injections
                    if sa_helper.key:
                        row.update(sa_helper.get_news_features(t))
                        row.update(sa_helper.get_ratings(t))
                    else:
                        row['sa_news_count_3d'] = 0
                        row['sa_sentiment_score'] = 0

                    # Fundamental features from FMP
                    fund_feats = fmp.get_fundamental_features(t)
                    row.update(fund_feats)

                    # Dynamic news weighting
                    _ns = all_news_scores.get(t, 0)
                    row['news_impact_weight'] = float(_ns) if _ns else 0.0

                    # Hygiene
                    bad_feats = [k for k in ai.active_features if k in row and not np.isfinite(row[k])]
                    if bad_feats: return None

                    # Predict R-value with quantile-based probability intervals
                    predicted_r, p_win, p_hold, p_loss, uncertainty = ai.predict_with_intervals(row)

                    # ATR
                    daily_atr = get_daily_atr_polygon(poly, t)
                    if not daily_atr or np.isnan(daily_atr) or daily_atr <= 0: daily_atr = row.get('ATR', prices.get(t, row['Close']) * 0.02)

                    # ==============================================================================
                    # INSTITUTIONAL-GRADE CONFIDENCE FILTERING (>50% WR, >1.5 PF TARGET)
                    # ==============================================================================

                    # 1. Base R-value threshold (adaptive based on market regime)
                    base_r_threshold = 0.10  # Default

                    # Increase threshold in high-volatility regimes (harder to predict)
                    vix_level = row.get('VIX_Level', 20.0)
                    if vix_level > 25:
                        base_r_threshold = 0.15
                    elif vix_level > 30:
                        base_r_threshold = 0.20

                    # Skip if R-value too close to zero (equivalent to HOLD)
                    if abs(predicted_r) < base_r_threshold:
                        return None

                    # 2. Skip if prediction interval too wide (unreliable)
                    if uncertainty > 2.0:
                        return None

                    # 3. VPIN Filter: Block toxic order flow (informed trading present)
                    vpin = row.get('VPIN', 0.0)
                    if vpin > 0.85:  # 85th percentile = very toxic
                        logging.debug(f"   🚫 {t} blocked by VPIN={vpin:.2f} (toxic order flow)")
                        return None

                    # 4. Amihud Illiquidity Filter: Block low-liquidity conditions (high slippage risk)
                    amihud = row.get('Amihud_Illiquidity', 0.5)
                    if amihud > 0.90:  # 90th percentile = very illiquid
                        logging.debug(f"   🚫 {t} blocked by Amihud={amihud:.2f} (low liquidity)")
                        return None

                    # 5. Enhanced Confidence Filter: Use quantile spread as confidence proxy
                    # Wide quantile spread (q90 - q10) = high uncertainty = skip
                    if uncertainty > 1.5:  # Tighter threshold for high-accuracy trading
                        logging.debug(f"   🚫 {t} blocked by high uncertainty={uncertainty:.2f}")
                        return None

                    # 6. VWAP Z-Score Filter: Extreme deviations often mean-revert (fade the extremes)
                    vwap_z = row.get('VWAP_ZScore', 0.0)
                    predicted_side = "LONG" if predicted_r > 0 else "SHORT"

                    # If VWAP Z-score is extreme, ensure we're trading WITH the reversion, not against it
                    if abs(vwap_z) > 2.5:
                        # Extreme high: expect mean reversion DOWN (favor SHORTS)
                        if vwap_z > 2.5 and predicted_side == "LONG":
                            logging.debug(f"   🚫 {t} blocked: VWAP_Z={vwap_z:.2f} (overextended, fade the move)")
                            return None
                        # Extreme low: expect mean reversion UP (favor LONGS)
                        if vwap_z < -2.5 and predicted_side == "SHORT":
                            logging.debug(f"   🚫 {t} blocked: VWAP_Z={vwap_z:.2f} (oversold, fade the move)")
                            return None

                    # 7. Regime-Based Filtering: Use GEX Proxy to match strategy to regime
                    gex_proxy = row.get('Regime_GEX_Proxy', 0)

                    # Positive GEX (mean reversion regime): Only take mean-reversion signals
                    if gex_proxy == 1:
                        # Mean reversion: favor trades when price is stretched from VWAP
                        if abs(vwap_z) < 1.0:  # Not stretched enough
                            logging.debug(f"   🚫 {t} blocked in +GEX regime: VWAP_Z={vwap_z:.2f} (need stretch for MR)")
                            return None

                    # Negative GEX (trending regime): Only take trend-following signals
                    elif gex_proxy == -1:
                        # Trend following: favor trades with strong VWAP slope
                        vwap_slope = row.get('VWAP_Slope', 0.0)
                        if abs(vwap_slope) < 0.001:  # Flat VWAP = no trend
                            logging.debug(f"   🚫 {t} blocked in -GEX regime: VWAP_Slope={vwap_slope:.4f} (need trend)")
                            return None

                    # 8. Minimum Win Probability Filter (for >50% win rate target)
                    if p_win < 0.52:  # Slightly above 50% to account for costs
                        logging.debug(f"   🚫 {t} blocked by low p_win={p_win:.2%}")
                        return None

                    # ==============================================================================

                    # Determine side based on R-value sign
                    side = predicted_side

                    # 🚀 TREND FILTER BOOSTER: "The Golden Gate"
                    # Filter out counter-trend signals to maximize win-rate
                    last_close = row['Close']
                    ema_200 = row.get('EMA_200', 0)
                    if ema_200 > 0:
                        # If Price < EMA200, only SHORTs allowed
                        if last_close < ema_200 and side == "LONG":
                            # Check exception: Deep oversold bounce? (RSI < 25)
                            if row.get('RSI', 50) > 25:
                                return None

                        # If Price > EMA200, only LONGs allowed
                        if last_close > ema_200 and side == "SHORT":
                            # Check exception: Deep overbought dump? (RSI > 75)
                            if row.get('RSI', 50) < 75:
                                return None

                    entry_price = prices.get(t, row['Close'])

                    # --- HYBRID GOD MODE LOGIC (Dual-Tier) ---
                    hurst_val = row.get('Hurst', 0.5)
                    adx_val = row.get('ADX', 20)

                    t1_cfg = SETTINGS.get("TIER_1", {})
                    t2_cfg = SETTINGS.get("TIER_2", {})

                    # Tier 1: Specialist (High Conf, Low Chop, High Trend)
                    is_tier_1 = (
                        (p_win >= t1_cfg.get("MIN_PROB", 0.30)) and
                        (hurst_val < t1_cfg.get("MAX_HURST", 0.55)) and
                        (adx_val > t1_cfg.get("MIN_ADX", 25))
                    )

                    # Tier 2: Grinder (Med Conf, Med Chop)
                    is_tier_2 = (
                        (p_win >= t2_cfg.get("MIN_PROB", 0.20)) and
                        (hurst_val < t2_cfg.get("MAX_HURST", 0.45)) and
                        (adx_val > t2_cfg.get("MIN_ADX", 20))
                    )

                    active_tier = None
                    tier_type = "Filtered"

                    if is_tier_1:
                        active_tier = t1_cfg
                        tier_type = "Tier 1 (Specialist)"
                    elif is_tier_2:
                        active_tier = t2_cfg
                        tier_type = "Tier 2 (Grinder)"

                    # FINAL GATE
                    if not active_tier:
                        # Fail
                        return None

                    # --- Apply Tier Parameters ---
                    risk_mult = active_tier.get("RISK_MULT", 1.0)
                    rr_target = active_tier.get("RR", 1.5)

                    # Burn-in / News Penalties
                    if burnin_size_mult < 1.0: risk_mult *= burnin_size_mult
                    if news_penalty: risk_mult *= 0.75

                    # Calculate Stops & Targets
                    base_sl_mult = SETTINGS.get('STOP_MULT', 1.5)
                    tp_dist_mult = base_sl_mult * rr_target

                    sl = entry_price - base_sl_mult * daily_atr if side == "LONG" else entry_price + base_sl_mult * daily_atr
                    tp = entry_price + tp_dist_mult * daily_atr if side == "LONG" else entry_price - tp_dist_mult * daily_atr

                    # Calculate EV (Net)
                    stop_dist = abs(entry_price - sl)
                    cost_r = estimate_cost_in_R(df_15m, stop_dist)

                    # EV Gate (Legacy safety check)
                    # p_win is already high from Tier logic, but check +EV
                    # EV = (Win% * Reward) - (Loss% * Risk) - Cost
                    # Reward = RR, Risk = 1.0
                    ev_net = (p_win * rr_target) - ((1.0 - p_win) * 1.0) - cost_r

                    if ev_net < SETTINGS.get("EV_GATE_R", 0.02):
                        return None

                    # Score for Top-K Sorting
                    # Prioritize Tier 1, then EV
                    score = ev_net
                    if is_tier_1: score += 10.0 # Strict priority to Tier 1

                    _entry_features = {k: row.get(k, 0) for k in ai.active_features}

                    return {
                        'symbol': t, 'side': side, 'type': tier_type,
                        'score': score, 'p_win': p_win, 'ev': ev_net, 'price': entry_price,
                        'sl': sl, 'tp': tp, 'daily_atr': daily_atr,
                        'tier_mult': risk_mult, 'hurst_scalar': 1.0, # Baked into risk_mult
                        'burnin_size_mult': burnin_size_mult,
                        'rr_net': rr_target, 'prob_vec': (p_win, p_hold, p_loss),
                        'entry_features': _entry_features
                    }

                except Exception as eobj:
                     ERROR_TRACKER.record_failure(f"Scan_{t}", str(eobj))
                     return None

            # RUN PARALLEL SCANNER
            scored_candidates = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS) as scanner_pool:
                future_to_cand = {scanner_pool.submit(check_candidate, t): t for t in candidates}
                for future in concurrent.futures.as_completed(future_to_cand):
                    res = future.result()
                    if res:
                        scored_candidates.append(res)

            # --- Block new entries near close (overnight gap risk) ---
            if gap_model.is_pre_close():
                logging.info("🌙 Pre-close window: skipping new entries, managing existing only.")
                scored_candidates = []

            # --- Pairs Trading (market-neutral, complements directional) ---
            if HAS_COINT and regime == 'CHOP' and not gap_model.is_pre_close():
                try:
                    pairs_scanner.scan_pairs(universe)
                    pair_signals = pairs_scanner.get_pair_signals(prices)
                    for ps in pair_signals[:2]:  # Max 2 pairs trades per cycle
                        logging.info(f"🔗 Pairs signal: {ps['sideA']} {ps['legA']} / "
                                     f"{ps['sideB']} {ps['legB']} z={ps['z_score']:.1f}")
                        # Execute pair as two bracket orders with tight stops
                        pair_atr_a = get_daily_atr_polygon(poly, ps['legA']) or (prices.get(ps['legA'], 100) * 0.015)
                        pair_atr_b = get_daily_atr_polygon(poly, ps['legB']) or (prices.get(ps['legB'], 100) * 0.015)
                        pair_risk_pct = SETTINGS['RISK_PER_TRADE'] * 0.5  # Half-size per leg
                        for leg, side_key, atr_leg in [
                            (ps['legA'], 'sideA', pair_atr_a),
                            (ps['legB'], 'sideB', pair_atr_b)
                        ]:
                            if leg in alpaca.pos_cache:
                                continue
                            lp = prices.get(leg, 0)
                            if lp <= 0:
                                continue
                            sd = 1.5 * atr_leg
                            pair_qty = max(1, int((alpaca.equity * pair_risk_pct) / sd))
                            sl_p = lp - sd if ps[side_key] == 'LONG' else lp + sd
                            tp_p = lp + 2.0 * atr_leg if ps[side_key] == 'LONG' else lp - 2.0 * atr_leg
                            alpaca.submit_bracket(leg, ps[side_key], pair_qty, lp, sl_p, tp_p, atr_override=atr_leg)
                except Exception as e_pairs:
                    logging.debug(f"Pairs trading error: {e_pairs}")

            # --- Execution Phase (Top-K + Portfolio Optimization) ---
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)

            open_positions = len(alpaca.pos_cache) + len(alpaca.pending_orders)
            max_new_trades = max(1, SETTINGS['MAX_POSITIONS'] - open_positions)
            top_k_limit = min(8, max_new_trades) # Increased to 8 for optimizer to have choices

            # Select Top Candidates for Optimization
            final_batch = scored_candidates[:top_k_limit]

            if final_batch:
                # 1. Prepare Data for Optimizer
                opt_symbols = [c['symbol'] for c in final_batch]

                # Fetch history for covariance + SPY for Beta
                fetch_symbols = list(set(opt_symbols + ['SPY']))

                try:
                    # Quick fetch of daily data for correlation (more robust than 15m for broad covariance)
                    cov_prices = poly.fetch_batch_bars(fetch_symbols, days=30)
                except:
                    cov_prices = pd.DataFrame() # Fallback

                # 2. Run Optimizer
                logging.info(f"🧠 Optimizing Portfolio for {len(final_batch)} candidates...")
                opt_weights = optimizer.get_optimal_weights(final_batch, cov_prices, market_ticker='SPY')

                # 3. Execute Optimal Allocations
                for cand in final_batch:
                    t = cand['symbol']
                    weight = opt_weights.get(t, 0.0)

                    if weight <= 0.01:
                        logging.info(f"📉 Optimizer Zeroed out {t} (Risk/Correlation too high)")
                        continue

                    # --- RISK-BASED SIZING (was bypassed - stop distance was ignored) ---
                    stop_dist = abs(cand['price'] - cand['sl'])
                    if stop_dist <= 0:
                        continue

                    # A. Risk-based qty: respects stop distance (SETTINGS['RISK_PER_TRADE'])
                    risk_qty = alpaca.calculate_position_size(
                        cand['price'], cand['sl'], vix_mult=vix_mult
                    )

                    # B. Kelly-optimal risk fraction (was defined but never called)
                    kelly_scalar = 1.0
                    if SETTINGS.get('USE_KELLY', False):
                        pW, pH, pL = cand.get('prob_vec', (0.50, 0.05, 0.45))
                        rr = cand.get('rr_net', 2.0)
                        kelly_f = kelly_3_outcome(pW, pL, pH, b=rr, d=0.15,
                                                  f_max=SETTINGS.get('KELLY_MAX_RISK', 0.04))
                        if kelly_f > 0:
                            base_risk = SETTINGS['RISK_PER_TRADE']
                            kelly_scalar = min(
                                SETTINGS.get('KELLY_MAX_RISK', 0.03) / base_risk,
                                max(SETTINGS.get('KELLY_MIN_RISK', 0.003) / base_risk,
                                    kelly_f / base_risk * SETTINGS.get('KELLY_FRACTION', 0.75))
                            )

                    # C. Optimizer weight as a proportion cap
                    opt_qty = int((alpaca.equity * weight * 1.2) / cand['price'])

                    # D. Combine: use min of risk-based and optimizer-based, then apply scalars
                    mc_scalar = mc_governor.get_risk_scalar() if mc_governor else 1.0
                    ev_scalar = ev_to_size_mult(cand['ev'])
                    qty = min(risk_qty, opt_qty)
                    # Apply all scalars: vol, kelly, monte-carlo, GEX, EV-based sizing
                    qty = int(qty * kelly_scalar * mc_scalar * vol_scalar * gex_mult * ev_scalar)
                    qty = max(qty, 0)

                    if qty == 0: continue

                    tag = "⭐" if t in CORE else "Quant"
                    logging.info(
                        f"🚀 {tag} {cand['side']} {t} | OptW: {weight:.1%} "
                        f"| RiskQty: {risk_qty} OptQty: {opt_qty} Final: {qty} "
                        f"| Kelly: {kelly_scalar:.2f} Vol: {vol_scalar:.2f} MC: {mc_scalar:.2f} "
                        f"| EV: {cand['ev']:.2f} EVx: {ev_scalar:.2f}"
                    )

                    alpaca.submit_bracket(
                        t, cand['side'], qty, cand['price'],
                        cand['sl'], cand['tp'],
                        atr_override=cand['daily_atr'],
                        entry_features=cand.get('entry_features')
                    )

            # Legacy loop removed. Optimizer handles execution.

            # Manage positions
            # FIX: Ensure we have prices for all held positions (even if scan failed)
            for t in list(alpaca.pos_cache.keys()):
                if t not in prices:
                    try:
                        df = poly.fetch_data(t, days=1)
                        if len(df) > 0:
                            prices[t] = df['Close'].iloc[-1]
                    except:
                        pass

            for t, p in list(alpaca.pos_cache.items()):
                try:
                    if t not in prices:
                        logging.warning(f"No price for {t}, skipping management")
                        continue

                    curr = prices[t]
                    entry = p['entry']

                    # OVERNIGHT GAP RISK: exit high-risk positions before close
                    if gap_model.is_pre_close():
                        init_risk = p.get('init_risk', 0)
                        if init_risk <= 0: init_risk = 1.0
                        amt = (curr - entry) if p['side'] == 'LONG' else (entry - curr)
                        pos_pnl_r = amt / init_risk
                        if gap_model.should_exit_pre_close(t, vix, pos_pnl_r):
                            logging.info(f"🌙 Pre-close exit: {t} (PnL={pos_pnl_r:.2f}R, VIX={vix:.1f})")
                            alpaca.close_position(t, reason="OvernightGapRisk")
                            continue
                        else:
                            # Tighten stop for overnight hold
                            new_sl = gap_model.pre_close_stop_tightening(
                                p['side'], entry, p.get('sl', entry), p.get('atr', 0), pos_pnl_r
                            )
                            if new_sl is not None and new_sl != p.get('sl', entry):
                                if alpaca.replace_stop(t, new_sl):
                                    alpaca.pos_cache[t]['sl'] = new_sl
                                    logging.info(f"🌙 Overnight tighten: {t} SL -> ${new_sl:.2f}")

                    # v15: VOL-ADAPTIVE TIME STOP
                    # High-vol names get more time (wider ATR = slower resolution).
                    # Base: 180 min. Scale by ATR_Pct relative to median (~0.8%).
                    duration_mins = (datetime.datetime.now() - datetime.datetime.fromisoformat(p['ts'])).total_seconds() / 60
                    atr_pct = p.get('atr', 0) / entry if entry > 0 else 0.008
                    # Low vol (< 0.5%) -> 150 min, high vol (> 1.5%) -> 240 min
                    time_limit = max(120, min(300, 180 * (atr_pct / 0.008))) if atr_pct > 0 else 180

                    if duration_mins > time_limit:
                        init_risk = p.get('init_risk', 0)
                        if init_risk <= 0: init_risk = 1.0 # prevent div/0

                        amt = (curr - entry) if p['side'] == 'LONG' else (entry - curr)
                        pnl_r = amt / init_risk

                        # Close if stale and low profit
                        if pnl_r < 0.5:
                            logging.info(f"⌛ Time Stop: {t} held {duration_mins:.0f}m (limit {time_limit:.0f}m) PnL ({pnl_r:.2f}R). Closing.")
                            alpaca.close_position(t, reason="TimeStop")
                            continue

                    # MANAGE PROFITS (Trailing Stop & Ratchet)
                    atr = p['atr']
                    side = p['side']
                    current_sl = p.get('sl', entry)

                    # Use stored init_risk for accurate R calc
                    init_risk = p.get('init_risk', 0)
                    if init_risk <= 0:
                        init_risk = abs(entry - current_sl) if current_sl != entry else (1.5 * atr)

                    move = (curr - entry) if side == 'LONG' else (entry - curr)
                    r_val = move / init_risk if init_risk > 0 else 0

                    # v15.10: RATCHET / BREAKEVEN LOGIC
                    # If R > 1.0 and not yet moved to BE, move SL to Entry + 0.05R
                    if r_val >= SETTINGS.get("RATCHET_R", 1.0) and not p.get('ratcheted', False):
                        buffer = SETTINGS.get("RATCHET_BUFFER_ATR", 0.10) * atr

                        if side == 'LONG':
                            new_sl = entry + buffer
                            if new_sl > current_sl: # Only move up
                                if alpaca.replace_stop(t, new_sl):
                                    alpaca.pos_cache[t]['sl'] = new_sl
                                    alpaca.pos_cache[t]['ratcheted'] = True
                                    logging.info(f"🛡️ RATCHET {t}: SL -> ${new_sl:.2f} (+{r_val:.2f}R)")
                        else: # SHORT
                            new_sl = entry - buffer
                            if new_sl < current_sl: # Only move down
                                if alpaca.replace_stop(t, new_sl):
                                    alpaca.pos_cache[t]['sl'] = new_sl
                                    alpaca.pos_cache[t]['ratcheted'] = True
                                    logging.info(f"🛡️ RATCHET {t}: SL -> ${new_sl:.2f} (+{r_val:.2f}R)")

                    # ENHANCED: PARTIAL PROFIT TAKING at 1.5R and 2.5R
                    if r_val >= 1.5 and not p.get('scaled_1', False) and p['qty'] >= 3:
                        # Take 1/3 off at 1.5R
                        close_qty = p['qty'] // 3
                        if close_qty > 0:
                            try:
                                # Close partial position
                                alpaca.api.submit_order(
                                    symbol=t,
                                    qty=close_qty,
                                    side='sell' if side == 'LONG' else 'buy',
                                    type='market',
                                    time_in_force='day'
                                )
                                alpaca.pos_cache[t]['scaled_1'] = True
                                alpaca.pos_cache[t]['qty'] -= close_qty
                                logging.info(f"📉 SCALED 1/3 of {t} @ +{r_val:.2f}R (qty: {close_qty})")
                            except Exception as e:
                                logging.debug(f"Partial close 1 failed for {t}: {e}")

                    if r_val >= 2.5 and not p.get('scaled_2', False) and p['qty'] >= 2:
                        # Take half of remaining at 2.5R
                        close_qty = p['qty'] // 2
                        if close_qty > 0:
                            try:
                                alpaca.api.submit_order(
                                    symbol=t,
                                    qty=close_qty,
                                    side='sell' if side == 'LONG' else 'buy',
                                    type='market',
                                    time_in_force='day'
                                )
                                alpaca.pos_cache[t]['scaled_2'] = True
                                alpaca.pos_cache[t]['qty'] -= close_qty
                                logging.info(f"📉 SCALED 50% remaining of {t} @ +{r_val:.2f}R (qty: {close_qty})")
                            except Exception as e:
                                logging.debug(f"Partial close 2 failed for {t}: {e}")

                    # v15.10: TRAILING STOP (Start after +2R, regime-adaptive distance)
                    if r_val >= SETTINGS.get("TRAIL_START_R", 2.0) and atr > 0:
                        trail_mult = SETTINGS.get("TRAIL_ATR_MULT", 1.0)
                        if regime == 'CHOP':
                            trail_mult *= 0.7   # Tighter trail in chop (take profits)
                        elif regime in ('BULL', 'BEAR'):
                            trail_mult *= 1.2   # Wider trail in trends (let runners run)
                        trail_dist = trail_mult * atr
                        if side == 'LONG':
                            trail_sl = curr - trail_dist
                            # Never loosen stop
                            new_sl = max(current_sl, trail_sl)
                            if new_sl > current_sl + 0.01:
                                if alpaca.replace_stop(t, new_sl):
                                    alpaca.pos_cache[t]['sl'] = new_sl
                                    logging.info(f"🔒 TRAIL {t}: SL -> ${new_sl:.2f} (+{r_val:.2f}R)")
                        else: # SHORT
                            trail_sl = curr + trail_dist
                            new_sl = min(current_sl, trail_sl)
                            if new_sl < current_sl - 0.01:
                                if alpaca.replace_stop(t, new_sl):
                                    alpaca.pos_cache[t]['sl'] = new_sl
                                    logging.info(f"🔒 TRAIL {t}: SL -> ${new_sl:.2f} (+{r_val:.2f}R)")
                    # FIX: Only pyramid if enabled (disabled by default - bracket stacking is messy)
                    if SETTINGS.get('ENABLE_PYRAMIDING', False) and r_val > SETTINGS['PYRAMID_R'] and not p.get('pyramided'):
                        add_qty = int(p['qty'] * 0.5)
                        if add_qty > 0:
                            ext_tp = curr + (4.0 * atr) if side == 'LONG' else curr - (4.0 * atr)
                            # FIX: Pass atr_override to preserve correct ATR for pyramid
                            # FIX: Use CURRENT STOP LOSS (current_sl) not ENTRY for new bracket stop
                            if alpaca.submit_bracket(t, side, add_qty, curr, current_sl, ext_tp, atr_override=atr):
                                with alpaca._pos_lock:  # FIX: Thread safety
                                    alpaca.pos_cache[t]['pyramided'] = True

                except Exception as e:
                    ERROR_TRACKER.record_failure(f"Manage_{t}", str(e))

            if (datetime.datetime.now() - last_equity_log).total_seconds() > 300:
                db.log_equity(alpaca.equity)
                last_equity_log = datetime.datetime.now()

            db.save_dashboard_state(alpaca.equity, regime, len(universe), hedger.is_hedged, vix)

            # Hybrid wait: WS bar-close event OR REST fallback timeout
            if bar_stream.is_connected:
                bar_stream.wait_for_bars(timeout=SETTINGS['SCAN_INTERVAL'])
                ws_ready = bar_stream.get_ready_tickers()
                if ws_ready:
                    # Invalidate scan cache for tickers with fresh bars
                    for _t in ws_ready:
                        scan_cache.invalidate(_t)
                    logging.debug(f"🔌 WS triggered scan for {len(ws_ready)} tickers")
            else:
                time.sleep(SETTINGS['SCAN_INTERVAL'])

        except KeyboardInterrupt:
            logging.info("\n🛑 Stopped")
            bar_stream.stop()
            break
        except Exception as e:
            ERROR_TRACKER.record_failure("MainLoop", str(e))
            logging.error(f"Loop: {e}\n{traceback.format_exc()}")
            time.sleep(60)

if __name__ == "__main__":
    try:
        run_god_mode()
    except KeyboardInterrupt:
        logging.info("🛑 KeyboardInterrupt - shutting down cleanly.")
    except Exception as e:
        import traceback
        logging.error(f"Fatal error in __main__: {e}\n{traceback.format_exc()}")
        raise

