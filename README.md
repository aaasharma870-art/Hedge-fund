# Quantitative Algorithmic Trading System

A hedge-fund-grade quantitative trading system that combines machine learning signal generation, Bayesian parameter optimization, and institutional risk management to trade US equities.

**Two components work together:**
- **`backtester.py`** (V6.1) -- Optuna Bayesian optimizer that finds the best trading parameters across 18 diversified stocks using walk-forward validation, Monte Carlo significance testing, and out-of-sample holdout verification.
- **`bot.py`** (v14.3) -- Live trading bot that executes on Alpaca using XGBoost predictions, ATR-bracket trades, and multi-layered risk controls. Designed for Google Colab Pro+.

The backtester discovers the optimal parameters. The bot uses them to trade live.

---

## What This System Does

This is a **systematic quantitative trading system** -- the same type of approach used by quantitative hedge funds like Renaissance Technologies, Two Sigma, and DE Shaw. It replaces human judgment with statistical models and strict risk rules.

**The core loop:**
1. Download 3 years of hourly price data for 18 stocks across 6 sectors
2. Engineer 27 features per bar (technicals, microstructure, regime detection)
3. Train XGBoost models via walk-forward validation to predict trade outcomes
4. Use Optuna Bayesian optimization to find the filter/sizing parameters that maximize risk-adjusted returns
5. Validate the best parameters on holdout data the optimizer never saw
6. The live bot uses these parameters to execute real bracket trades on Alpaca

**What makes it "quant-level":**
- Walk-forward validation (no peeking at future data)
- Monte Carlo significance testing (is the edge real or random?)
- Out-of-sample holdout validation (does it work on unseen data?)
- Execution cost modeling in R-space (realistic slippage and spread)
- Regime-aware filtering (different rules for trending vs mean-reverting markets)
- Dynamic position sizing via Kelly criterion with shrinkage
- Equity-curve-aware risk scaling (MonteCarloGovernor reduces size during drawdowns)
- Partial profit taking (1/3 scale-out with breakeven stop)

---

## Project Structure

```
Hedge-fund/
├── backtester.py              # V6.1 Optuna Bayesian optimizer
├── bot.py                     # v14.3 Live trading bot (Colab Pro+)
├── hedge_fund/                # Shared quantitative library
│   ├── __init__.py            # Package exports
│   ├── indicators.py          # ManualTA: RSI, ATR, Bollinger Bands, ADX
│   ├── math_utils.py          # Kalman filter, Hurst exponent
│   ├── simulation.py          # Bracket trade simulation, ML label generation
│   ├── features.py            # VPIN, VWAP, Amihud, volatility regime, RRS,
│   │                          # liquidity sweeps, CrossSectionalRanker
│   ├── risk.py                # Position sizing, 3-outcome Kelly criterion,
│   │                          # OvernightGapModel, SlippageCalculator
│   ├── governance.py          # MonteCarloGovernor (DD-adaptive risk scalar)
│   ├── optimization.py        # PortfolioOptimizer (Mean-Variance + Ledoit-Wolf)
│   ├── analysis.py            # DecisionTree trade attribution analysis
│   ├── data.py                # Token-bucket rate limiter
│   └── objectives.py          # XGBoost custom loss functions
├── tests/                     # 108 unit tests
│   ├── test_backtester.py     # Risk metrics, Monte Carlo, per-ticker breakdown
│   ├── test_backtester_v6.py  # V6 stateful sim, partial profits, label buckets
│   ├── test_simulation.py     # Bracket exits, trailing stops, label generation
│   ├── test_indicators.py     # RSI, ATR, Bollinger Bands, ADX
│   ├── test_risk.py           # Position sizing, Kelly criterion
│   └── test_broker.py         # Mock Alpaca order submission
├── optuna_backtester.db       # Persistent Optuna study (auto-created)
└── .gitignore
```

---

## Setup

### Dependencies

```bash
pip install numpy pandas scipy xgboost scikit-learn rich requests optuna joblib
```

The live bot additionally requires:

```bash
pip install alpaca-trade-api yfinance websocket-client pandas_ta
```

### Environment Variables

Set API keys as environment variables (**never hardcode them**):

```bash
export POLYGON_API_KEY="your-polygon-key"       # Required for data
export FMP_API_KEY="your-fmp-key"                # Optional for fundamentals
export ALPACA_API_KEY="your-alpaca-key"           # Required for live bot
export ALPACA_SECRET_KEY="your-alpaca-secret"     # Required for live bot
```

---

## How to Run

### 1. Run the Backtester / Optimizer

```bash
python backtester.py
```

This runs the full optimization pipeline:

1. **Parallel data download** -- Fetches 3 years of hourly OHLCV from Polygon for 18 stocks across Tech, Financials, Healthcare, Energy, Industrials, and Consumer sectors (using 16 concurrent workers)
2. **Holdout split** -- Reserves the most recent ~3 months per ticker as final validation data that Optuna never sees
3. **Walk-forward training** -- For each of 4 SL/TP label buckets, trains XGBoost models on rolling 9-month windows and predicts on the next 3-month window. Labels are regenerated per bucket so the model's training targets match the simulation parameters
4. **Optuna Bayesian optimization** -- Runs 60 trials using the TPE (Tree-Parzen Estimator) sampler across 10 parameter dimensions: prediction threshold, stop-loss multiplier, reward-risk ratio, max bars held, trailing stop, Hurst limit, ADX minimum, filter mode, scale-out R, and regime filtering
5. **Results table** -- Ranks all configurations by composite score: `PF * WR * sqrt(N) / (1 + |MaxDD|)`
6. **Per-ticker breakdown** -- Shows which stocks are profitable and which are losing for the best config
7. **Monte Carlo significance test** -- Shuffles trade outcomes 1000 times to compute a p-value (probability the edge is random noise)
8. **Holdout validation** -- Evaluates the best config on the reserved out-of-sample data and reports PF decay percentage

### 2. Run the Live Bot

```bash
python bot.py
```

Designed for Google Colab Pro+ with GPU. The bot:

1. Connects to Alpaca paper trading API
2. Downloads historical data and trains XGBoost models
3. Enters a 60-second scan loop: predicts, ranks cross-sectionally, filters, and executes ATR-bracket trades
4. Manages positions with trailing stops, partial profit scaling, time stops, and overnight gap risk
5. Uses MonteCarloGovernor to dynamically scale risk based on equity drawdown
6. Uses PortfolioOptimizer for Mean-Variance weight allocation with beta constraints
7. Logs everything to SQLite database and rich console dashboard

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

108 tests covering indicators, simulation, risk math, broker order flow, stateful simulation, partial profits, and label bucket selection.

---

## How It Trades

### Signal Generation Pipeline

```
Raw OHLCV Data
    │
    ▼
Feature Engineering (27 features per bar)
    ├── Technical: RSI, ADX, ATR, Bollinger Bands, ROC
    ├── Trend: Kalman filter distance, EMA-50/200
    ├── Regime: Hurst exponent (trending vs mean-reverting)
    ├── Microstructure: VPIN (order flow toxicity), Amihud (illiquidity)
    ├── Institutional: VWAP Z-score/slope, liquidity sweeps
    ├── Volatility: Vol rank, vol surge, vol regime (GEX proxy)
    └── Temporal: Hour of day, day of week
    │
    ▼
XGBoost Regression Model
    ├── Predicts signed R-multiple for each bar
    ├── Positive R = LONG signal, Negative R = SHORT signal
    ├── Walk-forward trained (no data leakage)
    └── Feature pruning (drops bottom 50% by importance)
    │
    ▼
Entry Filters (10+ institutional gates)
    ├── VPIN < 0.85 (block toxic informed flow)
    ├── Amihud < 0.90 (block illiquid conditions)
    ├── Hurst regime check (trend-following in trending, reversion in ranging)
    ├── ADX > minimum (require directional strength)
    ├── VWAP Z-score bounds (block extreme deviations)
    ├── EMA-200 alignment (trade with the macro trend)
    ├── Volatility rank > 0.5 (avoid dead markets)
    └── GEX regime match (mean-reversion vs breakout)
    │
    ▼
Trade Execution
    ├── ATR-based bracket: SL at SL_MULT * ATR, TP at TP_MULT * ATR
    ├── Optional trailing stop at TRAIL_MULT * ATR
    ├── Partial profit: scale out 1/3 at SCALE_OUT_R, move stop to breakeven
    └── Position sized by Kelly criterion * Governor risk scalar
```

### Risk Management Stack

| Layer | Component | What It Does |
|-------|-----------|-------------|
| **Position** | Kelly Criterion | Optimal bet size from win rate and payoff ratio (0.35x shrinkage) |
| **Position** | SlippageCalculator | Deducts realistic execution costs (spread + impact) in R-space |
| **Portfolio** | MonteCarloGovernor | Scales all positions 0.5x-1.0x based on equity curve drawdown |
| **Portfolio** | PortfolioOptimizer | Mean-Variance allocation with Ledoit-Wolf covariance + beta <= 0.90 |
| **Portfolio** | Heat limit | Max 6% total equity at risk across all open positions |
| **Trade** | Partial profit | Lock 1/3 at 1.5R, move stop to breakeven for remaining 2/3 |
| **Trade** | Trailing stop | ATR-based trailing stop ratchets up with price |
| **Trade** | Time stop | Exit at max_bars if neither SL nor TP hit |
| **Overnight** | OvernightGapModel | Exits high-risk positions before market close (earnings, high VIX) |
| **System** | Kill switch | Shuts down at 10% account drawdown |

---

## How the Optimizer Finds Best Parameters

The backtester uses **Optuna Bayesian optimization** -- the same approach used by ML researchers at Google and Meta for hyperparameter tuning.

### Why Bayesian > Grid Search

| Approach | Trials Needed | Intelligence |
|----------|--------------|-------------|
| Grid search | Exponential (e.g., 10^10 for 10 params) | None -- tries everything |
| Random search | Many hundreds | None -- random sampling |
| **Bayesian (TPE)** | **60 trials** | **Learns which regions work, focuses search there** |

The TPE (Tree-Parzen Estimator) sampler builds a probabilistic model of the objective function. After each trial, it updates its belief about which parameter regions are promising and samples more densely from those regions.

### Search Space (10 dimensions)

| Parameter | Range | What It Controls |
|-----------|-------|-----------------|
| `pred_threshold` | 0.10 - 0.40 | Minimum prediction strength to take a trade |
| `sl_mult` | 1.0 - 2.5x ATR | Stop-loss distance |
| `tp_rr` | 1.5 - 3.5x SL | Reward-to-risk ratio |
| `max_bars` | 6 - 16 bars | Maximum holding period |
| `trail_mult` | 0.5 - 2.0x ATR | Trailing stop distance |
| `hurst_limit` | 0.35 - 0.65 | Hurst exponent filter threshold |
| `adx_min` | 15 - 30 | Minimum ADX for trend strength |
| `filter_mode` | STRICT/MODERATE/MINIMAL | How many entry gates are active |
| `scale_out_r` | 1.0 - 2.0 R | Partial profit level |
| `regime_hurst_filter` | True/False | Enable Hurst-based regime filtering |

### Label-Parameter Alignment

A critical correctness fix: the XGBoost model's training labels must match the simulation parameters. If Optuna tests SL=2.5x with a model trained on SL=1.5x labels, the predictions are misaligned.

**Solution:** Pre-compute 4 SL/TP label buckets:
- (SL=1.0, TP=2.0, MB=8)
- (SL=1.5, TP=3.0, MB=10)
- (SL=2.0, TP=4.0, MB=12)
- (SL=2.5, TP=5.0, MB=14)

Each Optuna trial automatically selects the closest bucket, ensuring label-simulation consistency.

### Overfitting Protection

| Guard | How It Works |
|-------|-------------|
| Walk-forward validation | Model never sees future data during training |
| Feature pruning | Drops the weakest 50% of features by importance |
| Overfitting dampener | If train R² - test R² > 0.15, predictions are dampened by 10% |
| Monte Carlo test | Shuffles outcomes to compute p-value of the observed profit factor |
| Holdout validation | Final 3 months of data reserved -- Optuna never touches it |
| Optuna persistence | Studies saved to SQLite so you can track optimization history across runs |

---

## Quant / Hedge-Fund-Level Features

### Microstructure Signals
- **VPIN** (Volume-Synchronized Probability of Informed Trading): Detects when institutional "smart money" is moving large orders. High VPIN = adverse selection risk. Based on Easley, Lopez de Prado, & O'Hara (2012).
- **Amihud Illiquidity Ratio**: Price impact per dollar traded. Flags low-liquidity conditions where slippage would eat the edge.
- **Liquidity Sweep Detection**: Identifies "stop hunts" where price briefly breaks a key level to trigger retail stops, then reverses. High win-rate mean-reversion signal.

### Regime Detection
- **Hurst Exponent**: H > 0.6 = trending (use momentum), H < 0.4 = mean-reverting (use reversion), 0.4-0.6 = random walk (be cautious). Different filter rules apply per regime.
- **Volatility Regime (GEX Proxy)**: Proxies gamma exposure using realized volatility patterns. Positive GEX = mean-reversion environment, Negative GEX = breakout environment.
- **Kalman Filter**: Recursive Bayesian state estimator for "true" price. Distance from Kalman = mean-reversion opportunity.

### Portfolio Construction
- **Mean-Variance Optimization**: Maximizes Sharpe Ratio subject to constraints, using Ledoit-Wolf shrinkage for stable covariance estimation with limited data.
- **Beta Constraint**: Portfolio beta capped at 0.90 to limit systematic market exposure.
- **Cross-Sectional Ranking**: Ranks all stocks by momentum (50%), volume (25%), and value (25%) to prioritize the highest-conviction trades.

### Risk Governance
- **MonteCarloGovernor**: Tracks the equity curve of recent trades and reduces position sizes by 25-50% during drawdowns. Automatically restores full sizing when performance recovers.
- **3-Outcome Kelly**: Standard Kelly assumes win/loss. This version handles a third outcome (timeout/scratch trades) for more realistic bet sizing.
- **Overnight Gap Model**: Before 3:45 PM ET, evaluates gap risk for each position. Exits positions with high VIX + low PnL + earnings risk. Tightens stops on the rest.

---

## Universe

18 stocks across 6 sectors for diversification:

| Sector | Tickers |
|--------|---------|
| Tech / Growth | NVDA, PLTR, TSLA, AMD, MSFT, META |
| Small-cap Momentum | RKLB, ASTS |
| Financials | JPM, GS |
| Healthcare | UNH, LLY |
| Energy | XOM, CVX |
| Industrials | CAT, GE |
| Consumer | AMZN, COST |

---

## Future Improvements

### High Priority
1. **Expand label buckets** -- Add more SL/TP combinations or make labels fully dynamic per Optuna trial for tighter alignment
2. **Multi-objective Optuna** -- Optimize PF and MaxDD as separate objectives using `optuna.create_study(directions=["maximize", "minimize"])` for a Pareto frontier
3. **Anchored walk-forward** -- Use expanding (anchored) training windows instead of fixed-width for better use of historical data
4. **Live parameter sync** -- Automatically push the best backtester parameters to the live bot configuration

### Medium Priority
5. **Continue extracting bot.py** -- Break the remaining ~4000 lines into focused modules: `hedge_fund/broker.py`, `hedge_fund/scanner.py`, `hedge_fund/websocket.py`
6. **Options data integration** -- Replace the GEX proxy with real gamma exposure from options chains for more accurate regime detection
7. **Alternative data** -- Add sentiment features from news/social media, earnings surprise data, or short interest
8. **Ensemble models** -- Stack XGBoost with LightGBM and a linear model for more robust predictions

### Low Priority
9. **Feature importance visualization** -- Save Optuna's parameter importance plots and walk-forward feature importance to files
10. **Multi-timeframe signals** -- Combine hourly bars with daily and 15-minute for multi-resolution features
11. **Futures/crypto expansion** -- Extend the framework to trade ES futures or crypto for 24/7 market coverage
12. **Cloud deployment** -- Migrate from Colab to a dedicated cloud VM with monitoring and alerting

---

## Package API

The `hedge_fund` package can be imported directly for custom strategies:

```python
from hedge_fund.indicators import ManualTA
from hedge_fund.math_utils import get_kalman_filter, get_hurst
from hedge_fund.simulation import simulate_exit, compute_bracket_labels
from hedge_fund.features import calculate_vpin, calculate_amihud_illiquidity
from hedge_fund.risk import calculate_position_size, kelly_criterion, SlippageCalculator
from hedge_fund.governance import MonteCarloGovernor
from hedge_fund.optimization import PortfolioOptimizer

# Compute RSI without pandas_ta
rsi = ManualTA.rsi(df['Close'], length=14)

# Kalman-filtered price
kalman = get_kalman_filter(df['Close'].values)

# Simulate a bracket trade with trailing stop
outcome, exit_price = simulate_exit(
    highs, lows, sl=95.0, tp=110.0, side='LONG', trail_dist=2.0
)

# Risk-based position sizing
qty = calculate_position_size(
    equity=100_000, entry_price=150.0, stop_price=145.0
)

# 3-outcome Kelly criterion
kelly_f = kelly_criterion(
    win_rate=0.55, avg_win_r=2.0, avg_loss_r=1.0,
    timeout_rate=0.10, avg_timeout_r=-0.1, shrinkage=0.35
)

# Execution cost in R-multiples
slip = SlippageCalculator(spread_pct=0.03, impact_pct=0.02)
cost_r = slip.cost_in_r(entry_price=150.0, sl_distance=3.0)

# Drawdown-adaptive risk scaling
gov = MonteCarloGovernor(dd_warning=0.05, dd_critical=0.08)
gov.add_trade(pnl=150.0, risk_dollars=100.0, side='LONG')
gov.apply_adjustments()
scalar = gov.get_risk_scalar()  # 0.5 - 1.0
```

---

## Risk Warning

This system is for **paper trading and educational purposes only**. Past backtested performance does not guarantee future results. Backtests are subject to survivorship bias, data-snooping bias, and execution assumptions that may not hold in live markets. Do not trade real money without independent validation across broader markets and time periods.
