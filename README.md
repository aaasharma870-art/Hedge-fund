# Algorithmic Trading System — v14.3

**A fully autonomous, institutional-grade equity trading engine built from scratch in Python.**

This project is a complete algorithmic hedge fund system — not a simple moving-average crossover script, but a 9,000+ line production codebase that integrates machine learning, quantitative risk management, portfolio optimization, and real-time market execution into a unified architecture. Every component, from signal generation to position sizing, was designed, implemented, and tested by hand.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Quantitative Features in Depth](#quantitative-features-in-depth)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Risk Management Framework](#risk-management-framework)
6. [Trading Strategy](#trading-strategy)
7. [Technology Stack](#technology-stack)
8. [Strengths](#strengths)
9. [Limitations and Honest Critique](#limitations-and-honest-critique)
10. [Roadmap for Improvement](#roadmap-for-improvement)
11. [Installation and Usage](#installation-and-usage)
12. [Testing](#testing)
13. [Project Structure](#project-structure)

---

## Project Overview

This system trades US equities through the Alpaca brokerage API, making autonomous buy and sell decisions on a 15-minute cadence throughout market hours. At its core, the system answers a deceptively simple question: *given the current state of the market, should I buy, sell, or do nothing — and how much should I risk?*

To answer that question rigorously, the system draws on concepts from:

- **Statistical learning theory** (gradient-boosted ensemble models with walk-forward validation)
- **Stochastic processes** (Hurst exponent for regime detection, Kalman filtering for trend estimation)
- **Market microstructure** (Volume-Synchronized Probability of Informed Trading, Amihud illiquidity ratios)
- **Portfolio theory** (Markowitz mean-variance optimization with Ledoit-Wolf shrinkage)
- **Information theory** (Kelly Criterion for optimal position sizing)
- **Monte Carlo simulation** (equity-curve-aware dynamic risk scaling)

The defining architectural principle is what I call the **"One Brain" design** — the exact same code that generates signals in backtesting generates signals in live trading. There is no separate "backtest version" and "live version." This eliminates an entire class of bugs where a strategy looks profitable in testing but behaves differently in production.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SHARED CORE (hedge_fund/)                │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ Features │  │ Ensemble │  │  Simulation│  │  Indicators  │  │
│  │ (27 eng.)│  │ (XGB+LGB │  │  (Bracket  │  │  (RSI, ATR,  │  │
│  │          │  │  +Ridge)  │  │   Exits)   │  │  Bollinger)  │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └──────┬───────┘  │
│       │              │              │                 │          │
│  ┌────┴──────────────┴──────────────┴─────────────────┴───────┐  │
│  │                    Risk & Governance                       │  │
│  │  Kelly Criterion · Monte Carlo Governor · Gap Model        │  │
│  │  Slippage Calculator · Portfolio Optimizer (MVO)           │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────┴───────────────────────────────────┐  │
│  │              Scanner (10+ Institutional Gates)             │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────┬───────────────────┘
                        │                     │
              ┌─────────┴──────┐    ┌─────────┴──────┐
              │   backtester   │    │     bot.py     │
              │   (Optimize)   │    │  (Live Trade)  │
              │  Walk-Forward  │    │  Real-Time     │
              │  Optuna Trials │    │  WebSocket     │
              │  Monte Carlo   │    │  Bracket Exec  │
              │  Significance  │    │  Dashboard     │
              └────────────────┘    └────────────────┘
```

The `hedge_fund/` package contains 19 modules totaling thousands of lines, all importable by both the backtester and the live trading engine. This modular design means every component can be unit-tested in isolation, and changes propagate identically to both environments.

---

## Quantitative Features in Depth

This section explains the mathematical and financial concepts used throughout the system. Each feature was chosen for a specific theoretical reason — not because it "looked good on a chart."

### Signal Generation: 27 Engineered Features

The ML model does not receive raw price data. Instead, it receives 27 carefully engineered features that encode different aspects of market behavior:

| Category | Features | What They Capture |
|---|---|---|
| **Technical** | RSI, ADX, ATR, Bollinger Band width, VWAP Z-score | Momentum, trend strength, volatility, mean-reversion signals |
| **Microstructure** | VPIN (toxic flow), Amihud Illiquidity | Whether institutional ("smart money") traders are active, and how costly it is to trade |
| **Regime** | Hurst Exponent, Kalman Filter slope, GEX Proxy | Whether the market is trending, mean-reverting, or random — and the likely influence of options market-makers |
| **Cross-Sectional** | Relative strength rank, volume surge ratio | How a stock compares to its peers right now |
| **Temporal** | Hour of day, day of week | Well-documented intraday seasonality patterns (e.g., the "power hour" before close) |

### Key Concepts Explained

**Hurst Exponent (H):** A measure from fractal geometry that classifies time series behavior. When H < 0.5, prices tend to revert to the mean. When H > 0.5, prices tend to trend. The system dynamically switches between mean-reversion and momentum strategies based on H. This is computed using a rescaled range (R/S) analysis over rolling windows — the same technique used by Benoit Mandelbrot to study Nile River flooding patterns.

**VPIN (Volume-Synchronized Probability of Informed Trading):** Developed by Marcos López de Prado and others, VPIN estimates the fraction of trading volume likely driven by informed traders (those with non-public information). When VPIN is high (>85th percentile), the system avoids entering new positions because the risk of trading against someone who knows something you don't is elevated. The implementation uses Bulk Volume Classification (BVC), which infers trade direction from price changes rather than requiring tick-level data.

**Kalman Filter:** A recursive Bayesian estimator originally developed for NASA's Apollo navigation systems. Here, it estimates the "true" slope of price movement while filtering out noise. Unlike a simple moving average, the Kalman Filter adapts its smoothing dynamically based on how noisy recent observations have been.

**Kelly Criterion:** An information-theoretic formula for optimal bet sizing, originally derived by John Kelly at Bell Labs in 1956. Given a win rate and average win/loss size, Kelly tells you the fraction of capital to risk that maximizes long-term growth. The system implements a 3-outcome variant (win, loss, timeout) and applies a fractional Kelly multiplier (typically 0.3-0.5x) for additional conservatism.

**Mean-Variance Optimization (MVO):** Harry Markowitz's Nobel Prize-winning framework for portfolio construction. Given expected returns and a covariance matrix, MVO finds the allocation that maximizes the Sharpe ratio (return per unit of risk). The system uses **Ledoit-Wolf shrinkage** on the covariance matrix — a technique that blends the sample covariance with a structured estimator to reduce estimation error when the number of assets is large relative to the number of observations.

**Monte Carlo Governor:** A custom risk-scaling mechanism that monitors the simulated equity curve of the most recent 50 trades. It detects drawdown states and automatically reduces position sizes: 100% risk in normal conditions, 75% during warnings (5-8% drawdown), and 50% during critical states (>8% drawdown). This prevents the common failure mode where a losing streak leads to oversized bets on "recovery" trades.

---

## Machine Learning Pipeline

### Model Architecture: Stacked Ensemble

The system does not rely on a single model. Instead, it uses an ensemble of three diverse base learners combined by a meta-learner:

```
Input Features (27)
       │
       ├──→ XGBoost (100 trees, depth=2)    ──→ Prediction A
       ├──→ LightGBM (100 trees, depth=3)   ──→ Prediction B
       ├──→ Ridge Regression (alpha=1.0)     ──→ Prediction C
       │
       └──→ Ridge Meta-Learner ──→ Final Prediction (R-multiple)
```

Each base learner sees the same features but learns different patterns due to architectural differences. XGBoost uses exact greedy splits, LightGBM uses histogram-based leaf-wise growth, and Ridge captures linear relationships the tree models might miss. The meta-learner (Ridge regression) learns the optimal weighting of these three perspectives using out-of-fold predictions to prevent data leakage.

### Walk-Forward Validation

The system never trains on data it later tests on. It uses **anchored walk-forward analysis**:

```
Window 1: Train [bar 0 ──────── bar 1500]  Test [bar 1500 ── bar 2000]
Window 2: Train [bar 0 ──────────────── bar 2000]  Test [bar 2000 ── bar 2500]
Window 3: Train [bar 0 ──────────────────────── bar 2500]  Test [bar 2500 ── bar 3000]
                 ↑ Training always starts from bar 0 (anchored)
```

This simulates what would happen if you deployed the model in real-time: at each point, the model only knows what has happened in the past. The expanding (anchored) window means the model accumulates more training data over time, similar to how a real system improves as it gathers more market history.

### Label Generation: Triple-Barrier Method

Rather than predicting whether a stock will go "up" or "down," the model predicts the expected R-multiple — how many units of risk a trade would return. Labels are generated using a bracket simulation:

- **Take-Profit barrier**: If price reaches +N × ATR, the trade returns +R
- **Stop-Loss barrier**: If price reaches -N × ATR, the trade returns -R
- **Timeout barrier**: If neither is hit within K bars, the trade is marked to market with a decay penalty

This produces continuous labels (e.g., +1.8R, -0.7R, +0.3R) rather than binary classification, giving the model much richer information to learn from.

### Hyperparameter Optimization

The system uses **Optuna with Tree-structured Parzen Estimator (TPE)** sampling — a Bayesian optimization algorithm that models the search space probabilistically and focuses trials on promising regions. It runs 60 trials, optimizing for the Pareto frontier between **Profit Factor** (total wins / total losses) and **Maximum Drawdown**. This multi-objective approach avoids the trap of maximizing returns at the cost of catastrophic risk.

---

## Risk Management Framework

Risk management is not an afterthought — it is the primary concern. The system implements multiple layers of protection:

### Layer 1: Position-Level Controls
- **Risk-based sizing**: Each position risks a fixed percentage of equity (default 1%), calculated as: `Quantity = (Equity × Risk%) / (Entry Price − Stop Price)`
- **Maximum position size**: No single position can exceed 20% of total equity
- **Slippage modeling**: Every simulated trade deducts 0.03% spread + 0.02% market impact

### Layer 2: Portfolio-Level Controls
- **Portfolio heat limit**: Total concurrent open risk capped at 5% of equity
- **Beta constraint**: Portfolio optimizer enforces beta ≤ 0.90 to the broad market
- **VIX adjustment**: Position sizes shrink automatically when the CBOE Volatility Index spikes

### Layer 3: Event-Driven Guards
- **Earnings Guard**: Refuses to hold positions within two weeks of earnings announcements
- **Fundamental Guard**: Blocks trades on companies showing debt spirals or revenue collapse
- **Overnight Gap Model**: Widens stop-losses before market close based on historical pre-market volatility
- **News Sentiment Scoring**: Aggregates sentiment from Financial Modeling Prep and Seeking Alpha; downgrades conviction on negative news

### Layer 4: System-Level Protection
- **Monte Carlo Governor**: Dynamic risk scaling based on recent equity curve (described above)
- **Kill Switch**: Automatic liquidation of all positions if cumulative drawdown exceeds 8%
- **WebSocket heartbeat monitoring**: Detects data feed disconnections and halts trading

---

## Trading Strategy

The system uses a **dual-tier hybrid strategy** inspired by how institutional desks allocate capital:

### Tier 1 — The Specialist
High-conviction trades that meet strict criteria: model prediction exceeding 30 basis points of expected edge, technical alignment (RSI, ADX within optimal ranges), favorable regime (Hurst confirms directionality), and clean microstructure (low VPIN, adequate liquidity). These trades receive **2x standard risk allocation**. The Specialist may go days without triggering — and that is by design.

### Tier 2 — The Grinder
Moderate-conviction trades that pass core safety checks: prediction exceeding 20 basis points, basic liquidity requirements, and no fundamental red flags. These receive **1x standard risk allocation** and keep the equity curve compounding during periods when Tier 1 is inactive.

### Exit Management
- **Bracket orders**: Every entry has a pre-computed stop-loss and take-profit submitted simultaneously
- **Partial profit-taking**: At 1.5R, one-third of the position is closed and the stop moves to breakeven — locking in profit while letting the remainder run
- **Trailing stops**: Optional ATR-based trailing stop that ratchets upward as the trade progresses
- **Timeout exits**: Positions not resolved within a configurable number of bars are closed at market price

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Runtime | Python 3.11 | Core language |
| ML Models | XGBoost, LightGBM, Scikit-Learn | Ensemble prediction |
| Optimization | Optuna (Bayesian TPE) | Hyperparameter search |
| Data (Historical) | Polygon.io | OHLCV bars, corporate actions |
| Data (Real-Time) | Polygon WebSocket | Streaming 15-min bars with 93% API call reduction |
| Data (Fundamental) | Financial Modeling Prep | Earnings, balance sheets, news |
| Data (Volatility) | YFinance | VIX, fallback quotes |
| Brokerage | Alpaca Trade API | Order execution, position management |
| Portfolio Math | SciPy, NumPy | Optimization solvers, statistical distributions |
| Covariance | Ledoit-Wolf (Scikit-Learn) | Shrinkage estimator for portfolio construction |
| Database | SQLite | Trade history, analysis cache, Optuna storage |
| Concurrency | Threading (16 workers) | Parallel data fetching |
| UI | Rich | Real-time terminal dashboard |
| Deployment | Docker | Production containerization |
| Testing | Pytest (13 test suites) | Comprehensive unit and integration tests |

---

## Strengths

**1. Intellectual Rigor Over Shortcuts.**
Every quantitative decision has a theoretical foundation. The Hurst exponent is not included because it "might help" — it is there because Mandelbrot's work on long-range dependence provides a principled way to distinguish trending from mean-reverting markets. VPIN is not a custom indicator — it is a peer-reviewed measure of adverse selection risk. This distinction matters: the system is built on published research, not indicator mining.

**2. Walk-Forward Validation Prevents Overfitting.**
The most common failure in algorithmic trading is overfitting — building a model that perfectly explains the past but cannot predict the future. Walk-forward analysis with anchored windows simulates real deployment conditions. The system also runs a **Monte Carlo significance test** (1,000 shuffles of trade outcomes) to determine whether observed performance could be attributed to random chance, reporting a p-value alongside every backtest.

**3. Risk-First Design Philosophy.**
The Monte Carlo Governor, Kelly Criterion sizing, portfolio heat limits, and kill switch all serve the same principle: *survival is more important than returns*. A 50% drawdown requires a 100% gain to recover. By cutting risk dynamically during losing streaks, the system avoids the asymmetric math that destroys most trading systems.

**4. Production-Grade Engineering.**
This is not a Jupyter notebook experiment. It is a modular, tested, containerized system with 13 test suites, WebSocket-based real-time data, concurrent execution, and graceful failure handling. The "One Brain" architecture eliminates the gap between research and production that plagues many quantitative systems.

**5. Comprehensive Market Awareness.**
The system does not trade in a vacuum. It checks earnings calendars, monitors fundamental health, adjusts for overnight gap risk, tracks VIX, scores news sentiment, and measures informed trading pressure. Each of these "guard rails" prevents a specific category of loss that backtests alone would never reveal.

---

## Limitations and Honest Critique

No system is without weaknesses, and intellectual honesty demands acknowledging them:

**1. Execution Assumptions May Be Optimistic.**
While the slippage calculator models spread and market impact, real-world execution involves complexities that are difficult to simulate: partial fills, queue priority, delayed confirmations, and broker-side latency. The backtest assumes fills at modeled prices, which may not hold during volatile conditions or for less-liquid names.

**2. Regime Detection Is Retrospective.**
The Hurst exponent and volatility regime indicators are computed on trailing data. By the time the system classifies a regime change, the transition may already be underway. This creates a lag between the actual market shift and the system's response — a fundamental limitation of any lookback-based regime detector.

**3. Limited Universe and Asset Classes.**
The system currently trades only US equities through a single broker. It cannot trade options, futures, forex, or crypto. This limits diversification and means the system is inherently correlated with the US stock market, even with beta constraints.

**4. Single-Strategy Concentration.**
Despite the dual-tier structure, both tiers rely on the same underlying ML model and feature set. A systematic failure in the model (e.g., a structural market change that invalidates learned patterns) would affect all trades simultaneously. True institutional hedge funds run multiple uncorrelated strategies.

**5. No Live Track Record Yet.**
While the backtesting framework is rigorous and the Monte Carlo significance test provides statistical validation, no amount of historical testing can fully substitute for a verified live track record. The system has been paper-traded but has not been validated with real capital over a full market cycle.

**6. Data Dependency.**
The system relies on third-party APIs (Polygon, Alpaca, FMP) for its data. API outages, schema changes, or pricing tier modifications could disrupt operations. While fallbacks exist (e.g., YFinance for VIX), complete data loss would halt the system.

---

## Roadmap for Improvement

### Near-Term Enhancements
- **Alternative Data Integration**: Incorporate NLP-based sentiment analysis from earnings call transcripts, SEC filings (10-K/10-Q), and social media to capture information before it is reflected in price
- **Multi-Timeframe Confluence**: Train separate models on daily and hourly data, requiring agreement between timeframes before entry — reducing false signals
- **Adaptive Feature Selection**: Implement rolling feature importance tracking so the model can drop features that become stale and weight features that gain predictive power

### Medium-Term Expansions
- **Options Strategy Module**: Introduce defined-risk options strategies (vertical spreads, iron condors) for income generation and tail-risk hedging
- **Macro Regime Overlay**: Integrate Federal Reserve rate decisions, yield curve data, and inflation expectations to adjust the system's overall aggressiveness based on the macroeconomic environment
- **Multi-Broker Execution**: Support additional brokerages (Interactive Brokers, TD Ameritrade) for redundancy and better execution quality
- **Reinforcement Learning**: Explore RL-based position management where the agent learns optimal exit timing through interaction with a simulated market environment

### Long-Term Vision
- **Low-Latency Execution Engine**: Rewrite the execution layer in Rust or C++ for sub-millisecond order routing, enabling the system to compete in faster timeframes
- **Multi-Asset Universe**: Extend to futures, forex, and cryptocurrency markets for true cross-asset diversification
- **Distributed Backtesting**: Parallelize walk-forward optimization across GPU clusters to test thousands of parameter combinations in minutes rather than hours
- **Live Performance Dashboard**: Build a web-based monitoring interface with real-time P&L, drawdown charts, and model confidence visualization

---

## Installation and Usage

### Prerequisites
- Python 3.11+
- API keys for Polygon.io, Alpaca, and Financial Modeling Prep

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
POLYGON_API_KEY=your_polygon_key
FMP_API_KEY=your_fmp_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

### 3. Run the Backtester
The backtester performs walk-forward optimization to find optimal parameters:
```bash
python backtester.py
```
This trains the ML ensemble, runs Optuna trials, and outputs optimal parameters to `data/optimal_params.json`.

### 4. Run the Live Trading Bot
```bash
python bot.py
```
The bot loads optimized parameters, connects to market data via WebSocket, and begins autonomous trading during market hours.

### 5. Docker Deployment
```bash
docker build -t hedge-fund .
docker run -d --env-file .env hedge-fund
```

---

## Testing

The project includes 13 test suites covering all major subsystems:

```bash
pytest tests/ -v
```

| Test Suite | Coverage |
|---|---|
| `test_backtester_v6.py` | Walk-forward training, label buckets, partial profits |
| `test_ensemble.py` | Stacking pipeline, out-of-fold predictions, cross-validation |
| `test_risk.py` | Kelly criterion, position sizing, overnight gap model |
| `test_scanner.py` | Candidate filtering across all 10+ gates |
| `test_broker.py` | Alpaca integration, order lifecycle, position sync |
| `test_indicators.py` | RSI, ATR, Bollinger Band, ADX accuracy |
| `test_simulation.py` | Bracket exit logic, timeout handling, partial exits |
| `test_websocket.py` | Bar caching, stream reliability, reconnection |
| `test_data_providers.py` | API integration, rate limiting, error handling |
| `test_config.py` | Parameter persistence and loading |

All external APIs are mocked in tests to ensure deterministic, repeatable results.

---

## Project Structure

```
Hedge-fund/
├── bot.py                    # Live trading engine (5,100+ lines)
├── backtester.py             # Walk-forward optimizer (1,500+ lines)
├── hedge_fund/               # Shared core library
│   ├── ensemble.py           #   Stacked ML ensemble (XGB + LGB + Ridge)
│   ├── features.py           #   27-feature engineering pipeline
│   ├── indicators.py         #   Technical analysis (RSI, ATR, Bollinger, ADX)
│   ├── math_utils.py         #   Kalman filter, Hurst exponent
│   ├── simulation.py         #   Bracket exit simulation & label generation
│   ├── risk.py               #   Kelly criterion, position sizing, gap model
│   ├── governance.py         #   Monte Carlo Governor (dynamic risk scaling)
│   ├── optimization.py       #   Mean-variance portfolio optimizer
│   ├── scanner.py            #   Trade candidate filtering (10+ gates)
│   ├── broker.py             #   Alpaca API wrapper & order management
│   ├── data_providers.py     #   Multi-source data (Polygon, FMP, VIX)
│   ├── data.py               #   Rate limiting & caching layer
│   ├── config.py             #   Configuration & parameter persistence
│   ├── dashboard.py          #   Real-time terminal UI
│   ├── websocket.py          #   Polygon WebSocket bar stream
│   ├── analysis.py           #   Performance attribution
│   ├── objectives.py         #   Custom XGBoost loss functions
│   └── reliability.py        #   Failure classification & monitoring
├── tests/                    # 13 comprehensive test suites
├── docker/                   # Container configuration
├── Dockerfile                # Production deployment
├── requirements.txt          # Python dependencies
└── verify_keys.py            # API key validation utility
```

---

*Built as an exercise in applied quantitative finance, software engineering, and machine learning systems design.*
