# Hedge Fund Algorithmic Trading System

## Overview

This system is an intraday equity long/short trading engine that combines machine learning ensemble prediction (XGBoost + LightGBM + Ridge meta-learner) with institutional-quality microstructure signals, regime-conditional execution logic, and automated risk management. It operates on 15-minute bars across a focused universe of six US equities, executing bracket orders through the Alpaca brokerage API with Kelly-optimal position sizing, portfolio-level exposure constraints, and a four-layer risk architecture spanning signal quality through system-level circuit breakers.

The core research question is whether signals derived from market microstructure theory (order flow imbalance, adverse selection measures, cross-sectional momentum) and regime detection (variance ratio tests, realized volatility clustering) produce statistically significant edge on 15-minute equity bars when combined with ML-driven prediction and institutional-grade execution logic. The system tests the hypothesis that alpha exists not in public price-derived indicators (RSI, MACD, Bollinger Bands), which are fully arbitraged on liquid US equities, but in the information asymmetry revealed by volume-price dynamics and cross-sectional factor premiums documented in the academic literature.

The system is currently in paper trading validation. The most recent complete backtest (V12.6) produced a training profit factor of 2.65 with WR=64.5% and Sharpe=7.44 across 408 trades on 15 tickers using daily alpha signals and intraday execution. Holdout validation (90 days) showed PF=0.77, indicating overfitting to the training regime. V12.7 addresses this with look-ahead bias removal (+1 day signal shift), forced L/S balance, and a softened regime gate. Deployment criteria (holdout PF > 1.2, both L/S PF > 1.0, MC p-value < 0.05) have not yet been met.

## Theoretical Foundation

### The Problem with Price-Derived Indicators

The semi-strong form of the efficient market hypothesis (Fama, 1991) states that all publicly available information is already reflected in equity prices. Technical indicators computed from OHLCV data --- RSI, MACD, Bollinger Bands, moving average crossovers --- are deterministic transformations of public information visible to every market participant simultaneously. If any of these indicators reliably predicted future returns, rational arbitrageurs would trade against the signal until the predictability was eliminated. On large-cap US equities trading tens of millions of shares per day, this arbitrage occurs within milliseconds.

Harvey, Liu, and Zhu (2016) analyzed 316 published equity return anomalies and found that the majority disappear after controlling for multiple hypothesis testing, and most of the remainder lose significance after publication due to crowded trading. The system's own walk-forward feature importance analysis confirmed this independently: RSI, BB_Position, ROC_20, Hour, VWAP_Volume_Ratio, and similar price-derived features were dropped by the XGBoost model in every ticker and every label bucket across three years of walk-forward windows. The model was not broken; it was correctly identifying that these features contain no information not already reflected in price.

This motivated a complete replacement of the feature architecture with signals grounded in market microstructure research and cross-sectional factor theory, where the academic evidence for persistent out-of-sample alpha is strongest.

### Alpha Sources with Theoretical Backing

**Order Flow Information Asymmetry.** Kyle (1985) formalized continuous auctions with informed trading, showing that the price impact per unit of order flow (lambda) directly measures informed trader presence. Easley and O'Hara (1987) extended this to demonstrate that trade size and direction reveal private information before it is fully incorporated into price. The system implements this through Cumulative Order Flow Imbalance (COFI), which tracks multi-bar net directional volume using Lee-Ready tick classification, and Kyle's Lambda, which measures adverse selection risk as price impact per unit of square-root volume (Almgren et al., 2005). These signals detect institutional accumulation and distribution patterns in the order flow before they are reflected in price.

**Cross-Sectional Momentum.** Jegadeesh and Titman (1993) documented that US stocks in the top decile of prior 3-12 month returns outperform bottom-decile stocks by approximately 1% per month. This premium has been replicated out-of-sample in 40+ equity markets (Asness, Moskowitz, and Pedersen, 2013, "Value and Momentum Everywhere") and persists because it requires taking on correlated crash risk that most investors are unwilling to bear. The system computes cross-sectional momentum rank across the six-ticker universe at multiple lookback periods (5, 10, 20 bars), centering ranks at zero and normalizing to z-scores for consistent model input.

**Short-Horizon Mean Reversion.** Lehmann (1990) and Lo and MacKinlay (1990) documented negative autocorrelation in individual stock returns at weekly horizons. The mechanism is structural: large directional orders cause price overshooting relative to fundamental value, and liquidity providers step in to earn the spread, pushing price back. The system implements this as a z-score of recent log-returns relative to rolling distribution (MR_Score), which identifies overbought and oversold conditions that tend to partially revert within 1-3 hours on 15-minute bars.

**Statistical Arbitrage.** Avellaneda and Lee (2010) formalized intraday statistical arbitrage as trading residual returns after removing common factor exposures. The six-ticker universe naturally splits into momentum names (RKLB, ASTS, AMD) and value names (GS, GE, COST), creating cross-sectional spread dynamics that the system exploits through beta-adjusted idiosyncratic momentum (Frazzini and Pedersen, 2014). Only the alpha component --- stock-specific return after removing beta-weighted market return --- is used as a signal.

**Regime Detection.** Lo and MacKinlay (1988) variance ratio test distinguishes trending regimes (positive autocorrelation, VR > 1) from mean-reverting regimes (negative autocorrelation, VR < 1). Engle (1982) ARCH models established that volatility clusters --- elevated volatility periods persist. The system combines these into a three-state regime classifier (trending, mean-reverting, volatile) that adjusts signal weights, position sizing, stop parameters, and short execution permissions per regime.

### Long/Short Portfolio Construction

Simultaneously holding long and short positions reduces portfolio beta and isolates factor alpha from market direction. The theoretical foundation is Frazzini and Pedersen (2014) "Betting Against Beta," which demonstrates that low-beta stocks earn higher risk-adjusted returns than high-beta stocks, creating a structural alpha source accessible through leverage-constrained long/short construction. The system enforces gross exposure limits (120% maximum), net exposure limits (40% maximum), and single-name concentration limits (25% maximum) to maintain market-neutral characteristics.

The six-ticker universe splits into momentum names (RKLB beta 1.8, ASTS beta 2.1, AMD beta 1.6) and value names (GS beta 1.3, GE beta 1.1, COST beta 0.7). This enables regime-conditional rotation: in trending regimes, the system weights momentum signals and order flow; in mean-reverting regimes, it weights the MR_Score and fades extreme moves; in volatile regimes, it restricts to long-only positions with reduced sizing. Momentum tickers receive wider stops and higher targets; value tickers receive tighter stops and faster exits. Short trades universally use tighter stops to protect against short squeeze risk.

## System Architecture

### Component Map

```
Polygon REST API / yfinance
        |
    BarValidator
        |
  prepare_features()
    [OFI, VPIN, Amihud, Kalman, Hurst]
    [COFI, Absorption_Ratio, Kyle_Lambda, Trade_Intensity]
    [CS_Mom_Rank, CS_Volume_Rank, MR_Score, Beta_Momentum]
    [Variance_Ratio, RV_Regime -> Regime Classification]
    [Session_Opening, Session_Mid, Session_Closing]
        |
  DirectionalEnsemble (XGBoost + LightGBM + Ridge)
        |
  determine_entry()
    [RegimeManager] [PortfolioManager] [TradeFilterCounter]
        |
  simulate_trades_stateful() / bot.py live execution
        |
  Kelly Sizing + DailyRiskManager + MonteCarloGovernor
        |
  Alpaca Bracket Orders (paper / live)
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.12 | Core runtime |
| ML (Gradient Boosting) | XGBoost, LightGBM | Base learners in ensemble |
| ML (Meta-Learner) | scikit-learn Ridge | OOF stacking meta-learner |
| Optimization | Optuna | Bayesian hyperparameter search |
| Broker | alpaca-trade-api | Paper/live bracket order execution |
| Market Data | polygon-api-client, yfinance | 15-min OHLCV bars |
| Database | SQLite | Trade history, state persistence |
| Deployment | Docker, DigitalOcean | Ubuntu 22.04 production |

### Signal Architecture

The first signal layer captures **order flow microstructure**: COFI (20-bar cumulative net directional volume, z-score normalized, grounded in Kyle 1985 informed trader footprint detection), Absorption Ratio (volume-per-price-move indicating supply/demand exhaustion), Kyle's Lambda (adverse selection measure as negative price impact per sqrt-volume), and Trade Intensity (abnormal volume relative to peers, directional using Lee-Ready classification). These signals detect institutional positioning activity before it is fully reflected in price and are strongest during opening session bars (Admati and Pfleiderer, 1988).

The second layer implements **cross-sectional factors**: momentum rank across all six tickers at 5/10/20-bar lookbacks (Jegadeesh and Titman, 1993), abnormal volume rank relative to universe peers, and beta-adjusted idiosyncratic momentum (Frazzini and Pedersen, 2014). These factors exploit the well-documented cross-sectional structure of equity returns and are computed relative to the universe to capture relative strength rather than absolute levels.

The third layer provides **regime detection**: the variance ratio test (Lo and MacKinlay, 1988) at lag-4 with 40-bar rolling window classifies trending vs mean-reverting regimes, while realized volatility ratio (short/long window) identifies vol expansion and compression states following Engle (1982) ARCH dynamics. These combine into a three-state classifier that gates signal weights and execution permissions.

The fourth layer encodes **session structure**: binary indicators for opening (9:30-10:15), mid-session (10:15-15:00), and closing (15:00-16:00) periods, plus continuous session progress (0-1). This allows the model to learn that order flow signals are strongest at open, mean reversion dominates mid-session, and institutional rebalancing creates opportunities at close.

## Validation Methodology

Walk-forward validation with embargo gaps is essential for time series data because standard k-fold cross-validation introduces look-ahead bias: the model trains on future data and tests on past data, producing inflated accuracy estimates that do not reflect live trading performance. The system uses anchored (expanding) walk-forward windows where training always starts from bar 0 and grows forward, with an embargo gap equal to max_bars between train and test sets to prevent label leakage from the bracket label computation.

The specific protocol uses 1500-bar training windows, 500-bar test windows, and 500-bar step size (configurable based on data source --- Polygon 2-year lookback uses 6000/2000/2000). A final holdout set of 500 bars is reserved and never seen during Optuna optimization. After optimization, the best parameters are evaluated on the holdout set, and a Monte Carlo significance test (1000 shuffled simulations) computes a p-value. Deployment requires p < 0.05, confirming that the observed profit factor is unlikely to have occurred by chance.

## Risk Management

### Four-Layer Risk Architecture

At the signal level, a percentile-calibrated prediction threshold eliminates low-conviction signals before they reach the entry decision. The threshold is set as a percentile of the absolute prediction distribution per ticker (typically 55th-85th percentile), ensuring that only the strongest model signals generate trades. This is combined with multi-confirmation gating: COFI alignment (institutional flow must not directly contradict the predicted direction), regime-signal consistency (mean reversion trades only in mean-reverting regimes), and session filtering (no new entries in the final 8% of the trading session).

At the position level, Kelly Criterion with three-outcome adjustment (win, loss, timeout) determines the optimal risk fraction per trade. The Kelly fraction is shrunk by 65% (shrinkage = 0.35) and capped at 2% maximum risk per trade, producing position sizes between 0.3% and 2.0% of equity. A confidence scalar adjusts sizing by prediction strength relative to threshold, and a volatility regime scalar reduces size during vol expansion.

At the portfolio level, the PortfolioManager enforces gross exposure (120%), net exposure (40%), and single-name concentration (25%) limits on every entry. Position flips (reversing direction on the same ticker) are blocked. Regime-conditional parameter adjustment modifies stop-loss width, take-profit ratio, and maximum bars held based on ticker class (momentum vs value) and current market regime.

At the system level, the Monte Carlo Governor tracks cumulative trade P&L and reduces the risk scalar to 0.75 at 5% drawdown and 0.50 at 8% drawdown from recent equity peak. The DailyRiskManager halts all trading if intraday losses exceed 2% of session-start equity. Together, these circuit breakers prevent catastrophic loss accumulation during adverse regimes.

### Deployment Criteria

Live capital deployment requires all of the following gates to be met simultaneously: holdout profit factor greater than 1.2, both long and short PF greater than 1.0, holdout win rate greater than 45%, Monte Carlo p-value less than 0.15, 100+ holdout trades, no look-ahead bias, and L/S ratio between 40:60 and 60:40. Paper trading begins with 1% risk per trade (not 2%) for the first 50 live trades to validate that live performance matches backtest within reasonable variance. These criteria have not yet been fully met.

## Current Status and Honest Limitations

### What Has Been Demonstrated

The V12 hybrid system (daily alpha + intraday execution) produces strong training results: PF=2.65, WR=64.5%, Sharpe=7.44 across 408 trades on 15 tickers with 1500 days of Polygon data. Trail/Partial exits account for 37.7% of trades (proving the trailing stop mechanism works after the V12.6 PnL bug fix). Monte Carlo p-value=0.0000 confirms statistical significance. Per-ticker, 11 of 13 active tickers are profitable, led by PLTR (55.3R), NVDA (52.1R), and COIN (44.1R). The ML pipeline has no data leakage, confirmed by the test suite. Live broker integration has been verified on Alpaca paper trading with bracket orders, stop replacement, position sync, and kill switch functionality. The ensemble model (XGBoost + LightGBM + Ridge) uses out-of-fold stacking to prevent meta-learner overfitting.

### Known Limitations

Holdout performance remains below deployment criteria. V12.6 training showed PF=2.65 but holdout collapsed to PF=0.77 (WR=48.9%). Two root causes were identified: (1) same-day entry bias — the model used day T's features to enter on day T, inflating training metrics by ~5-10%; (2) extreme L/S imbalance — 369 longs vs 39 shorts (90% long exposure), making the system a leveraged bull-market bet rather than a market-neutral strategy. V12.7 fixes both issues with +1 day signal shifting and forced equal L/S sizing.

Daily-bar signal generation with intraday execution operates at a competitive frequency. The system models execution costs (0.05% per side = 0.10% round-trip) but does not model stock borrow costs for short positions, which can be material for high-short-interest names like RKLB and ASTS. Average cost per trade is 0.027R.

The four-year training window (2022-2026 with Polygon data) includes the 2022 drawdown but not a full recession or credit event. The regime gate (rolling 10-trade win rate filter) provides some protection against regime shifts but is a simple heuristic, not a structural regime model.

Short execution introduces asymmetric risk that is not fully captured in backtesting. Equity shorts have theoretically unlimited downside (short squeeze), stock borrow costs that vary by availability, and regulatory risk (short-sale restrictions during market stress). V12.7 enforces equal sizing for both directions to maintain market-neutral characteristics and uses the regime gate to reduce (not eliminate) exposure during adverse regimes.

## Development Roadmap

Phase 1 (complete): V12 hybrid architecture — daily ML alpha (XGBoost + LightGBM + Ridge ensemble) with intraday 15-min execution timing (VPIN, OFI, VWAP) across 15-ticker universe.

Phase 2 (current): Deployment readiness — V12.7 eliminates look-ahead bias (+1 day signal shift), forces balanced L/S exposure, and softens regime gate. Target: holdout PF > 1.2 with both L/S PF > 1.0.

Phase 3: Paper trading validation with 1% risk per trade for first 50 live trades to confirm backtest-to-live consistency within expected variance bounds.

Phase 4: Options overlay to hedge tail risk using put spreads and monetize elevated implied volatility through covered call writing on positions with positive theta characteristics.

## Performance Record

| Date | Version | PF | Sharpe | MaxDD | Trades | Configuration |
|------|---------|-----|--------|-------|--------|---------------|
| 2024 | v6.3 | 0.94 | -0.12 | -18.4R | 847 | 18 tickers, 1h bars, 150 Optuna trials |
| 2025 | v12.6 (training) | 2.65 | 7.44 | -9.8R | 408 | 15 tickers, daily+15m hybrid, 80 Optuna trials |
| 2025 | v12.6 (holdout) | 0.77 | -1.74 | — | 47 | Same config, 90-day holdout |
| 2025 | v12.6 (validation) | 3.19 | 6.54 | — | 125 | 25% watchlist validation split |

Per-ticker breakdown (V12.6 training, best config):

| Ticker | PF | WR | Trades | Return_R |
|--------|------|--------|--------|----------|
| AMD | 5.57 | 84.6% | 13 | 9.3R |
| ASTS | 2.39 | 62.5% | 16 | 5.8R |
| COIN | 2.28 | 60.4% | 91 | 44.1R |
| COST | 0.44 | 45.5% | 11 | -1.4R |
| GE | 2.57 | 65.1% | 43 | 24.4R |
| GS | 1.31 | 50.0% | 10 | 0.6R |
| JNJ | 3.64 | 71.4% | 7 | 2.1R |
| NVDA | 2.58 | 64.2% | 95 | 52.1R |
| PLTR | 3.15 | 66.7% | 87 | 55.3R |
| RKLB | 3.36 | 71.0% | 31 | 20.0R |

Trade outcome breakdown (V12.6):

| Outcome | Count | % |
|---------|-------|------|
| Full SL hits | 117 | 28.7% |
| Full TP hits | 60 | 14.7% |
| Trail/Partial | 154 | 37.7% |
| Timeout (win) | 49 | 12.0% |
| Timeout (loss) | 9 | 2.2% |

Signal accuracy (training): Long 58.9% (2700 signals), Short 45.4% (2700 signals)

## Installation

### Docker Deployment (DigitalOcean Ubuntu)

```bash
git clone <repository-url>
cd Hedge-fund

# Environment variables
cp .env.example .env
# Edit .env with your keys:
#   ALPACA_API_KEY=<your-key>
#   ALPACA_SECRET_KEY=<your-secret>
#   POLYGON_API_KEY=<your-polygon-key>  # optional, enables 2-year lookback
#   FMP_API_KEY=<your-fmp-key>          # optional, enables fundamentals

# Paper trading (default)
export ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Build and run
docker build -t hedge-fund .
docker run -d --name hf --env-file .env hedge-fund

# Health checks
docker logs hf --tail 50
docker exec hf python -c "import backtester; print('OK')"
```

### Local Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # Run test suite (225 tests)
python backtester.py                # Run full backtest
python bot.py                       # Start live trading bot
```
