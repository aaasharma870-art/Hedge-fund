# Hedge Fund Algorithmic Trading System

## Overview

This system is an intraday equity long/short trading engine that combines machine learning ensemble prediction (XGBoost + LightGBM + Ridge meta-learner) with institutional-quality microstructure signals, regime-conditional execution logic, and automated risk management. It operates on 15-minute bars across a focused universe of six US equities, executing bracket orders through the Alpaca brokerage API with Kelly-optimal position sizing, portfolio-level exposure constraints, and a four-layer risk architecture spanning signal quality through system-level circuit breakers.

The core research question is whether signals derived from market microstructure theory (order flow imbalance, adverse selection measures, cross-sectional momentum) and regime detection (variance ratio tests, realized volatility clustering) produce statistically significant edge on 15-minute equity bars when combined with ML-driven prediction and institutional-grade execution logic. The system tests the hypothesis that alpha exists not in public price-derived indicators (RSI, MACD, Bollinger Bands), which are fully arbitraged on liquid US equities, but in the information asymmetry revealed by volume-price dynamics and cross-sectional factor premiums documented in the academic literature.

The system is currently in paper trading validation. The most recent complete backtest produced an overall profit factor of 0.94 with six individually profitable tickers out of eighteen tested. The v9.0 architecture concentrates on those six tickers with institutional signal integration, regime-gated long/short execution, and portfolio-level book management. Deployment criteria (PF > 2.0, Sharpe > 1.5, MC p-value < 0.05) have not yet been met.

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

Live capital deployment requires all of the following gates to be met simultaneously: profit factor greater than 2.0 on the walk-forward test set, Sharpe ratio greater than 1.5, maximum drawdown less than 20% of peak equity in R-multiples, Monte Carlo p-value less than 0.05 (confirming statistical significance), and a 60-day paper trading period with information coefficient greater than 0.05 on live predictions. These criteria have not yet been met.

## Current Status and Honest Limitations

### What Has Been Demonstrated

The walk-forward engine runs correctly and produces real composite scores (0.19-0.46 range confirmed in live optimization runs with 150+ trials). Six tickers showed individually profitable signals in the most recent 18-ticker backtest: RKLB (PF 1.26), AMD (PF 1.13), GS (PF 1.10), COST (PF 1.06), GE (PF 1.03), ASTS (PF 1.02). The ML pipeline has no data leakage, confirmed by the test suite (225 tests passing). Live broker integration has been verified on Alpaca paper trading with bracket orders, stop replacement, position sync, and kill switch functionality. The ensemble model (XGBoost + LightGBM + Ridge) uses out-of-fold stacking to prevent meta-learner overfitting.

### Known Limitations

Signal quality remains below deployment criteria. The last complete backtest produced PF 0.94 overall, meaning the system lost money in aggregate despite profitable signals on six individual tickers. The twelve unprofitable tickers dragged portfolio returns negative, which motivated the v9.0 focus on the six verified names. Whether concentrating on fewer tickers improves or degrades risk-adjusted returns is an empirical question that the current backtest will answer.

15-minute equity bars are a competitive trading frequency. High-frequency trading firms and statistical arbitrage desks at Renaissance Technologies, D.E. Shaw, and Two Sigma are well-capitalized competitors operating at this timescale with superior data, infrastructure, and execution. The signals implemented here are academically grounded but may not survive after transaction costs in practice. The system models execution costs (0.03% spread + 0.02% impact per side) but does not model stock borrow costs for short positions, which can be material for high-short-interest names like RKLB and ASTS.

The model has not been tested through a full market cycle including a bear market or volatility crisis. The three-year training window (when Polygon data is available) covers 2021-2024, which includes the 2022 drawdown but not a recession or credit event. Regime detection may fail during market regimes not represented in the training data.

Short execution introduces asymmetric risk that is not fully captured in backtesting. Equity shorts have theoretically unlimited downside (short squeeze), stock borrow costs that vary by availability, and regulatory risk (short-sale restrictions during market stress). The system mitigates this with tighter stops on shorts (80% of long stop width), regime-gated short permissions (no shorts in volatile regime), and portfolio net exposure limits.

## Development Roadmap

Phase 1 (current): Institutional signal integration with regime-gated long/short execution and cross-sectional factor signals across a focused six-ticker universe.

Phase 2: Expanding the universe to 15-20 tickers through information coefficient pre-screening, retaining only tickers where the model demonstrates statistically significant predictive power.

Phase 3: Options overlay to hedge tail risk using put spreads and monetize elevated implied volatility through covered call writing on positions with positive theta characteristics.

Phase 4: Multi-strategy portfolio combining the equity long/short system with the options bot for diversified alpha generation across uncorrelated strategy types.

## Performance Record

| Date | Version | PF | Sharpe | MaxDD | Trades | Configuration |
|------|---------|-----|--------|-------|--------|---------------|
| 2024 | v6.3 | 0.94 | -0.12 | -18.4R | 847 | 18 tickers, 1h bars, 150 Optuna trials |
| 2025 | v9.0 | TBD | TBD | TBD | TBD | 6 tickers, 15m bars, 300 Optuna trials, institutional signals |

Per-ticker breakdown (v6.3):

| Ticker | PF | Trades |
|--------|-----|--------|
| RKLB | 1.26 | 52 |
| AMD | 1.13 | 48 |
| GS | 1.10 | 45 |
| COST | 1.06 | 44 |
| GE | 1.03 | 47 |
| ASTS | 1.02 | 41 |

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
