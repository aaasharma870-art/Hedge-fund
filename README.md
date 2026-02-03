# Hedge Fund

Quantitative algorithmic trading system with ML-driven signal generation, ATR-bracket trade execution, and institutional-grade risk management.

Two entry points:
- **`backtester.py`** -- Grid-search parameter optimizer. Finds the best trading configurations ("Golden Parameters") targeting >1.5 profit factor and >50% win rate.
- **`bot.py`** -- Live trading bot. Executes trades on Alpaca (paper trading) using XGBoost predictions ranked across a stock universe with 23+ technical/institutional features.

## Project Structure

```
Hedge-fund/
├── backtester.py              # Grid-search backtester entry point
├── bot.py                     # Live trading bot entry point
├── hedge_fund/                # Shared library (extracted from both scripts)
│   ├── __init__.py
│   ├── indicators.py          # RSI, ATR, Bollinger Bands, ADX
│   ├── math_utils.py          # Kalman filter, Hurst exponent
│   ├── simulation.py          # Bracket trade simulation, ML label generation
│   ├── features.py            # VPIN, VWAP, Amihud, RRS, liquidity sweeps
│   ├── risk.py                # Position sizing, Kelly criterion
│   ├── data.py                # Token-bucket rate limiter
│   └── objectives.py          # XGBoost custom loss functions
├── tests/                     # 67 unit tests
│   ├── test_indicators.py     # RSI, ATR, Bollinger, ADX
│   ├── test_simulation.py     # Bracket exits, trailing stops, labels
│   ├── test_risk.py           # Position sizing, Kelly criterion
│   └── test_broker.py         # Mock Alpaca order submission
└── .gitignore
```

## Setup

### Dependencies

```bash
pip install numpy pandas scipy xgboost scikit-learn rich requests
```

The bot additionally requires:

```bash
pip install alpaca-trade-api yfinance websocket-client pandas_ta
```

Optional (for model stacking and hyperparameter tuning):

```bash
pip install lightgbm optuna joblib
```

### Environment Variables

Set your API keys as environment variables (**do not hardcode them**):

```bash
export POLYGON_API_KEY="your-polygon-key"
export FMP_API_KEY="your-fmp-key"
export ALPACA_API_KEY="your-alpaca-key"
export ALPACA_SECRET_KEY="your-alpaca-secret"
export DISCORD_WEBHOOK_URL="your-discord-webhook"   # optional
```

## How to Run

### Run the backtester

```bash
python backtester.py
```

This will:
1. Download 1 year of hourly bars for 8 tech stocks from Polygon
2. Compute 27 features per ticker (indicators, Kalman, Hurst, VPIN, etc.)
3. Grid-search across 256+ parameter combinations
4. Print a ranked table of configurations sorted by resolved profit factor

### Run the live bot

```bash
python bot.py
```

This will:
1. Connect to Alpaca paper trading
2. Download historical data and train XGBoost models
3. Enter a 60-second scan loop: predict, rank, filter, and execute bracket trades
4. Manage open positions (trailing stops, profit scaling, time stops)
5. Log everything to SQLite + console dashboard

### Run the tests

```bash
python -m pytest tests/ -v
```

67 tests covering indicators, simulation, risk math, and broker order flow.

## How the System Works

### Signal Generation
1. **Feature engineering** -- 23+ features per bar: technicals (RSI, ADX, ATR), Kalman filter trend, Hurst exponent regime, VPIN toxic flow, VWAP mean-reversion, Amihud liquidity, liquidity sweeps.
2. **XGBoost prediction** -- Predicts R-value (risk-adjusted return multiple) for each symbol. Positive = LONG, negative = SHORT.
3. **Cross-sectional ranking** -- Ranks all symbols by predicted R-value, filters for positive expected value only.

### Entry Filters
10+ institutional-grade gates that must pass before a trade is taken:
- VPIN < 0.85 (block toxic order flow)
- Amihud < 0.90 (block illiquid conditions)
- Hurst < threshold (avoid choppy markets)
- ADX > threshold (require trend strength)
- EMA-200 alignment (trade with the trend)
- GEX regime match (mean reversion vs trending)
- VWAP Z-score bounds (block extremes)
- Volatility rank > 0.5 (avoid sleepy markets)

### Risk Management
- **Position sizing**: Risk 1.5% of equity per trade, capped at 20% of equity per position
- **Kelly criterion**: 3-outcome Kelly with 0.35x shrinkage for conservative sizing
- **ATR-based brackets**: Stop-loss at 1.5x ATR, take-profit at 3.0x ATR
- **Trailing stops**: Drag stop-loss by 1.0-1.5x ATR as price moves favorably
- **Profit scaling**: Take 1/3 at 1.5R, another 1/3 at 2.5R
- **Portfolio heat**: Max 6% total equity at risk across all positions
- **Kill switch**: Automatic shutdown at 10% drawdown

## Package API

The `hedge_fund` package can be imported directly for custom strategies:

```python
from hedge_fund.indicators import ManualTA
from hedge_fund.math_utils import get_kalman_filter, get_hurst
from hedge_fund.simulation import simulate_exit, compute_bracket_labels
from hedge_fund.features import calculate_vpin, calculate_amihud_illiquidity
from hedge_fund.risk import calculate_position_size, kelly_criterion

# Compute RSI
rsi = ManualTA.rsi(df['Close'], length=14)

# Kalman-filtered price
kalman = get_kalman_filter(df['Close'].values)

# Simulate a bracket trade
outcome, exit_price = simulate_exit(
    highs, lows, sl=95.0, tp=110.0, side='LONG', trail_dist=2.0
)

# Calculate position size
qty = calculate_position_size(
    equity=100_000, entry_price=150.0, stop_price=145.0
)
```

## Risk Warning

This system is for **paper trading and educational purposes only**. Past backtested performance does not guarantee future results. The 8-stock tech universe introduces significant concentration and survivorship bias. Do not trade real money without independent validation across broader markets and time periods.
