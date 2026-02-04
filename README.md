# 🤖 Hedge Fund Bot v14.3 "Hybrid God Mode"

**An institutional-grade algorithmic trading system powering a "Hybrid" strategy: combining high-conviction "Specialist" trades with high-frequency "Grinder" income.**

![Version](https://img.shields.io/badge/version-14.3-blue) ![Strategy](https://img.shields.io/badge/strategy-Hybrid_Mean_Reversion-green) ![Risk](https://img.shields.io/badge/risk-Monte_Carlo_Governor-red)

## 📖 What is this?
This is a sophisticated Python-based trading system designed to trade US Equities (Stocks) via the **Alpaca API**. It is not a simple "TA bot" but a complete **Hedge Fund in a Box** that simulates the decision-making process of a professional desk:
*   **"One Brain" Architecture**: The exact same logic powers both the Backtester and the Live Bot, ensuring what you test is what you trade.
*   **Institutional Risk Management**: It cares more about *staying alive* than making a quick buck, using Volatility targeting, Kelly Criterion sizing, and a Monte Carlo Governor.

## 🧠 How It Trades (The "Hybrid" Strategy)
The bot operates on a **Hybrid** philosophy, recognizing that not all market opportunities are the same. It splits its capital between two distinct "desks":

### 1. The Specialist (Tier 1) 🎯
*   **Goal**: Big Wins on Perfect Setups.
*   **Trigger**: High-confidence predictions (>30% edge), perfect Technicals (RSI/ADX aligned), and favorable Market Regime.
*   **Sizing**: **2.0x Risk**. We bet big when the odds are heavily in our favor.
*   **Behavior**: Patient. It might not trade for days, waiting for the "Fat Pitch."

### 2. The Grinder (Tier 2) ⚙️
*   **Goal**: Consistent Flow & Income.
*   **Trigger**: Moderate-confidence setups (>20% edge) that pass basic safety checks.
*   **Sizing**: **1.0x Risk**. Smaller, standard bets to keep the equity curve moving.
*   **Behavior**: Active. It keeps the system engaged and compounding small edges while waiting for Tier 1 opportunities.

### 🔬 The Technology Stack
*   **Signal Gen**: **XGBoost** (Gradient Boosting) models trained on 23+ features (Kalman Filters, Hurst Exponent, VPIN Toxic Flow, Amihud Illiquidity).
*   **Optimization**: **Portfolio Optimizer** using Mean-Variance analysis to allocate capital efficiently across uncorrelated assets.
*   **Governance**: **Monte Carlo Governor** that monitors the equity curve in real-time. If the bot enters a drawdown, it automatically cuts risk by 50% until performance recovers.

## 🚀 Why Is It Good?
Most retail bots fail because they overfit to the past and ignore risk. This system is different:
1.  **It Adapts**: Use of **Walk-Forward Analysis** means the bot retrains itself on new data periodically. It evolves with the market.
2.  **It Protects**: The **"Earnings Guard"** prevents holding through volatile events, and the **"Overnight Gap Model"** adjusts stops based on pre-market volatility.
3.  **It Scales**: The **Kelly Criterion** ensures we bet optimally—betting more when we are winning and less when we are losing.
4.  **It's Honest**: The **Slippage Calculator** in the backtester assumes you will lose money on execution, giving you realistic (not fantasy) performance expectations.

## 🔮 Possible Improvements (Roadmap)
While sophisticated, there is always room to grow:
*   **Alternative Data**: Integrate sentiment analysis (Twitter/Reddit/News) to catch "meme" moves early.
*   **Options Trading**: Specific modules to trade Options (Calls/Puts) instead of shares for better leverage management.
*   **Macro Awareness**: Integrate Federal Reserve data (Rates, Inflation) to switch strategies during bear markets.
*   **Low-Latency Execution**: Rewrite the execution engine in C++ or Rust for millisecond-level reaction times.

## 🛠️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# OR
pip install numpy pandas scipy xgboost scikit-learn rich requests alpaca-trade-api yfinance pandas_ta
```

### 2. Configure Keys
Create a `.env` file in the root directory:
```env
POLYGON_API_KEY=your_key_here
FMP_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_key_here
```

### 3. Run It
**To optimize parameters (The Lab):**
```bash
python backtester.py
```

**To trade live (The Arena):**
```bash
python bot.py
```
