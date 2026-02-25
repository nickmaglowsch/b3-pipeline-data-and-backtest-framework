# B3 Quantitative Backtesting Framework

A modular, highly accurate Python backtesting framework specifically built for the Brazilian Stock Exchange (B3). 

Unlike generic backtesting engines, this framework interfaces directly with the native SQLite B3 database built by the `b3-data-pipeline`. It uses **ISIN-linked**, fully split-and-dividend-adjusted historical data (`adj_close`), making it completely free of survivorship bias and robust against the ticker changes/mergers that plague the Brazilian market.

Furthermore, it implements the **exact mechanics of Brazilian Capital Gains Tax** for variable income (Lei 11.033/2004), including infinite loss carryforwards, giving you realistic "After-Tax" equity curves.

## Framework Architecture

The framework is organized into reusable core modules located in `backtests/core/`:

- `data.py`: Handles data extraction from SQLite and Yahoo Finance. Calculates daily financial volume natively from COTAHIST logs, and seamlessly fetches daily CDI data directly from the Brazilian Central Bank (BCB API).
- `simulation.py`: The universal multi-asset portfolio simulator. It automatically handles exact target-weight rebalancing, slippage friction, margin borrowing calculations, and the unified Brazilian Capital Gains tax engine.
- `metrics.py`: Standard quantitative metrics (Sharpe, Drawdown, Calmar, Volatility) properly annualized across any trading frequency (Monthly, Quarterly, Weekly).
- `plotting.py`: Generates the beautiful, institutional-grade 4-panel "tear sheet" summarizing the backtest performance alongside multiple configurable benchmarks (like IBOV and CDI).

## How to Build a New Backtest

Building a new backtest has been streamlined down to building a simple matrix of **Target Weights**. The generic simulation engine handles all the complex ledgers and tax calculations.

Here is the standard skeleton for creating a new backtest script:

```python
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Import the Core Framework
from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.simulation import run_simulation

# 2. Configuration
REBALANCE_FREQ = "ME"  # ME=Monthly, QE=Quarterly, W-FRI=Weekly
START_DATE = "2015-01-01"  
END_DATE = "2026-01-01"
INITIAL_CAPITAL = 100_000 
MIN_ADTV = 1_000_000 # R$ 1M minimum average daily traded volume

# 3. Define your Signal & Target Weights
def generate_signals(adj_close, close_px, fin_vol):
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()
    
    # Create an empty target weight matrix of the same shape as returns
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    
    # ... Your logic goes here to populate target_weights ...
    # e.g., Set top 10 momentum stocks to 0.10 each per row
    
    return ret, target_weights

# 4. Implement the Loop
def main():
    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol)
    
    # Run the Generic Simulator Engine!
    result = run_simulation(
        returns_matrix=ret.fillna(0.0),
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=0.15,
        slippage=0.001  # 0.1% cost per trade
    )
    
    # ... Plot using plot_tax_backtest(result...)
```

### Advanced Features

#### Modeling Margin Leverage
To model leverage, simply allow your `target_weights` row to sum to more than `1.0`, and add a negative cash asset (representing the loan). The simulator natively handles the negative balance, correctly charging the CDI risk-free rate + your defined `BORROW_SPREAD` on the margin, without mistakenly granting you capital gains tax write-offs for the interest paid.

```python
target_weights.iloc[i]["PETR4"] = 1.0     # 100% long PETR4
target_weights.iloc[i]["CDI_ASSET"] = -0.50 # Borrow 50% on margin
```

#### Rebalance Frequencies
Because the framework strictly uses Pandas offset aliases (`ME`, `QE`, `W-FRI`), the backtest effortlessly scales from Monthly down to Weekly. If you change frequency, remember to recalculate your `LOOKBACK_PERIODS` (e.g. 1 year lookback = 12 `ME` periods, but 52 `W-FRI` periods).

### Key Concepts for Agentic / Human Development

1. **Liquidity Filtering (`MIN_ADTV`)**: The Brazilian market has thousands of illiquid micro-caps and "ghost" tickers. Always filter your signal candidates using the `fin_vol` DataFrame (Financial Volume). A standard minimum is `1_000_000` BRL Average Daily Traded Volume in the preceding month.
2. **Penny Stock Filter**: Always ensure the `close_px` (raw close) was >= R$ 1.00 in the preceding month to avoid massive percentage swings caused by a R$ 0.01 tick movement on a R$ 0.10 stock.

## Running the Example Backtests

The `backtests/` folder comes pre-loaded with several complete, highly-accurate strategies for the Brazilian market:

- `momentum_tax_backtest.py`: The classic cross-sectional momentum anomaly.
- `low_volatility_backtest.py`: The low-risk anomaly.
- `mean_reversion_backtest.py`: Reversal trading the biggest 1-month losers.
- `multifactor_cdi_ivvb11_backtest.py`: An institutional "All-Weather" multi-asset blend of B3 Factors, S&P 500 exposure, and CDI yield, using the new universal simulation engine.
