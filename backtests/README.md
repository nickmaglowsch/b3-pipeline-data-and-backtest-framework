# B3 Quantitative Backtesting Framework

A modular, highly accurate Python backtesting framework specifically built for the Brazilian Stock Exchange (B3). 

Unlike generic backtesting engines, this framework interfaces directly with the native SQLite B3 database built by the `b3-data-pipeline`. It uses **ISIN-linked**, fully split-and-dividend-adjusted historical data (`adj_close`), making it completely free of survivorship bias and robust against the ticker changes/mergers that plague the Brazilian market.

Furthermore, it implements the **exact mechanics of Brazilian Capital Gains Tax** for variable income (Lei 11.033/2004), including infinite loss carryforwards, giving you realistic "After-Tax" equity curves.

## Framework Architecture

The framework is organized into reusable core modules located in `backtests/core/`:

- `data.py`: Handles data extraction from SQLite and Yahoo Finance. Calculates daily financial volume natively from COTAHIST logs.
- `portfolio.py`: Contains the actual simulation ledgers, equal-weight rebalancing logic, and the Brazilian tax computation engine.
- `metrics.py`: Standard quantitative metrics (Sharpe, Drawdown, Calmar, Volatility) and console table rendering.
- `plotting.py`: Generates the beautiful, institutional-grade 4-panel "tear sheet" summarizing the backtest performance.

## How to Build a New Backtest

Building a new backtest is incredibly simple. All you have to do is define your **alpha signal**.

Here is the standard skeleton for creating a new backtest script:

```python
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Import the Core Framework
from core.data import load_b3_data, download_benchmark
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.portfolio import rebalance_positions, apply_returns, compute_tax

# 2. Configuration
START_DATE = "2010-01-01"  
END_DATE = "2026-01-01"
INITIAL_CAPITAL = 100_000 
MIN_ADTV = 1_000_000 # R$ 1M minimum average daily traded volume

# 3. Define your Signal Logic
def custom_signal(monthly_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Your logic goes here. 
    Return a DataFrame where rows=dates, columns=tickers, 
    values=signal score (higher = better).
    """
    # Example: 1-month mean reversion (buy the biggest losers)
    return -monthly_ret.shift(1)

# 4. Implement the Loop
def run_backtest(adj_close, close_px, fin_vol):
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()
    monthly_raw_close = close_px.resample("ME").last()
    monthly_adtv = fin_vol.resample("ME").mean()
    
    signal = custom_signal(monthly_ret)
    
    # Setup ledger dictionaries
    pretax_positions, aftertax_positions = {}, {}
    loss_carryforward = 0.0

    # ... Loop through time (see momentum_tax_backtest.py for standard loop)
    # Apply Liquidity Rule -> Select Top N -> Apply Returns -> Rebalance -> Record
```

### Key Concepts for Agentic / Human Development

1. **Liquidity Filtering (`MIN_ADTV`)**: The Brazilian market has thousands of illiquid micro-caps and "ghost" tickers. Always filter your signal candidates using the `fin_vol` DataFrame (Financial Volume). A standard minimum is `1_000_000` BRL Average Daily Traded Volume in the preceding month.
2. **Penny Stock Filter**: Always ensure the `close_px` (raw close) was >= R$ 1.00 in the preceding month to avoid massive percentage swings caused by a R$ 0.01 tick movement on a R$ 0.10 stock.
3. **Glitch Protection**: Because B3 split adjustments can occasionally feature a 1-day misaligned jump, always cap your portfolio returns using `apply_returns(..., min_ret=-0.90)`.

## Running the Example Backtest

A complete, production-ready momentum strategy is provided.

```bash
cd backtests
python momentum_tax_backtest.py
```

It will execute the simulation, print the metric summary to the terminal, and save `momentum_dynamic_backtest.png` to the directory.
