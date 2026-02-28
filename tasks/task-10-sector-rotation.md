# Task 10: Sector Rotation Strategy

## Objective
Test whether rotating among B3 sectors based on relative momentum improves risk-adjusted returns compared to stock-level selection. Instead of picking the best individual stocks, this approach picks the best sector and buys the top stocks within that sector.

## Context
The B3 market is heavily concentrated in a few sectors: financials (Itau, Bradesco, B3SA3), commodities (Vale, Petrobras), utilities (Eletrobras, Sabesp), and consumer/retail (Magazine Luiza, Lojas Renner). These sectors have very different sensitivity to macro conditions:
- **Commodities**: driven by global commodity prices and USD/BRL
- **Financials**: benefit from high interest rates (spread income)
- **Utilities**: defensive, benefit from rate cuts (lower cost of capital)
- **Consumer**: sensitive to domestic GDP and consumer confidence

None of the existing 30+ backtests implement sector-aware strategies. All stock selection is cross-sectional within the full universe.

## Requirements
- Create a new file `backtests/sector_rotation_backtest.py`
- Implement sector classification for B3 tickers
- Test three sector rotation variants:

### Variant 1: Sector Momentum
Rank sectors by trailing 6-month return. Go long the top 2 sectors (equal weight within each), sit in CDI if no sector has positive momentum. Rebalance monthly.

### Variant 2: Sector Momentum + Stock Selection
Rank sectors by trailing 6-month return. Within the top 2 sectors, apply the MultiFactor stock selection (top decile by 50% mom + 50% low-vol). This combines sector timing with stock picking.

### Variant 3: Sector Rotation + COPOM Regime
Apply sector rotation only during COPOM easing. During tightening, sit in CDI. This combines sector rotation with the proven regime filter.

### For all variants:
- Universe: ADTV >= R$1M, price >= R$1.0, standard lot tickers
- Rebalance: Monthly
- Tax: 15% CGT, 0.1% slippage, R$20K monthly exemption
- Period: 2005-present
- Compare against: MultiFactor (sector-agnostic), COPOM Easing, IBOV, CDI

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/data.py` -- `load_b3_data()` returns wide-format DataFrames with ticker columns
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` -- Pattern for MultiFactor signal computation (lines 239-266)
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/copom_momentum_backtest.py` -- Pattern for COPOM regime + stock selection
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/simulation.py` -- `run_simulation()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `plot_tax_backtest()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/storage.py` -- DB schema reference. The `prices` table has `ticker` but no sector column.

## Implementation Details

### Sector Classification
B3 does not include sector data in COTAHIST files, and the database has no sector column. We need a heuristic mapping based on well-known ticker prefixes and a manual lookup table. Create a dict mapping ticker -> sector:

```python
# Heuristic B3 sector mapping
# Most B3 tickers can be classified by their first 4 characters
SECTOR_MAP = {
    # Financials
    "ITUB": "Financials", "BBDC": "Financials", "BBAS": "Financials",
    "SANB": "Financials", "BPAC": "Financials", "B3SA": "Financials",
    "ITSA": "Financials", "BRSR": "Financials", "ABCB": "Financials",
    "BMGB": "Financials", "BPAN": "Financials",

    # Commodities / Mining / Oil&Gas
    "VALE": "Commodities", "PETR": "Commodities", "CSNA": "Commodities",
    "GGBR": "Commodities", "USIM": "Commodities", "GOAU": "Commodities",
    "CMIN": "Commodities", "BRAP": "Commodities", "SUZB": "Commodities",
    "KLBN": "Commodities", "DXCO": "Commodities",

    # Utilities
    "ELET": "Utilities", "SBSP": "Utilities", "CMIG": "Utilities",
    "CPFE": "Utilities", "EGIE": "Utilities", "ENGI": "Utilities",
    "TAEE": "Utilities", "TRPL": "Utilities", "CPLE": "Utilities",
    "NEOE": "Utilities", "AURE": "Utilities", "SAPR": "Utilities",
    "ENEV": "Utilities",

    # Consumer / Retail
    "MGLU": "Consumer", "LREN": "Consumer", "AMER": "Consumer",
    "VIIA": "Consumer", "PETZ": "Consumer", "SOMA": "Consumer",
    "ARZZ": "Consumer", "GRND": "Consumer", "ALPA": "Consumer",
    "NTCO": "Consumer", "ABEV": "Consumer", "JBSS": "Consumer",
    "BRFS": "Consumer", "MRFG": "Consumer", "BEEF": "Consumer",
    "MDIA": "Consumer", "PCAR": "Consumer", "CRFB": "Consumer",
    "ASAI": "Consumer", "RAIZ": "Consumer",

    # Real Estate
    "CYRE": "RealEstate", "MRVE": "RealEstate", "EZTC": "RealEstate",
    "EVEN": "RealEstate", "DIRR": "RealEstate", "TEND": "RealEstate",
    "MULT": "RealEstate", "IGTI": "RealEstate", "BRML": "RealEstate",
    "ALSO": "RealEstate",

    # Healthcare
    "HAPV": "Healthcare", "RDOR": "Healthcare", "FLRY": "Healthcare",
    "QUAL": "Healthcare", "HYPE": "Healthcare",

    # Telecom / Tech
    "VIVT": "TechTelecom", "TIMS": "TechTelecom", "TOTS": "TechTelecom",
    "LWSA": "TechTelecom", "POSI": "TechTelecom", "INTB": "TechTelecom",

    # Transportation / Infrastructure
    "CCRO": "Infrastructure", "ECOR": "Infrastructure", "RAIL": "Infrastructure",
    "AZUL": "Infrastructure", "GOLL": "Infrastructure", "EMBR": "Infrastructure",
    "RENT": "Infrastructure", "MOVI": "Infrastructure",

    # Insurance
    "BBSE": "Insurance", "PSSA": "Insurance", "SULA": "Insurance",
    "IRBR": "Insurance",
}

def classify_ticker(ticker: str) -> str:
    """Classify a B3 ticker into a sector using the first 4 chars."""
    prefix = ticker[:4]
    return SECTOR_MAP.get(prefix, "Other")
```

Tickers not in the map get classified as "Other". The "Other" sector should be excluded from sector rotation (it is too heterogeneous) but stocks in it can still be selected if they fall within a known sector's expanded universe.

### Sector Return Computation
```python
# Assign each ticker to a sector
sector_assignments = {t: classify_ticker(t) for t in ret.columns}

# Compute equal-weighted sector return each month
sector_returns = pd.DataFrame()
for sector in set(sector_assignments.values()):
    if sector == "Other":
        continue
    sector_tickers = [t for t, s in sector_assignments.items() if s == sector]
    # Only include liquid tickers
    valid_tickers = [t for t in sector_tickers if t in ret.columns]
    if valid_tickers:
        sector_returns[sector] = ret[valid_tickers].mean(axis=1)
```

### Sector Momentum Signal
```python
sector_mom = sector_returns.rolling(6).apply(lambda x: (1 + x).prod() - 1)
sector_rank = sector_mom.rank(axis=1, ascending=False)
top_sectors = sector_rank.apply(lambda row: row[row <= 2].index.tolist(), axis=1)
```

### Output Files
- `sector_rotation_backtest.png` -- Standard 4-panel tear sheet for the best variant
- `sector_rotation_analysis.png` -- Sector momentum heatmap (sectors x years, colored by return)
- Console: comparison table, sector coverage statistics

## Acceptance Criteria
- [ ] File `backtests/sector_rotation_backtest.py` exists and runs successfully
- [ ] Sector classification covers at least 70% of liquid-universe tickers (ADTV >= R$1M)
- [ ] Sector returns are computed for at least 5 distinct sectors
- [ ] All 3 variants produce valid equity curves
- [ ] Sector momentum heatmap provides visual insight into sector cycles
- [ ] Results compared against MultiFactor, COPOM Easing, IBOV, CDI
- [ ] If any variant has Sharpe > 0.3, report its correlation with existing strategies
- [ ] Both plots saved

## Dependencies
- Depends on: Task 1 (data loading patterns, signal computation patterns)
- Blocks: None
