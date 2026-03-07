"""
Configuration constants for B3 data pipeline.
"""

from pathlib import Path
from datetime import datetime

BASE_COTAHIST_URL = (
    "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
)
DAILY_COTAHIST_URL = (
    "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{date:%d%m%Y}.ZIP"
)
B3_CASH_DIVIDENDS_URL = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/{payload}"
B3_STOCK_CORP_ACTIONS_URL = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/{payload}"
B3_COMPANY_LIST_URL = (
    "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/"
    "CompanyCall/GetInitialCompanies/{payload}"
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"

DATA_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1994


def get_current_year() -> int:
    """Return the current year at call time (not import time)."""
    return datetime.now().year


CURRENT_YEAR = get_current_year()  # Keep backward compat; prefer get_current_year() in new code

EQUITY_BDI_CODES = {"02"}

# ── CVM open data portal ──────────────────────────────────────────────────────
CVM_DFP_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/"
CVM_ITR_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/"
CVM_FRE_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/"
CVM_IPE_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
CVM_CAD_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/"
CVM_CAD_FILENAME = "cad_cia_aberta.csv"
CVM_DATA_DIR = PROJECT_ROOT / "data" / "cvm"
CVM_DATA_DIR.mkdir(parents=True, exist_ok=True)
CVM_START_YEAR = 2010

# NOTE: This layout dictionary is for reference only and may not exactly match
# the field positions used in parser.py (which hardcodes the correct offsets).
# Do not use this dict for programmatic parsing.
COTAHIST_LAYOUT = {
    "tipo_registro": (0, 2),
    "data_pregao": (2, 10),
    "cod_bdi": (10, 12),
    "cod_negociacao": (12, 24),
    "tipo_mercado": (24, 27),
    "nome_empresa": (27, 39),
    "especificacao_papel": (39, 49),
    "prazo_dias_termo": (49, 52),
    "moeda_referencia": (52, 56),
    "preco_abertura": (56, 69),
    "preco_negocio": (69, 82),
    "preco_maximo": (82, 95),
    "preco_minimo": (95, 108),
    "preco_medio": (108, 121),
    "preco_ultimo_negocio": (121, 134),
    "preco_melhor_oferta_compra": (134, 147),
    "preco_melhor_oferta_venda": (147, 160),
    "total_negocios": (147, 152),
    "quantidade_total_titulos": (152, 170),
    "volume_total_titulos": (170, 188),
    "preco_exercicio": (188, 201),
    "indicador_correcao": (201, 202),
    "data_vencimento": (202, 210),
    "fator_cotacao": (210, 217),
    "preco_exercicio_pontos": (217, 230),
    "cod_isin": (230, 242),
    "num_distribuicao": (242, 245),
}

B3_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

B3_CORP_ACTIONS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

SPLIT_DETECTION_THRESHOLD_HIGH = 1.8
SPLIT_DETECTION_THRESHOLD_LOW = 0.55

# Delay (seconds) each worker thread sleeps between its own sequential HTTP requests.
# With MAX_WORKERS concurrent workers the aggregate request rate ceiling is
# approximately MAX_WORKERS / RATE_LIMIT_DELAY requests/second.
RATE_LIMIT_DELAY = 0.05

# Maximum number of concurrent worker threads for the corporate actions fetch step.
# Increase cautiously -- the B3 API has no published rate limit.
MAX_WORKERS = 10

EVENT_TYPE_CASH_DIVIDEND = "CASH_DIVIDEND"
EVENT_TYPE_JCP = "JCP"
EVENT_TYPE_STOCK_SPLIT = "STOCK_SPLIT"
EVENT_TYPE_REVERSE_SPLIT = "REVERSE_SPLIT"
EVENT_TYPE_BONUS_SHARES = "BONUS_SHARES"

B3_LABEL_DIVIDEND = "DIVIDENDO"
B3_LABEL_JCP = "JRS CAP PROPRIO"
B3_LABEL_RENDIMENTO = "RENDIMENTO"
B3_LABEL_DESDOBRAMENTO = "DESDOBRAMENTO"
B3_LABEL_GRUPAMENTO = "GRUPAMENTO"
B3_LABEL_BONIFICACAO = "BONIFICACAO"

# Non-split labels that affect share count but are NOT standard splits.
# These are stored in skipped_events for manual review.
B3_LABEL_RESG_TOTAL_RV = "RESG TOTAL RV"    # Total share redemption (delisting event)
B3_LABEL_CIS_RED_CAP = "CIS RED CAP"        # Spin-off with capital reduction (complex)
B3_LABEL_INCORPORACAO = "INCORPORACAO"       # Merger/incorporation (complex)
