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
STATUSINVEST_PROVENTS_URL = "https://statusinvest.com.br/acao/companytickerprovents?ticker={ticker}&chartProventsType=2"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"

DATA_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1994
CURRENT_YEAR = datetime.now().year

EQUITY_BDI_CODES = {"02"}

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

STATUSINVEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://statusinvest.com.br/acoes/",
}

B3_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

SPLIT_DETECTION_THRESHOLD_HIGH = 1.8
SPLIT_DETECTION_THRESHOLD_LOW = 0.55

COMMON_SPLIT_RATIOS = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
    (1, 5),
    (5, 1),
    (1, 10),
    (10, 1),
]

RATE_LIMIT_DELAY = 0.1
