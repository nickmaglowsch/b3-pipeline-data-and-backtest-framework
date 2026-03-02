"""
Tests for Phase 1: COTAHIST fator_cotacao parsing and price normalization.

These tests verify:
1. quotation_factor is parsed from positions 210-217 of each COTAHIST line
2. Prices (OHLC) are divided by quotation_factor to normalize to per-share basis
3. Edge cases: fatcot=0 defaults to 1 (no division by zero), fatcot=1 is unchanged
4. volume is NOT divided (it's already in BRL, not share count)
"""
import io
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# We test at the function level to avoid needing real ZIP files.
# Import the helpers directly.
from b3_pipeline.parser import _parse_int, _parse_price, parse_cotahist_file


def _build_cotahist_line(
    date_str: str = "20050103",
    cod_bdi: str = "02",
    ticker: str = "PETR4       ",  # 12 chars
    tipo_mercado: str = "010",
    open_price_raw: int = 2800000,   # R$28,000.00 in raw (implied 2 decimals) for 1000-lot => R$28.00/share
    high_price_raw: int = 2900000,
    low_price_raw: int = 2700000,
    close_price_raw: int = 2850000,
    volume_raw: int = 1234567800,    # large int for volume
    fatcot: int = 1000,
    isin_code: str = "BRPETRACNPR6",
) -> str:
    """
    Build a minimal COTAHIST fixed-width line of exactly 245+ characters.

    COTAHIST record layout (positions are 0-based):
      [0:2]   tipo_registro = "01"
      [2:10]  data_pregao   = YYYYMMDD
      [10:12] cod_bdi
      [12:24] cod_negociacao (ticker, 12 chars)
      [24:27] tipo_mercado
      [27:56] filler (nome_empresa 12 chars + especificacao_papel 10 chars + etc.)
      [56:69] preco_abertura  (13 digits, 2 implied decimals)
      [69:82] preco_maximo
      [82:95] preco_minimo
      [95:108] preco_medio (not used)
      [108:121] preco_ultimo_negocio (close)
      [121:170] filler (other prices, etc.)
      [170:188] volume_total_titulos (18 digits, 2 implied decimals)
      [188:210] filler
      [210:217] fator_cotacao (7 digits)
      [217:230] preco_exercicio_pontos (filler)
      [230:242] cod_isin
      [242:245] num_distribuicao
    """
    line = list(" " * 250)

    # tipo_registro
    for i, c in enumerate("01"):
        line[i] = c

    # date
    for i, c in enumerate(date_str):
        line[2 + i] = c

    # cod_bdi (2 chars)
    for i, c in enumerate(cod_bdi.ljust(2)):
        line[10 + i] = c

    # ticker (12 chars)
    ticker_padded = ticker.ljust(12)[:12]
    for i, c in enumerate(ticker_padded):
        line[12 + i] = c

    # tipo_mercado (3 chars)
    for i, c in enumerate(tipo_mercado):
        line[24 + i] = c

    # open price at [56:69] - 13 digit zero-padded int
    open_str = str(open_price_raw).zfill(13)
    for i, c in enumerate(open_str):
        line[56 + i] = c

    # high at [69:82]
    high_str = str(high_price_raw).zfill(13)
    for i, c in enumerate(high_str):
        line[69 + i] = c

    # low at [82:95]
    low_str = str(low_price_raw).zfill(13)
    for i, c in enumerate(low_str):
        line[82 + i] = c

    # close at [108:121]
    close_str = str(close_price_raw).zfill(13)
    for i, c in enumerate(close_str):
        line[108 + i] = c

    # volume at [170:188] - 18 digits
    vol_str = str(volume_raw).zfill(18)
    for i, c in enumerate(vol_str):
        line[170 + i] = c

    # fatcot at [210:217] - 7 digits
    fatcot_str = str(fatcot).zfill(7)
    for i, c in enumerate(fatcot_str):
        line[210 + i] = c

    # ISIN at [230:242] - 12 chars
    isin_padded = isin_code.ljust(12)[:12]
    for i, c in enumerate(isin_padded):
        line[230 + i] = c

    return "".join(line) + "\n"


def _make_zip_with_lines(lines: list, tmp_path: Path) -> Path:
    """Create a COTAHIST ZIP file containing the given lines."""
    zip_path = tmp_path / "COTAHIST_A2005.ZIP"
    txt_content = "".join(lines).encode("latin-1")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("COTAHIST_A2005.TXT", txt_content)
    return zip_path


class TestFatcotParsing:
    """Test that fator_cotacao is parsed and prices are normalized."""

    def test_fatcot_1000_divides_prices(self, tmp_path):
        """
        GIVEN a COTAHIST record with fatcot=1000 and close=2850000 (raw)
        WHEN the parser reads it
        THEN close should be 28.50 (2850000 / 100 / 1000 = 28.50) per share
        """
        line = _build_cotahist_line(close_price_raw=2850000, fatcot=1000)
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty, "Should have parsed at least one record"
        row = df.iloc[0]
        # raw parse: 2850000 / 100 = 28500.0 (implied 2 decimals)
        # fatcot normalization: 28500.0 / 1000 = 28.50
        assert abs(row["close"] - 28.50) < 0.001, (
            f"close should be 28.50 after fatcot=1000 normalization, got {row['close']}"
        )

    def test_fatcot_1_unchanged(self, tmp_path):
        """
        GIVEN a COTAHIST record with fatcot=1 and close=2850000 (raw)
        WHEN the parser reads it
        THEN close should be 28500.0 (unchanged, modern per-share price)
        """
        line = _build_cotahist_line(close_price_raw=2850000, fatcot=1)
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty
        row = df.iloc[0]
        # raw parse: 2850000 / 100 = 28500.0; fatcot=1 means no further division
        assert abs(row["close"] - 28500.0) < 0.001, (
            f"close should be 28500.0 with fatcot=1, got {row['close']}"
        )

    def test_fatcot_0_defaults_to_1(self, tmp_path):
        """
        GIVEN a COTAHIST record with fatcot=0 (invalid)
        WHEN the parser reads it
        THEN it should default to fatcot=1 (no division by zero)
        """
        line = _build_cotahist_line(close_price_raw=2850000, fatcot=0)
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty
        row = df.iloc[0]
        # fatcot=0 => default to 1, no division
        assert abs(row["close"] - 28500.0) < 0.001, (
            f"close should be 28500.0 with fatcot=0 (defaulted to 1), got {row['close']}"
        )

    def test_fatcot_10000_divides_prices(self, tmp_path):
        """
        GIVEN a COTAHIST record with fatcot=10000 and close=1000000 (raw = R$100.00/lot-of-10000)
        WHEN the parser reads it
        THEN close should be 0.01 per share (100.0 / 10000)
        """
        line = _build_cotahist_line(close_price_raw=1000000, fatcot=10000)
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty
        row = df.iloc[0]
        # raw parse: 1000000 / 100 = 10000.0; / 10000 = 1.0
        assert abs(row["close"] - 1.0) < 0.0001, (
            f"close should be 1.0 with fatcot=10000, got {row['close']}"
        )

    def test_fatcot_normalizes_all_ohlc(self, tmp_path):
        """
        GIVEN a COTAHIST record with fatcot=1000
        WHEN the parser reads it
        THEN open, high, low, close should all be divided by 1000
        """
        line = _build_cotahist_line(
            open_price_raw=2800000,
            high_price_raw=2900000,
            low_price_raw=2700000,
            close_price_raw=2850000,
            fatcot=1000,
        )
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty
        row = df.iloc[0]
        assert abs(row["open"] - 28.0) < 0.001, f"open={row['open']}"
        assert abs(row["high"] - 29.0) < 0.001, f"high={row['high']}"
        assert abs(row["low"] - 27.0) < 0.001, f"low={row['low']}"
        assert abs(row["close"] - 28.50) < 0.001, f"close={row['close']}"

    def test_volume_not_normalized_by_fatcot(self, tmp_path):
        """
        Volume (in BRL) should NOT be divided by fatcot -- it's already a monetary amount.
        """
        raw_volume = 1234567890
        line = _build_cotahist_line(volume_raw=raw_volume, fatcot=1000)
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty
        row = df.iloc[0]
        # Volume parsed with _parse_price (/ 100): 1234567890 / 100 = 12345678.9
        expected_volume = raw_volume / 100.0
        assert abs(row["volume"] - expected_volume) < 1.0, (
            f"volume should not be affected by fatcot, expected {expected_volume}, got {row['volume']}"
        )

    def test_quotation_factor_stored_in_dataframe(self, tmp_path):
        """
        The parsed DataFrame should include a quotation_factor column for auditability.
        """
        line = _build_cotahist_line(fatcot=1000)
        zip_path = _make_zip_with_lines([line], tmp_path)

        df = parse_cotahist_file(zip_path)

        assert not df.empty
        assert "quotation_factor" in df.columns, (
            "DataFrame should have 'quotation_factor' column"
        )
        assert df.iloc[0]["quotation_factor"] == 1000
