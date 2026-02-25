"""
B3 Corporate Actions API client.

Fetches corporate actions (dividends, JCP, splits, bonuses) directly from B3's
listedCompaniesProxy backend, making B3 the single authoritative source.
"""

import base64
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)


def _encode_payload(payload: dict) -> str:
    """Encode a dictionary as base64 string for B3 API."""
    json_str = json.dumps(payload, separators=(",", ":"))
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def _parse_b3_date(date_str: str) -> Optional[datetime]:
    """Parse B3 date string in DD/MM/YYYY format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y")
    except ValueError:
        return None


def _parse_b3_float(value_str: str) -> float:
    """Parse B3 localized float string (comma as decimal separator)."""
    if not value_str:
        return 0.0
    try:
        normalized = value_str.strip().replace(".", "").replace(",", ".")
        return float(normalized)
    except ValueError:
        return 0.0


def _isin_to_ticker_suffix(isin_code: str) -> Tuple[str, str]:
    """
    Extract company code and share type from ISIN.

    ISIN format: BR + 4-char company code + share type + check digit
    Example: BRPETRACNOR9 -> ('PETR', 'ON')
             BRPETRACNPR6 -> ('PETR', 'PN')

    Returns:
        Tuple of (company_code, share_type)
    """
    if len(isin_code) < 8:
        return "", ""

    company_code = isin_code[2:6]

    if "OR" in isin_code[6:11]:
        share_type = "ON"
    elif "PR" in isin_code[6:11]:
        share_type = "PN"
    else:
        share_type = "ON"

    return company_code, share_type


def _map_isin_to_ticker(isin_code: str, available_tickers: List[str]) -> Optional[str]:
    """
    Map an ISIN code to a ticker symbol.

    Args:
        isin_code: ISIN code (e.g., 'BRPETRACNOR9')
        available_tickers: List of tickers available in our database

    Returns:
        Matching ticker or None
    """
    company_code, share_type = _isin_to_ticker_suffix(isin_code)

    if not company_code:
        return None

    matching = [t for t in available_tickers if t.startswith(company_code)]

    if not matching:
        return None

    if share_type == "ON":
        for t in matching:
            if t.endswith("3"):
                return t
    elif share_type == "PN":
        for t in matching:
            if t.endswith("4") or t.endswith("5") or t.endswith("6"):
                return t

    if matching:
        return matching[0]

    return None


def fetch_company_data(trading_name: str) -> Optional[dict]:
    """
    Fetch all corporate action data for a company from B3.

    This calls GetListedSupplementCompany which returns:
    - cashDividends: dividends, JCP, rendimento
    - stockDividends: desdobramento, grupamento, bonificação
    - subscriptions: subscription rights

    Args:
        trading_name: Company trading name (e.g., 'PETROBRAS')

    Returns:
        Raw JSON response or None on failure
    """
    payload = {
        "issuingCompany": trading_name.split()[0]
        if " " in trading_name
        else trading_name,
        "language": "pt-br",
    }
    encoded = _encode_payload(payload)
    url = config.B3_STOCK_CORP_ACTIONS_URL.format(payload=encoded)

    try:
        resp = requests.get(url, headers=config.B3_CORP_ACTIONS_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # B3 API sometimes returns a JSON string that needs to be parsed again
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass

        # If it's a list, we probably want the first element
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        elif isinstance(data, dict):
            return data

        return None
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch company data for {trading_name}: {e}")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse response for {trading_name}: {e}")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse response for {trading_name}: {e}")
        return None


def fetch_cash_dividends_paginated(
    trading_name: str, page_size: int = 100
) -> List[dict]:
    """
    Fetch all cash dividends with pagination.

    The GetListedCashDividends endpoint returns paginated results with
    totalRecords and totalPages fields.

    Args:
        trading_name: Company trading name (e.g., 'PETROBRAS')
        page_size: Number of records per page

    Returns:
        List of all dividend records
    """
    all_results = []
    page_number = 1
    total_pages = 1

    while page_number <= total_pages:
        payload = {
            "language": "pt-br",
            "pageNumber": page_number,
            "pageSize": page_size,
            "tradingName": trading_name,
        }
        encoded = _encode_payload(payload)
        url = config.B3_CASH_DIVIDENDS_URL.format(payload=encoded)

        try:
            resp = requests.get(url, headers=config.B3_CORP_ACTIONS_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            page_info = data.get("page", {})
            total_pages = page_info.get("totalPages", 1)
            results = data.get("results", [])
            all_results.extend(results)

            logger.debug(
                f"Fetched page {page_number}/{total_pages} for {trading_name}, "
                f"got {len(results)} records"
            )

            page_number += 1

            if page_number <= total_pages:
                time.sleep(config.RATE_LIMIT_DELAY)

        except requests.RequestException as e:
            logger.warning(
                f"Failed to fetch dividends page {page_number} for {trading_name}: {e}"
            )
            break
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse dividends for {trading_name}: {e}")
            break

    return all_results


def parse_cash_dividends(
    records: List[dict], ticker_root: str, ticker_to_isin: dict
) -> pd.DataFrame:
    """
    Parse cash dividend records from B3 response.

    Maps B3 labels to event types:
    - DIVIDENDO -> CASH_DIVIDEND
    - JRS CAP PROPRIO -> JCP
    - RENDIMENTO -> CASH_DIVIDEND

    Args:
        records: List of cash dividend records from B3

    Returns:
        DataFrame with columns: [isin_code, event_date, event_type, value, source]
    """
    events = []

    for record in records:
        label = record.get("label", record.get("corporateAction", ""))
        value_str = record.get("rate", record.get("valueCash", "0"))
        isin_code = record.get("isinCode", record.get("assetIssued", ""))
        date_str = record.get("lastDatePrior", record.get("lastDatePriorEx", ""))

        if not isin_code:
            type_stock = record.get("typeStock", "")
            guessed_ticker = None
            if type_stock == "ON":
                guessed_ticker = f"{ticker_root}3"
            elif type_stock == "PN":
                guessed_ticker = f"{ticker_root}4"
            elif type_stock == "PNA":
                guessed_ticker = f"{ticker_root}5"
            elif type_stock == "PNB":
                guessed_ticker = f"{ticker_root}6"
            elif type_stock == "UNT":
                guessed_ticker = f"{ticker_root}11"

            if guessed_ticker and guessed_ticker in ticker_to_isin:
                isin_code = ticker_to_isin[guessed_ticker]
            else:
                continue

        if label == config.B3_LABEL_DIVIDEND:
            event_type = config.EVENT_TYPE_CASH_DIVIDEND
        elif label == config.B3_LABEL_JCP:
            event_type = config.EVENT_TYPE_JCP
        elif label == config.B3_LABEL_RENDIMENTO:
            event_type = config.EVENT_TYPE_CASH_DIVIDEND
        else:
            continue

        event_date = _parse_b3_date(date_str)
        if event_date is None:
            continue

        value = _parse_b3_float(value_str)
        if value <= 0:
            continue

        events.append(
            {
                "isin_code": isin_code,
                "event_date": event_date.date(),
                "event_type": event_type,
                "value": value,
                "source": "B3",
            }
        )

    if events:
        return pd.DataFrame(events)
    return pd.DataFrame(
        columns=["isin_code", "event_date", "event_type", "value", "source"]
    )


def parse_stock_dividends(records: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse stock dividend records (splits, reverse splits, bonuses) from B3.

    Maps B3 labels to action types:
    - DESDOBRAMENTO -> STOCK_SPLIT (factor > 1)
    - GRUPAMENTO -> REVERSE_SPLIT (factor < 1)
    - BONIFICACAO -> BONUS_SHARES

    Args:
        records: List of stock dividend records from B3

    Returns:
        Tuple of (corporate_actions_df, stock_actions_df)
    """
    corp_events = []
    stock_events = []

    for record in records:
        label = record.get("label", "")
        factor_str = record.get("factor", "1")
        isin_code = record.get("isinCode", record.get("assetIssued", ""))
        date_str = record.get("lastDatePrior", "")

        if not isin_code:
            continue

        factor = _parse_b3_float(factor_str)
        if factor <= 0:
            continue

        ex_date = _parse_b3_date(date_str)
        if ex_date is None:
            continue

        if label == config.B3_LABEL_DESDOBRAMENTO:
            action_type = config.EVENT_TYPE_STOCK_SPLIT
        elif label == config.B3_LABEL_GRUPAMENTO:
            action_type = config.EVENT_TYPE_REVERSE_SPLIT
        elif label == config.B3_LABEL_BONIFICACAO:
            action_type = config.EVENT_TYPE_BONUS_SHARES
        else:
            continue

        stock_events.append(
            {
                "isin_code": isin_code,
                "ex_date": ex_date.date(),
                "action_type": action_type,
                "factor": factor,
                "source": "B3",
            }
        )

        corp_events.append(
            {
                "isin_code": isin_code,
                "event_date": ex_date.date(),
                "event_type": action_type,
                "value": None,
                "factor": factor,
                "source": "B3",
            }
        )

    corp_df = (
        pd.DataFrame(corp_events)
        if corp_events
        else pd.DataFrame(
            columns=[
                "isin_code",
                "event_date",
                "event_type",
                "value",
                "factor",
                "source",
            ]
        )
    )
    stock_df = (
        pd.DataFrame(stock_events)
        if stock_events
        else pd.DataFrame(
            columns=["isin_code", "ex_date", "action_type", "factor", "source"]
        )
    )

    return corp_df, stock_df


def fetch_all_corporate_actions(
    trading_names: List[str], ticker_to_isin: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all corporate actions for multiple trading names.

    Args:
        trading_names: List of 4-character ticker roots (e.g., 'PETR')
        ticker_to_isin: Dictionary mapping tickers to ISINs

    Returns:
        Tuple of (corporate_actions_df, stock_actions_df)
    """
    all_corp_actions = []
    all_stock_actions = []
    total = len(trading_names)

    for i, name in enumerate(trading_names, 1):
        if i % 10 == 0 or i == total:
            logger.info(f"Fetching corporate actions: {i}/{total} ({name})")

        # 1. Fetch company data using 4-letter root as issuingCompany
        company_data = fetch_company_data(name)
        if company_data is None:
            continue

        # Extract full trading name needed for the cash dividends endpoint
        full_trading_name = company_data.get("tradingName", "").strip()

        # 2. Fetch stock actions (splits, bonuses) from supplement endpoint
        stock_divs = company_data.get("stockDividends", [])

        if stock_divs:
            corp_from_stock, stock_df = parse_stock_dividends(stock_divs)
            if not corp_from_stock.empty:
                all_corp_actions.append(corp_from_stock)
            if not stock_df.empty:
                all_stock_actions.append(stock_df)

        time.sleep(config.RATE_LIMIT_DELAY)

        # 3. Fetch cash dividends from paginated endpoint using full trading name
        if full_trading_name:
            cash_divs = fetch_cash_dividends_paginated(full_trading_name)
            if cash_divs:
                corp_df = parse_cash_dividends(cash_divs, name, ticker_to_isin)
                if not corp_df.empty:
                    all_corp_actions.append(corp_df)

        time.sleep(config.RATE_LIMIT_DELAY)

    valid_corp = [
        df for df in all_corp_actions if not df.empty and not df.isna().all().all()
    ]
    valid_stock = [
        df for df in all_stock_actions if not df.empty and not df.isna().all().all()
    ]

    final_corp = (
        pd.concat(valid_corp, ignore_index=True)
        if valid_corp
        else pd.DataFrame(
            columns=[
                "isin_code",
                "event_date",
                "event_type",
                "value",
                "factor",
                "source",
            ]
        )
    )
    final_stock = (
        pd.concat(valid_stock, ignore_index=True)
        if valid_stock
        else pd.DataFrame(
            columns=[
                "isin_code",
                "ex_date",
                "action_type",
                "factor",
                "source",
            ]
        )
    )

    if not final_corp.empty:
        final_corp = final_corp.drop_duplicates(
            subset=["isin_code", "event_date", "event_type"], keep="last"
        )
        logger.info(f"Total corporate actions: {len(final_corp)}")

    if not final_stock.empty:
        final_stock = final_stock.drop_duplicates(
            subset=["isin_code", "ex_date", "action_type"], keep="last"
        )
        logger.info(f"Total stock actions: {len(final_stock)}")

    return final_corp, final_stock


def build_trading_name_to_code_map(tickers: List[str]) -> Dict[str, str]:
    """
    Build a mapping from ticker roots to trading names.

    B3 uses trading names like 'PETROBRAS' but we have tickers like 'PETR4'.
    This function creates a simple mapping based on the first 4 characters.

    Args:
        tickers: List of ticker symbols

    Returns:
        Dictionary mapping ticker roots to potential trading names
    """
    return {t[:4]: t[:4] for t in tickers if len(t) >= 4}
