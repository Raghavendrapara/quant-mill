from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests
import pandas as pd

from quant_signal.config import DEFAULT_CONFIG


@dataclass
class YahooSearchConfig:
    """
    Config for Yahoo symbol search.

    region: e.g. "IN" for India, "US" for United States
    lang:   UI language; doesn't affect symbols
    quotes_count: max quotes to return
    """
    region: str = "IN"
    lang: str = "en-US"
    quotes_count: int = 10


def search_yahoo_symbols(
    query: str,
    cfg: Optional[YahooSearchConfig] = None,
    exchange_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search Yahoo Finance for symbols matching `query`.

    Args:
        query: partial company name or ticker fragment ("tcs", "reliance", "infy")
        cfg:   YahooSearchConfig; controls region/lang/quotes_count
        exchange_filter:
            - None: return all exchanges that match
            - "NSI" or "NSE": filter to NSE
            - "BSE": filter to BSE (BSE India)
            - any other code from Yahoo's 'exchange' field

    Returns:
        DataFrame with columns:
            symbol      (e.g. "TCS.NS")
            shortname   (company display name)
            longname    (full name if available)
            exchange    (e.g. "NSI", "BSE")
            quoteType   ("EQUITY", "ETF", etc.)
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must be non-empty.")

    if cfg is None:
        cfg = YahooSearchConfig()

    params = {
        "q": query,
        "quotesCount": cfg.quotes_count,
        "newsCount": 0,
        "lang": cfg.lang,
        "region": cfg.region,
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (quant-mill; Python requests)",
        "Accept": "application/json",
    }
    financialDataSourceUrl = DEFAULT_CONFIG.search.yahoo_search_url

    resp = requests.get(financialDataSourceUrl, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    quotes = data.get("quotes", []) or []
    if not quotes:
        return pd.DataFrame(columns=["symbol", "shortname", "longname", "exchange", "quoteType"])

    rows: List[dict] = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue

        exch = (q.get("exchange") or "").upper()
        if exchange_filter is not None:
            # normalize filter ("nse" -> "NSI", "bse" -> "BSE") for convenience
            f = exchange_filter.strip().upper()
            if f == "NSE":
                f = "NSI"
            if exch != f:
                continue

        rows.append(
            {
                "symbol": sym,
                "shortname": q.get("shortname") or "",
                "longname": q.get("longname") or "",
                "exchange": exch,
                "quoteType": q.get("quoteType") or "",
            }
        )

    return pd.DataFrame(rows)
