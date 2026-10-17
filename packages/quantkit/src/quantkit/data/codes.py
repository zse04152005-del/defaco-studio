"""Security code conventions.

Canonical code: ``600519.SH`` / ``000001.SZ`` / ``830799.BJ``.
"""
from __future__ import annotations

import datetime as dt


def to_canonical(code: str) -> str:
    """Accept 'sh.600519', 'sh600519', '600519.SH' or '600519' and return '600519.SH'."""
    c = code.strip()
    if "." in c and c[:2].lower() in ("sh", "sz", "bj"):  # baostock: sh.600519
        ex, num = c.split(".")
        return f"{num}.{ex.upper()}"
    if c[:2].lower() in ("sh", "sz", "bj") and c[2:].isdigit():  # sina: sh600519
        return f"{c[2:]}.{c[:2].upper()}"
    if "." in c:
        num, ex = c.split(".")
        return f"{num}.{ex.upper()}"
    return f"{c}.{exchange_of(c)}"


BSE_PREFIXES = ("43", "83", "87", "88", "92")  # Beijing Stock Exchange (920xxx since 2024)


def exchange_of(num: str) -> str:
    if num.startswith(BSE_PREFIXES):
        return "BJ"
    if num.startswith(("6", "9")):
        return "SH"
    if num.startswith(("0", "2", "3")):
        return "SZ"
    return "BJ"


def to_baostock(code: str) -> str:
    num, ex = to_canonical(code).split(".")
    return f"{ex.lower()}.{num}"


def board_of(code: str) -> str:
    num = to_canonical(code).split(".")[0]
    if num.startswith("688") or num.startswith("689"):
        return "STAR"
    if num.startswith(("300", "301")):
        return "ChiNext"
    if to_canonical(code).endswith(".BJ"):
        return "BSE"
    return "Main"


CHINEXT_REFORM = dt.date(2020, 8, 24)  # ChiNext switched to a 20% daily limit


def limit_pct(code: str, date: dt.date, is_st: bool) -> float:
    """Daily price-limit band for a stock on a date (ignores first trading days after IPO)."""
    board = board_of(code)
    if board == "STAR":
        return 0.20
    if board == "BSE":
        return 0.30
    if board == "ChiNext" and date >= CHINEXT_REFORM:
        return 0.20
    return 0.05 if is_st else 0.10
