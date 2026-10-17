"""Pinyin initials for the securities search (scripts/export_securities.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from export_securities import initials  # noqa: E402


def test_phrase_reading_first():
    assert initials("贵州茅台").split("|")[0] == "gzmt"
    assert initials("长江电力").split("|")[0] == "cjdl"      # 长 read chang in 长江
    assert initials("招商银行").split("|")[0] == "zsyh"      # 行 read hang in 银行
    assert initials("重庆啤酒").split("|")[0] == "cqpj"      # 重 read chong in 重庆


def test_heteronym_variants_and_ascii():
    assert "zxyh" in initials("中信银行").split("|")
    assert initials("*ST国华").startswith("st")
    assert initials("TCL科技").split("|")[0] == "tclkj"


def test_exported_table_if_present():
    p = ROOT / "web" / "public" / "demo-data" / "securities_all.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = {r[0]: r for r in d["rows"]}
    assert len(rows) > 5000 and sum(r[5] for r in d["rows"]) >= 1800
    assert rows["600519.SH"][2].startswith("gzmt") and rows["600519.SH"][5] == 1
    assert any(c.endswith(".BJ") for c in rows)
