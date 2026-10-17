"""Shenwan (SW2021) level-1 industry names keyed by the first two digits of the industry code.

Verified against known constituents (e.g. 600519 -> 34 食品饮料, 300750 -> 63 电力设备,
600036 -> 48 银行). Prefixes not listed belong to the pre-2021 (SW2014) scheme and only
appear in older history.
"""
SW_L1 = {
    "11": "农林牧渔", "22": "基础化工", "23": "钢铁", "24": "有色金属", "27": "电子", "28": "汽车",
    "33": "家用电器", "34": "食品饮料", "35": "纺织服饰", "36": "轻工制造", "37": "医药生物",
    "41": "公用事业", "42": "交通运输", "43": "房地产", "45": "商贸零售", "46": "社会服务",
    "48": "银行", "49": "非银金融", "51": "综合", "61": "建筑材料", "62": "建筑装饰",
    "63": "电力设备", "64": "机械设备", "65": "国防军工", "71": "计算机", "72": "传媒",
    "73": "通信", "74": "煤炭", "75": "石油石化", "76": "环保", "77": "美容护理",
}


def sw_name(prefix: str) -> str:
    return SW_L1.get(str(prefix), f"旧口径{prefix}")
