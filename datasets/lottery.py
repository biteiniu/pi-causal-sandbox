"""
大乐透数据加载与统计特征提取。
"""

import os
import csv
from typing import Dict, List


def load_history(path: str = None) -> List[dict]:
    """
    从 CSV 加载大乐透历史。
    格式（无表头）：f1,f2,f3,f4,f5,b1,b2
    文件不存在时返回空列表。
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "lottery_history.csv",
        )

    if not os.path.exists(path):
        print(f"  [lottery] {path} 不存在。")
        print(f"  [lottery] 请先运行：python datasets/download_lottery.py")
        return []

    draws = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                nums = [int(x) for x in row[:7]]
            except ValueError:
                continue
            draws.append({
                "front": sorted(nums[:5]),
                "back": sorted(nums[5:7]),
            })

    print(f"  [lottery] 从 {path} 加载 {len(draws)} 期")
    return draws


def extract_features(draws: List[dict]) -> List[dict]:
    """把每期开奖转成统计特征。"""
    out = []
    for d in draws:
        f, b = d["front"], d["back"]
        out.append({
            "sum_front":  float(sum(f)),
            "span_front": float(max(f) - min(f)),
            "odd_count":  float(sum(1 for x in f if x % 2 == 1)),
            "big_count":  float(sum(1 for x in f if x > 17)),
            "sum_back":   float(sum(b)),
            "back_span":  float(max(b) - min(b)),
        })
    return out