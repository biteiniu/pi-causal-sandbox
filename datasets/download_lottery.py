"""
自动下载大乐透全部历史开奖数据。

数据源：体彩官方 API
  https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry
  gameNo=85

输出：datasets/lottery_history.csv
"""

import os
import csv
import time
import requests
from typing import List, Dict

API_URL = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.lottery.gov.cn/",
    "Accept": "application/json, text/plain, */*",
}

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "lottery_history.csv")


def fetch_page(page_no: int, page_size: int = 100, max_retries: int = 3) -> dict:
    params = {
        "gameNo": 85,
        "provinceId": 0,
        "pageSize": page_size,
        "isVerify": 1,
        "pageNo": page_no,
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(
                API_URL, params=params, headers=HEADERS, timeout=15
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"第 {page_no} 页拉取失败（已重试 {max_retries} 次）：{e}"
                )
            time.sleep(2 ** attempt)
    return {}


def parse_draw(item: dict) -> dict:
    """
    体彩 API 返回的每条记录字段：
      lotteryDrawNum    -> 期号，如 "24001"
      lotteryDrawTime   -> 开奖日期，如 "2024-01-01"
      lotteryDrawResult -> 号码，空格分隔的 7 个数字：前5 + 后2
    """
    nums = item["lotteryDrawResult"].split()
    nums = [int(x) for x in nums]
    if len(nums) != 7:
        raise ValueError(f"号码数异常：{item['lotteryDrawResult']}")

    front = sorted(nums[:5])
    back = sorted(nums[5:])

    return {
        "period": item["lotteryDrawNum"],
        "date": item["lotteryDrawTime"],
        "f1": front[0], "f2": front[1], "f3": front[2],
        "f4": front[3], "f5": front[4],
        "b1": back[0], "b2": back[1],
    }


def download_all(verbose: bool = True) -> List[dict]:
    all_draws = []
    page_no = 1

    first = fetch_page(page_no)
    if not first or "value" not in first:
        raise RuntimeError("API 返回格式异常：" + str(first)[:200])

    # 调试：打印第一条 item 的字段名
    sample_item = first["value"]["list"][0]
    if verbose:
        print(f"  样本字段：{list(sample_item.keys())}")
        print(f"  样本号码：{sample_item.get('lotteryDrawResult')}")

    total_pages = first["value"]["pages"]
    total_count = first["value"]["total"]

    if verbose:
        print(f"共 {total_count} 期，{total_pages} 页")

    for item in first["value"]["list"]:
        all_draws.append(parse_draw(item))

    if verbose:
        print(f"  第 1/{total_pages} 页完成（累计 {len(all_draws)} 期）")

    for page_no in range(2, total_pages + 1):
        data = fetch_page(page_no)
        for item in data["value"]["list"]:
            all_draws.append(parse_draw(item))

        if verbose and (page_no % 10 == 0 or page_no == total_pages):
            print(
                f"  第 {page_no}/{total_pages} 页完成"
                f"（累计 {len(all_draws)} 期）"
            )

        time.sleep(0.5)

    return all_draws


def save_csv(draws: List[dict], path: str = OUTPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    draws_sorted = sorted(draws, key=lambda d: d["date"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for d in draws_sorted:
            writer.writerow([
                d["f1"], d["f2"], d["f3"], d["f4"], d["f5"],
                d["b1"], d["b2"],
            ])
    print(f"已保存 {len(draws_sorted)} 期到 {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("大乐透历史开奖数据下载器")
    print("=" * 60)
    draws = download_all(verbose=True)
    save_csv(draws)
    print("\n完成。")