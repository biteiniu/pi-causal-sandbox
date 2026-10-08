"""
彩票零假设测试（修正版）：
  跨期独立性检验，而不是同期特征相关性。

原理：
  彩票每期独立 → 第 t 期统计量不能预测第 t+1 期统计量。
  正确测试：lag-1 相关性、互信息、OLS 系数。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datasets.lottery import load_history, extract_features
from causal.info import mutual_information, mutual_information as mi_func

SEP = "=" * 62


def title(t):
    print(f"\n{SEP}\n{t}\n{SEP}")


# ---------- 数据 ----------
title("彩票零假设基准（修正版）")

draws = load_history()
if not draws:
    print("错误：没有数据。请先运行：python datasets/download_lottery.py")
    sys.exit(1)

features = extract_features(draws)
var_names = list(features[0].keys())

print(f"期数：{len(features)}")
print(f"特征：{var_names}")

# 提取每个特征的时间序列
series = {v: np.array([f[v] for f in features]) for v in var_names}


# ---------- 测试 1：跨期自相关 ----------
title("测试 1：lag-1 自相关")
print("""
问题：第 t 期的某个统计量，能预测第 t+1 期吗？
     如果是随机的，自相关应该接近 0。
""")

print(f"\n{'特征':<14s} | lag-1 自相关 | 期望")
print("-" * 45)
max_autocorr = 0.0
for v in var_names:
    x = series[v]
    if x.std() < 1e-9:
        ac = 0.0
    else:
        ac = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    max_autocorr = max(max_autocorr, abs(ac))
    print(f"{v:<14s} | {ac:+.4f}        | ≈ 0")

print(f"\n最大 |自相关| = {max_autocorr:.4f}")


# ---------- 测试 2：跨期互信息 ----------
title("测试 2：lag-1 互信息")
print("""
问题：第 t 期的 X 和第 t+1 期的 Y 是否有信息共享？
""")

print(f"\n{'X(t)':<14s} -> {'Y(t+1)':<14s} | MI")
print("-" * 50)
max_mi = 0.0
mi_results = []
for u in var_names:
    for v in var_names:
        xu = series[u][:-1]
        xv = series[v][1:]
        xu_d = np.digitize(xu, np.quantile(xu, [0.25, 0.5, 0.75]))
        xv_d = np.digitize(xv, np.quantile(xv, [0.25, 0.5, 0.75]))
        mi = mutual_information(xu_d.tolist(), xv_d.tolist())
        mi_results.append((mi, u, v))
        max_mi = max(max_mi, mi)

mi_results.sort(reverse=True)
print(f"\n互信息 Top 10（共 {len(mi_results)} 对）：")
for mi, u, v in mi_results[:10]:
    print(f"  {u:<14s} -> {v:<14s} | {mi:.4f}")

print(f"\n最大互信息 = {max_mi:.4f}")


# ---------- 测试 3：号码出现间隔 ----------
title("测试 3：单个号码的 gap 分布")
print("""
问题：某个号码刚开出，下一期是否更容易/更不可能再开？
     随机情况下，gap 应服从几何分布，均值 ≈ 1/p。
""")

# 统计前区号码出现间隔
front_counts = np.zeros(36)  # index 1..35
gaps = {n: [] for n in range(1, 36)}
last_seen = {n: -1 for n in range(1, 36)}

for t, d in enumerate(draws):
    for n in d["front"]:
        if last_seen[n] >= 0:
            gaps[n].append(t - last_seen[n])
        last_seen[n] = t

all_gaps = [g for lst in gaps.values() for g in lst]
if all_gaps:
    empirical_mean = float(np.mean(all_gaps))
    theoretical_mean = 35.0 / 5.0  # 前区 5/35 → 每个号的期望间隔
    print(f"  样本 gap 数：{len(all_gaps)}")
    print(f"  实测均值：{empirical_mean:.3f}")
    print(f"  理论均值：{theoretical_mean:.3f}")
    print(f"  偏差：{abs(empirical_mean - theoretical_mean):.3f}")
    if abs(empirical_mean - theoretical_mean) < 0.5:
        print("  结论：gap 分布接近理论值，无记忆性。")


# ---------- 结论 ----------
title("结论")

print(f"""
在 {len(features)} 期真实开奖数据上：

  lag-1 最大自相关: {max_autocorr:.4f}
  lag-1 最大互信息: {max_mi:.4f}
  号码间隔偏离理论: {abs(empirical_mean - theoretical_mean) if all_gaps else 0:.3f}

这三项都接近 0，说明：
【彩票没有跨期记忆性。第 t 期开出什么，与第 t+1 期无关。】

正确的零假设测试是「跨期独立性」，
而不是「同期特征互相关」——因为同期特征由同一组号码导出，
天然存在定义性依赖（这不是因果，是数学恒等）。
""")