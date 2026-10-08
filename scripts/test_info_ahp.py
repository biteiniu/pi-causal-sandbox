"""单独测试 info 和 ahp，不依赖完整沙盒"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from causal.info import entropy, binary_entropy, gini, information_gain, mutual_information
from policy.ahp import AHPWeights


# ---------- 1. 信息度量 ----------

def test_entropy():
    print("\n[1] 熵 / 信息增益 / 基尼")
    print(f"  H(均匀二值)   = {binary_entropy(0.5):.4f}  (期望 1.0)")
    print(f"  H(确定二值)   = {binary_entropy(1.0):.4f}  (期望 0.0)")
    print(f"  H([0.5,0.5]) = {entropy([0.5, 0.5]):.4f}  (期望 1.0)")
    print(f"  H([1/3]*3)   = {entropy([1/3, 1/3, 1/3]):.4f}  (期望 ~1.585)")
    print(f"  Gini([0,1])  = {gini([0, 1]):.4f}  (期望 0.5)")
    print(f"  Gini([1,1])  = {gini([1, 1]):.4f}  (期望 0.0)")


def test_information_gain():
    print("\n[2] 信息增益（无噪声人工数据）")

    data = []
    for _ in range(20):
        data.append(({"Force_A": 1.0}, {"Fallen_B": 1.0}))  # 推 -> 全倒
        data.append(({"Force_A": 0.0}, {"Fallen_B": 0.0}))  # 不推 -> 全不倒

    ig_push = information_gain(data, {"type": "push", "force": 1.0}, "Fallen_B")
    ig_obs = information_gain(data, {"type": "observe"}, "Fallen_B")
    print(f"  IG(do(push))    = {ig_push:.4f}  (期望 1.0)")
    print(f"  IG(do(observe)) = {ig_obs:.4f}  (期望 0.0)")


def test_mutual_information():
    print("\n[3] 互信息")
    x = [0, 0, 1, 1, 0, 0, 1, 1]
    y = [0, 1, 0, 1, 0, 1, 0, 1]  # 独立
    print(f"  MI(独立)  = {mutual_information(x, y):.4f}  (期望 0.0)")
    y2 = x[:]  # 完全相关
    print(f"  MI(相同)  = {mutual_information(x, y2):.4f}  (期望 1.0)")


# ---------- 2. AHP ----------

class FakePi:
    """供 AHP 调用的最小 π 桩"""
    def _info_gain(self, action, state):
        return 0.9 if action["type"] == "push" else 0.2

    def _goal_reward(self, action, state):
        return 1.0 if action["type"] == "push" else 0.0


def test_ahp():
    print("\n[4] AHP 权重与一致性")
    ahp = AHPWeights(lam=0.3)
    cr, ok = ahp.consistency_check()
    print(f"  lambda_max = {ahp.lambda_max:.4f}")
    print(f"  CR         = {cr:.4f}  ({'通过' if ok else '未通过'})")
    print(f"  权重:")
    for c, w in zip(ahp.criteria, ahp.w):
        print(f"    {c:15s} {w:.4f}")


def test_ahp_scoring():
    print("\n[5] AHP 动作评分")
    ahp = AHPWeights()
    pi = FakePi()
    state = {"observations": {}, "depth": 0}
    actions = [
        {"type": "push", "force": 1.0},
        {"type": "observe"},
        {"type": "wait"},
    ]
    scores = {}
    for a in actions:
        s = ahp.score(a, state, pi)
        scores[a["type"]] = s
        print(f"  {a['type']:8s} -> S = {s:.4f}")
    best = max(scores, key=scores.get)
    print(f"  最优动作: {best}  (期望 push)")


def test_ahp_meta_update():
    print("\n[6] AHP 元学习更新")
    ahp = AHPWeights()
    pi = FakePi()
    state = {"observations": {}, "depth": 0}
    action = {"type": "push", "force": 1.0}

    w_before = ahp.w.copy()
    for _ in range(20):
        ahp.update_from_reward(action, reward=1.0, causal_state=state, pi=pi)
    ahp.score(action, state, pi)
    w_after = ahp.w

    print(f"  更新前权重: {w_before.round(4)}")
    print(f"  更新后权重: {w_after.round(4)}")
    print(f"  权重变化量: {np.abs(w_after - w_before).round(4)}")


if __name__ == "__main__":
    print("=" * 60)
    print("info + ahp 单模块测试")
    print("=" * 60)

    test_entropy()
    test_information_gain()
    test_mutual_information()
    test_ahp()
    test_ahp_scoring()
    test_ahp_meta_update()

    print("\n" + "=" * 60)
    print("完成。")
    print("=" * 60)