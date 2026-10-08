"""pi_core + info + ahp 联动测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from policy.pi_core import PiCore


class FakeLearner:
    """模拟 learner，提供 int_buffer（干预数据）"""

    def __init__(self):
        self.int_buffer = []

    def add(self, intervention: dict, sample: dict):
        self.int_buffer.append((intervention, sample))


def build_actions():
    return [
        {"type": "push", "force": 1.0},
        {"type": "push", "force": 0.5},
        {"type": "observe"},
        {"type": "wait"},
    ]


def test_pi_no_data():
    print("\n[1] 无数据时：退回启发式")
    pi = PiCore(build_actions())
    state = {"observations": {}, "uncertainty": {}, "depth": 0}
    action = pi(state)
    print(f"  选中动作: {action}")
    print(f"  AHP权重: {pi.ahp.w.round(4)}")
    print(f"  模式: {pi.mode}")


def test_pi_with_data():
    print("\n[2] 有干预数据时：用真实信息增益")

    learner = FakeLearner()
    # 构造强因果：do(push) -> Fallen_B = 1；do(wait) -> Fallen_B = 0
    for _ in range(20):
        learner.add({"Force_A": 1.0}, {"Fallen_B": 1.0})
        learner.add({"Force_A": 0.0}, {"Fallen_B": 0.0})

    pi = PiCore(build_actions(), learner=learner)
    state = {"observations": {}, "uncertainty": {}, "depth": 0}
    action = pi(state)
    print(f"  选中动作: {action}")
    print(f"  各动作评分:")
    for k, v in pi.last_scores.items():
        print(f"    {k:8s} -> {v:.4f}")


def test_pi_learning_loop():
    print("\n[3] 模拟训练循环：观察权重与模式变化")

    learner = FakeLearner()
    actions = build_actions()
    pi = PiCore(actions, learner=learner)

    # 模拟若干步
    for step in range(30):
        state = {
            "observations": {"Fallen_B": 0.0},
            "uncertainty": {("Force_A", "Fallen_B"): 0.5},
            "depth": step,
        }
        action = pi(state)

        # 模拟环境：push 成功
        if action["type"] == "push":
            sample = {"Fallen_B": 1.0}
            reward = 1.0
        else:
            sample = {"Fallen_B": 0.0}
            reward = 0.1

        intervention = _action_to_intervention(action)
        learner.add(intervention, sample)
        pi.update(reward, state, action)

        if step % 5 == 0:
            print(
                f"  step {step:2d} | 动作={action['type']:8s} | "
                f"模式={pi.mode:8s} | "
                f"w[IG]={pi.ahp.w[0]:.4f} w[GR]={pi.ahp.w[1]:.4f}"
            )

    print("\n  最终 AHP 报告:")
    rpt = pi.report()
    for k, v in rpt["ahp"]["weights"].items():
        print(f"    {k:15s} {v:.4f}")
    print(f"    CR = {rpt['ahp']['CR']}  (consistent={rpt['ahp']['consistent']})")


def test_pi_force_selection():
    print("\n[4] 力度选择：force=1.0 vs 0.5")

    learner = FakeLearner()
    for _ in range(20):
        learner.add({"Force_A": 1.0}, {"Fallen_B": 1.0})
        learner.add({"Force_A": 0.5}, {"Fallen_B": 0.3})

    actions = [
        {"type": "push", "force": 1.0},
        {"type": "push", "force": 0.5},
        {"type": "observe"},
    ]
    pi = PiCore(actions, learner=learner)
    state = {"observations": {}, "uncertainty": {}, "depth": 0}

    # 手动算各动作评分
    print("  动作评分:")
    for a in actions:
        s = pi.ahp.score(a, state, pi)
        print(f"    push(f={a.get('force', '-')}) -> {s:.4f}")

    best = pi(state)
    print(f"  选中: {best}  (期望 force=1.0)")


def _action_to_intervention(action: dict) -> dict:
    if action["type"] == "push":
        return {"Force_A": action.get("force", 1.0)}
    elif action["type"] == "observe":
        return {}
    elif action["type"] == "wait":
        return {"Force_A": 0.0}
    return {}


if __name__ == "__main__":
    print("=" * 60)
    print("pi_core + info + ahp 联动测试")
    print("=" * 60)

    test_pi_no_data()
    test_pi_with_data()
    test_pi_learning_loop()
    test_pi_force_selection()

    print("\n" + "=" * 60)
    print("完成。")
    print("=" * 60)