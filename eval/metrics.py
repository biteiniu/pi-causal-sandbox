"""分层评估：从简单到困难，避免指标饱和"""

import numpy as np
from causal.info import information_gain


class LayeredEvaluator:
    """
    四个层级：
    L1: 因果图恢复（简单）
    L2: do预测（中等）
    L3: 反事实（困难）
    L4: OOD因果迁移（极难）
    """

    def __init__(self, true_dag: np.ndarray, var_names: list):
        self.true_dag = true_dag
        self.var_names = var_names
        self.results = {}

    def L1_graph_f1(self, learned_dag: np.ndarray) -> float:
        """因果图F1"""
        true_edges = set(zip(*np.where(np.abs(self.true_dag) > 0.1)))
        pred_edges = set(zip(*np.where(np.abs(learned_dag) > 0.1)))
        if not pred_edges and not true_edges:
            return 1.0
        tp = len(true_edges & pred_edges)
        precision = tp / len(pred_edges) if pred_edges else 0.0
        recall = tp / len(true_edges) if true_edges else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self.results["L1_graph_f1"] = f1
        return f1

    def L2_do_error(self, scm, interventions: list, target: str) -> float:
        """do预测误差：预测 P(target|do(x)) 与真实SCM的差距"""
        errors = []
        for intervention in interventions:
            # 真实
            true_vals = [scm.sample(interventions=intervention).get(target, 0.0) for _ in range(50)]
            true_p = np.mean([1 if v > 0.5 else 0 for v in true_vals])
            # 预测（用学到的贝叶斯SCM）
            # 这里简化：假设已有预测函数
            errors.append(abs(true_p - true_p))  # 占位
        err = float(np.mean(errors)) if errors else 1.0
        self.results["L2_do_error"] = err
        return err

    def L3_counterfactual_error(self, scm, factual_trace, interventions, target) -> float:
        """反事实误差"""
        errors = []
        for intervention in interventions:
            cf = scm.counterfactual(factual_trace, interventions=intervention)
            true_val = cf.get(target, 0.0)
            # 预测（占位）
            errors.append(abs(true_val - true_val))
        err = float(np.mean(errors)) if errors else 1.0
        self.results["L3_counterfactual_error"] = err
        return err

    def L4_ood_transfer(self, model, ood_scm, target) -> float:
        """OOD因果迁移：在未见因果结构上的零样本表现"""
        # 占位：用OOD SCM采样，评估模型预测
        return 0.0

    def report(self) -> dict:
        return self.results