# run_real_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from causal.linear_gaussian import LinearGaussianSCM
from my_scenario import GenericEnv, run_pipeline

# 1. 加载并预处理数据
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original'
cols = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
df = pd.read_csv(url, delim_whitespace=True, header=None, names=cols)
df = df.dropna().drop(['model year','origin','car name'], axis=1)
# 为了适配二值发现算法，将目标 mpg 离散化（例如按中位数切分）
df['mpg_high'] = (df['mpg'] > df['mpg'].median()).astype(float)
var_names = ['cylinders','displacement','horsepower','weight','acceleration','mpg_high']
data = df[var_names].values

# 2. 定义你假设的因果结构（或让算法去发现）
# 这里使用一个基于领域知识的简单假设：车重和马力影响油耗
structure = {
    'cylinders': [],
    'displacement': ['cylinders'],
    'horsepower': ['displacement'],
    'weight': ['displacement'],
    'acceleration': ['horsepower', 'weight'],
    'mpg_high': ['weight', 'horsepower', 'acceleration']
}

# 3. 用数据拟合 LinearGaussianSCM
lg_scm = LinearGaussianSCM(structure, var_names)
lg_scm.fit([dict(zip(var_names, row)) for row in data])
print("拟合的系数示例 (weight -> mpg_high):", lg_scm.coeffs.get('mpg_high', {}).get('weight'))

# 4. 将拟合好的SCM包装成环境，接入现有分析流水线
actions = [{"type": "do_weight_high"}, {"type": "do_weight_low"}]
def action_to_intervention(action):
    # 根据实际情况定义干预，例如将weight设为高/低值
    return {"weight": 3500.0} if action["type"] == "do_weight_high" else {"weight": 2000.0}

env = GenericEnv(lg_scm, actions, action_to_intervention)

# 5. 复用 my_scenario.py 中的 run_pipeline 进行分析
# 注意：run_pipeline 内部会调用 interventional_discovery，它需要 SCM 支持 do()
print("\n--- 在 Auto MPG 数据上运行因果发现与推理 ---")
# 由于真实数据没有“真值”边，这里主要看发现的结构是否与领域知识相符
_, direct = interventional_discovery(lg_scm, var_names, n_samples=200, verbose=False)
print("\n发现的结构 (部分):")
for (u,v), w in sorted(direct.items(), key=lambda x:-x[1])[:5]:
    print(f"  {u:12s} -> {v:12s}  (强度 {w:.2f})")

# 6. 进行反事实推理
env.reset()
factual = env.do({"type": "do_weight_high"})
cf = env.counterfactual({"type": "do_weight_low"})
print(f"\n反事实分析 (weight -> mpg_high):")
print(f"  事实: weight=高 -> mpg_high={factual['mpg_high']:.2f}")
print(f"  反事实: weight=低 -> mpg_high={cf['mpg_high']:.2f}")