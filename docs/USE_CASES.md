# 这个项目能干啥

## 一句话

**给你一组变量和它们的观测/干预数据，它能帮你：**

1. 找出**谁导致谁**（因果结构）
2. 预测**"如果我做 X 会怎样"**（干预效应）
3. 判断**"这个因果效应能不能从现有数据估计"**（可识别性）
4. 回答**"如果当时不这样，会怎样"**（反事实）
5. 找出**最省力的干预方案**（规划）

---

## 三种典型用法

### 用法 1：我想知道谁导致谁

**场景**：一堆变量，不知道谁影响谁。

```python
from causal.interventional_discovery import interventional_discovery

# 你能干预的环境（SCM 或真实实验接口）
_, edges = interventional_discovery(scm, var_names, n_samples=300)

# edges = {('A','B'): 0.95, ('B','C'): 0.88, ...}
# 表示 A 导致 B，强度 0.95