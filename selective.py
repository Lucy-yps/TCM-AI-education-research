# -*- coding: utf-8 -*-
"""选择性预测模块：正确性探针 + 高合规模式。

方法来源：Phillips et al., "Entropy Alone is Insufficient for Safe Selective
Prediction in LLMs", arXiv:2603.21172 [cs.CL], 2026.

原论文发现仅依赖熵的不确定性度量会导致不可靠的弃答行为，需结合
"正确性探针"（correctness probe）信号。本项目将探针适配为：
  输入 —— 分类器概率分布的 5 维不确定性特征（不依赖标签）
  输出 —— P(分类器对该样本判断正确)

本项目独立实验结论（683 例去泄漏测试集，见 docs/前沿方法应用报告.md）：
  · 探针 AUC 0.9100，高于 p_max(0.9086) 与纯熵(0.9023)
  · 整体最优权重 alpha=0（即 p_max 已足够），组合在常规区间无增益
  · 但在高精度区间（99.5% 目标准确率）组合覆盖率 60.8% vs 仅概率 58.7%
  因此本模块的定位是「高合规模式」：仅在对弃答可靠性要求更高时启用。
"""
import json
import math
import os

_PROBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "docs", "selective_probe.json")
_PROBE = None
_ALPHA = 0.0        # 默认 0：与现有 p_max 策略等价（保守，不改变现有行为）


def load_probe():
    """载入正确性探针参数；缺失时返回 None（模块自动降级）。"""
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    if not os.path.exists(_PROBE_PATH):
        return None
    try:
        with open(_PROBE_PATH, encoding="utf-8") as f:
            _PROBE = json.load(f)
    except (OSError, ValueError):
        _PROBE = None
    return _PROBE


def uncertainty_features(proba):
    """从分类器概率向量提取 5 维不确定性特征（顺序须与探针系数一致）。"""
    p = [float(x) for x in proba]
    s = sum(p) or 1.0
    p = [x / s for x in p]
    k = len(p)
    pmax = max(p)
    ent = -sum(x * math.log(x + 1e-12) for x in p)
    H = ent / math.log(k) if k > 1 else 0.0
    srt = sorted(p, reverse=True)
    margin = (srt[0] - srt[1]) if len(srt) > 1 else 1.0
    mean = sum(p) / k
    var = sum((x - mean) ** 2 for x in p) / k
    return [pmax, H, 1.0 - H, margin, var]


def probe_score(proba):
    """正确性探针输出：P(分类器判断正确)。探针缺失时返回 None。"""
    pr = load_probe()
    if not pr:
        return None
    x = uncertainty_features(proba)
    z = float(pr.get("intercept", 0.0))
    for c, v in zip(pr.get("coef", []), x):
        z += float(c) * v
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


def combined_score(proba, alpha=None):
    """组合分数：alpha·(1-H_norm) + (1-alpha)·probe。

    alpha 默认取 _ALPHA（0.0），此时等价于 p_max 策略。
    """
    a = _ALPHA if alpha is None else float(alpha)
    x = uncertainty_features(proba)
    if a <= 0.0:
        return x[0]                       # 退化为 p_max
    ps = probe_score(proba)
    if ps is None:
        return x[0]
    return a * x[2] + (1.0 - a) * ps


def info():
    """模块状态（供 /api/health 暴露）。"""
    pr = load_probe()
    return {
        "available": pr is not None,
        "alpha_default": _ALPHA,
        "features": (pr or {}).get("feature_names", []),
        "train_n": (pr or {}).get("train_n"),
        "source": (pr or {}).get("source", ""),
        "note": "高合规模式：alpha>0 时启用组合信号（实测仅在目标准确率≥99.5%区间有增益）",
    }
