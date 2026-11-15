# -*- coding: utf-8 -*-
"""相似证型鉴别的统计显著性检验（区间约束版）

数据约束（见 docs/syndrome_compare.json 的 meta.note）：
  only_a 中的症状：A 组出现率 >= 25%，B 组出现率 < 25%（不是 0）
  shared 中的症状：两组率均已知（完整值）

因此不能用双比例 z 检验直接比较（B 组真实率未知）。本模块：
  · 对率已知的 shared 症状做精确检验（双比例 z 检验 + BH-FDR 校正）
  · 对 only_a / only_b 报告"本组 >= 25%，对比组 < 25%"的确定性结论

统计方法：双比例 z 检验；Newcombe 混合评分 95% 置信区间；
          Benjamini-Hochberg 多重比较校正。
"""
import math


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _wilson(k, n, z=1.959964):
    if n <= 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def two_prop(k1, n1, k2, n2):
    """双比例 z 检验 + Newcombe 95% 置信区间"""
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se <= 0:
        return {"diff": p1 - p2, "p": 1.0, "lo": 0.0, "hi": 0.0}
    z = (p1 - p2) / se
    pv = 2 * (1 - _norm_cdf(abs(z)))
    l1, u1 = _wilson(k1, n1)
    l2, u2 = _wilson(k2, n2)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {"diff": p1 - p2, "p": pv, "lo": lo, "hi": hi}


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 校正，返回与输入等长的布尔列表"""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = -1
    for rank, idx in enumerate(order, 1):
        if pvals[idx] <= alpha * rank / m:
            kmax = rank
    sig = [False] * m
    for rank, idx in enumerate(order, 1):
        if rank <= kmax:
            sig[idx] = True
    return sig


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def annotate_pair(pair, alpha=0.05):
    """为一个证型对生成统计标注（前端据此显示显著性与置信区间）"""
    na = pair.get("n_a", 0)
    nb = pair.get("n_b", 0)
    out = {"shared": [], "only_a": [], "only_b": []}
    pv, tmp = [], []
    for x in pair.get("shared", []):
        k1 = int(round(x["a"] * na))
        k2 = int(round(x["b"] * nb))
        st = two_prop(k1, na, k2, nb)
        tmp.append((x, st)); pv.append(st["p"])
    sig = bh_fdr(pv, alpha)
    for (x, st), s in zip(tmp, sig):
        out["shared"].append({
            "w": x["w"], "a": x["a"], "b": x["b"],
            "diff": round(st["diff"], 4), "p": st["p"], "stars": stars(st["p"]),
            "ci": [round(st["lo"], 4), round(st["hi"], 4)], "significant": bool(s),
        })
    for key in ("only_a", "only_b"):
        for x in pair.get(key, []):
            out[key].append({
                "w": x["w"], "rate": x["rate"],
                "constraint": "本组出现率 >= 25%，对比组 < 25%",
                "discriminative": True,
            })
    out["meta"] = {
        "n_a": na, "n_b": nb, "alpha": alpha,
        "fdr": "Benjamini-Hochberg",
        "ci_method": "Newcombe hybrid score (95%)",
        "limitation": "only_a/only_b 的对比组真实率未在数据文件中提供，故不做 p 值检验",
    }
    return out


def summarize(annotated):
    """把标注结果转成一句话结论（供界面展示）"""
    sh = annotated.get("shared", [])
    sig = [x for x in sh if x["significant"]]
    ns = [x for x in sh if not x["significant"]]
    parts = []
    if sig:
        parts.append("共有症状中有 %d 项差异显著（FDR 校正后）" % len(sig))
    if ns:
        parts.append("%d 项无显著差异，不能作为鉴别依据" % len(ns))
    na = annotated["meta"]["n_a"]
    nb = annotated["meta"]["n_b"]
    parts.append("对比样本量 n=%d vs %d" % (na, nb))
    return "；".join(parts)
