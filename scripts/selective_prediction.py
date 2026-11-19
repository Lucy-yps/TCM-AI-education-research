# -*- coding: utf-8 -*-
"""选择性预测：熵 + 正确性探针（前沿方法适配）

方法来源：Phillips et al., "Entropy Alone is Insufficient for Safe Selective
Prediction in LLMs", arXiv:2603.21172 [cs.CL], 2026.

原论文结论：仅依赖熵的不确定性度量会导致不可靠的弃答行为，
需结合"正确性探针"（correctness probe）信号。

本实现为适配版，差异在于：
  1) 任务：原文为开放域问答；本项目为中医证型辨识（16 类闭集分类）
  2) 数据：本项目用 TCM-SD 真实电子病历
  3) 严格隔离：探针训练数据来自 train/dev，与 683 例去泄漏测试集无重叠
"""
import io, json, os, random, sys
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = sys.argv[1]
os.chdir(BASE)
sys.path.insert(0, BASE)

LABEL_MAP = {
    "气滞血瘀证": "气滞血瘀", "肝胃不和证": "肝胃不和", "肝阳上亢证": "肝阳上亢",
    "肝郁气滞证": "肝郁气滞", "心血虚证": "心血虚", "气血两虚证": "气血两虚",
    "痰湿内阻证": "痰湿内阻", "肝郁脾虚证": "肝郁脾虚", "肝胆湿热证": "肝胆湿热",
    "肾气虚证": "肾气虚", "大肠湿热证": "大肠湿热", "中气下陷证": "中气下陷",
    "脾胃气虚证": "脾胃气虚", "阳虚寒凝证": "阳虚寒凝", "肺阴虚证": "肺阴虚",
    "肾阳虚证": "肾阳虚",
}

def join_text(r):
    return "。".join(x for x in [r.get("chief_complaint",""), r.get("description",""),
                                   r.get("detection","")] if x)

def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab = r.get("norm_syndrome") or r.get("syndrome")
            if lab in LABEL_MAP:
                out.append({"text": join_text(r), "gold": LABEL_MAP[lab]})
    return out

print("载入数据 ...")
tr = load_jsonl("external_data/train_dev/train.json")
dv = load_jsonl("external_data/train_dev/dev.json")
te_all = load_jsonl("external_data/tcm_sd_test.json")
print("  train %d | dev %d | test(16类) %d" % (len(tr), len(dv), len(te_all)))

# 去泄漏：与项目统一口径（见 scripts/confusion_report.py）
# 注意：对比集必须包含 train/dev 的【全部】文本，而不只是命中 16 类的子集，
#       否则会漏掉重叠样本（实测：只放子集会得到 686，全量得到 683）。
def ft(r):
    return "。".join(x for x in [r.get("chief_complaint",""), r.get("description",""),
                                   r.get("detection","")] if x)
seen = set()
for _p in ("external_data/train_dev/train.json", "external_data/train_dev/dev.json"):
    if os.path.exists(_p):
        with open(_p, encoding="utf-8") as _f:
            for _line in _f:
                if _line.strip():
                    seen.add(ft(json.loads(_line)))
test = [c for c in te_all if c["text"] not in seen]
print("  去泄漏后测试集: %d  (剔除 %d)" % (len(test), len(te_all) - len(test)))

# 探针数据：train+dev，与测试集隔离
test_keys = set(c["text"] for c in test)
pool = [c for c in (tr + dv) if c["text"] not in test_keys]
random.Random(42).shuffle(pool)
n_fit = int(len(pool) * 0.6)
fit_pool = pool[:n_fit]
cal_pool = pool[n_fit:]
print("  探针拟合集 %d | 阈值校准集 %d" % (len(fit_pool), len(cal_pool)))
print("  类别分布:", Counter(c["gold"] for c in test).most_common(5))
# ---------- 分类器探针 ----------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def unc_features(proba):
    """不确定性特征（不依赖标签，可用无标注数据）"""
    p = np.asarray(proba, dtype=float)
    p = p / max(p.sum(), 1e-12)
    pmax = float(p.max())
    ent = float(-(p * np.log(p + 1e-12)).sum())
    H = ent / np.log(len(p))                 # 归一化熵
    srt = np.sort(p)[::-1]
    margin = float(srt[0] - srt[1]) if len(srt) > 1 else 1.0
    var = float(p.var())
    return [pmax, H, 1.0 - H, margin, var]

print("加载分类器 ...")
import pickle
with open("external_data/tcm_clf.joblib", "rb") as f:
    clf = pickle.load(f)
classes = list(clf.classes_)
print("  类别数: %d" % len(classes))

def proba_of(texts):
    return clf.predict_proba(texts)

# ---------- 构造探针训练集（train.json，与测试集隔离）----------
random.Random(7).shuffle(pool)
fit_pool, cal_pool = pool[:3000], pool[3000:6000]
print("  探针拟合集 %d | 阈值校准集 %d" % (len(fit_pool), len(cal_pool)))

def build_xy(cases):
    X, y, correct = [], [], []
    B = 500
    for i in range(0, len(cases), B):
        chunk = cases[i:i+B]
        texts = [c["text"] for c in chunk]
        P = proba_of(texts)
        for c, p in zip(chunk, P):
            X.append(unc_features(p))
            idx = int(np.argmax(p))
            ok = 1 if classes[idx] == c["gold"] else 0
            y.append(ok)
            correct.append(ok)
    return np.array(X), np.array(y)

print("计算探针拟合集特征 ...")
Xf, yf = build_xy(fit_pool)
print("  形状 %s | 正确率 %.4f" % (Xf.shape, yf.mean()))
print("计算阈值校准集特征 ...")
Xc, yc = build_xy(cal_pool)

probe = LogisticRegression(max_iter=2000, C=1.0)
probe.fit(Xf, yf)
print("探针已训练")

# 探针在拟合集/校准集上的判别力（AUC 近似：用秩相关）
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 抽样计算，避免 O(n^2)
    rs = np.random.RandomState(0)
    a = rs.choice(pos, min(600, len(pos)), replace=False)
    b = rs.choice(neg, min(600, len(neg)), replace=False)
    return float(((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()))

pc = probe.predict_proba(Xc)[:, 1]
print("  校准集 AUC(探针) = %.4f" % auc(yc, pc))
print("  校准集 AUC(1-熵) = %.4f" % auc(yc, Xc[:, 2]))
print("  校准集 AUC(pmax)  = %.4f" % auc(yc, Xc[:, 0]))
# ---------- 在去泄漏测试集上评测 ----------
print()
print("计算测试集特征（%d 例）..." % len(test))
Xt, yt = build_xy(test)
print("  分类器 Top-1 准确率: %.4f (n=%d)" % (yt.mean(), len(yt)))
pmax_t = Xt[:, 0]
H_t = Xt[:, 1]
probe_t = probe.predict_proba(Xt)[:, 1]
print("  测试集 AUC(探针)  = %.4f" % auc(yt, probe_t))
print("  测试集 AUC(pmax)  = %.4f" % auc(yt, pmax_t))
print("  测试集 AUC(1-熵)  = %.4f" % auc(yt, 1 - H_t))

def curve(score, y, thresholds):
    """给定分数与阈值，返回 (覆盖率, 保留样本准确率) 列表"""
    out = []
    for th in thresholds:
        keep = score >= th
        cov = float(keep.mean())
        acc = float(y[keep].mean()) if keep.sum() else float("nan")
        out.append((float(th), cov, acc, int(keep.sum())))
    return out

# 组合分数：等权（简洁、可解释；权重在拟合集上确定）
# 先用拟合集找最佳权重 alpha
best = None
pf = probe.predict_proba(Xf)[:, 1]
for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    s = alpha * (1 - Xf[:, 1]) + (1 - alpha) * pf
    a = auc(yf, s)
    if best is None or a > best[1]:
        best = (alpha, a)
ALPHA = best[0]
print()
print("在拟合集上选出的 alpha = %.1f (AUC=%.4f)" % (ALPHA, best[1]))
# 也看校准集是否支持同一 alpha
best_c = max([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0], key=lambda a: auc(yc, a*(1-Xc[:,1]) + (1-a)*pc))
print("校准集最优 alpha = %.1f  %s" % (best_c, "(一致)" if abs(best_c-ALPHA)<0.01 else "(不一致，取拟合集值并注明)"))

score_comb = ALPHA * (1 - H_t) + (1 - ALPHA) * probe_t

# 阈值搜索：对每个覆盖率目标，三种策略各自的保留准确率
targets = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
print()
print("=" * 78)
print("风险-覆盖对比（%d 例去泄漏测试集）" % len(test))
print("=" * 78)
print("%-8s | %-26s | %-26s | %-26s" % ("覆盖率", "仅概率 p_max", "仅熵 1-H", "组合 alpha=%.1f" % ALPHA))
print("%-8s | %-26s | %-26s | %-26s" % ("", "准确率(保留n)", "准确率(保留n)", "准确率(保留n)"))
print("-" * 78)
rows = []
for t in targets:
    cells = []
    for sc in [pmax_t, 1 - H_t, score_comb]:
        th = float(np.quantile(sc, 1 - t))
        keep = sc >= th
        cov = float(keep.mean())
        acc = float(yt[keep].mean()) if keep.sum() else float("nan")
        cells.append((acc, int(keep.sum()), cov))
    line = "%6.0f%%  | %6.4f (%4d)            | %6.4f (%4d)            | %6.4f (%4d)" % (
        t * 100, cells[0][0], cells[0][1], cells[1][0], cells[1][1], cells[2][0], cells[2][1])
    print(line)
    rows.append({"target": t, "pmax": cells[0], "entropy": cells[1], "combined": cells[2]})

# 关键指标：达到 99% / 97% 保留准确率时，各策略能覆盖多少
print()
print("=" * 78)
print("反向指标：要达到指定保留准确率，各策略最多能覆盖多少样本")
print("=" * 78)
print("%-14s | %-16s | %-16s | %-16s" % ("目标准确率", "仅概率", "仅熵", "组合"))
print("-" * 78)
reverse = []
for target_acc in [0.995, 0.99, 0.98, 0.97, 0.95]:
    cells = []
    for sc in [pmax_t, 1 - H_t, score_comb]:
        order = np.argsort(-sc)
        ys = yt[order]
        cum = np.cumsum(ys) / np.arange(1, len(ys) + 1)
        ok = np.where(cum >= target_acc)[0]
        cov = float((ok[-1] + 1) / len(ys)) if len(ok) else 0.0
        cells.append(cov)
    print("%-14s | %-16s | %-16s | %-16s" % (
        "%.1f%%" % (target_acc * 100),
        "%.1f%%" % (cells[0] * 100), "%.1f%%" % (cells[1] * 100), "%.1f%%" % (cells[2] * 100)))
    reverse.append({"target_acc": target_acc, "pmax": cells[0], "entropy": cells[1], "combined": cells[2]})

# 保存结果
out = {
    "method_source": "Phillips et al., arXiv:2603.21172 (2026)",
    "task": "TCM syndrome classification (16 classes, closed-set)",
    "test_n": int(len(yt)),
    "classifier_top1": float(yt.mean()),
    "alpha": float(ALPHA),
    "auc": {"probe": float(auc(yt, probe_t)), "pmax": float(auc(yt, pmax_t)),
            "entropy": float(auc(yt, 1 - H_t))},
    "risk_coverage": rows,
    "coverage_at_accuracy": reverse,
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print()
print("结果已保存:", sys.argv[2])
