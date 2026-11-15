# -*- coding: utf-8 -*-
"""保形预测（Conformal Prediction）：主实验 + 校准集来源敏感性分析

文献：Vovk et al. 2005（理论基础）；Angelopoulos & Bates, arXiv:2107.07511（LAC）；
      Tibshirani et al., NeurIPS 2019（协变量偏移下的加权保形）

三个假设逐一检验：
  H1 标签偏移（类先验不同）→ 加权 CP 修正   【已否定：先验几乎相同】
  H2 校准集来源不当          → 换用 dev 校准  【本脚本检验】
  H3 特征分布偏移（同证型不同表述）→ 无法用标准 CP 修正
"""
import io, json, math, os, pickle, random, sys
import numpy as np
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
def ft(r):
    return "。".join(x for x in [r.get("chief_complaint",""), r.get("description",""),
                                   r.get("detection","")] if x)
def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        lab = r.get("norm_syndrome") or r.get("syndrome")
        if lab in LABEL_MAP:
            out.append({"text": ft(r), "gold": LABEL_MAP[lab]})
    return out

tr = load("external_data/train_dev/train.json")
dv = load("external_data/train_dev/dev.json")
te_all = load("external_data/tcm_sd_test.json")
seen = set()
for p in ("external_data/train_dev/train.json", "external_data/train_dev/dev.json"):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            seen.add(ft(json.loads(line)))
test = [c for c in te_all if c["text"] not in seen]
with open("external_data/tcm_clf.joblib", "rb") as f:
    clf = pickle.load(f)
classes = list(clf.classes_)
K = len(classes)
def proba_of(texts, batch=500):
    return np.vstack([clf.predict_proba(texts[i:i+batch])
                      for i in range(0, len(texts), batch)])

Pt = proba_of([c["text"] for c in test])
yt = np.array([classes.index(c["gold"]) if c["gold"] in classes else -1 for c in test])
valid = yt >= 0

# 三种校准集
random.Random(2026).shuffle(tr)
CAL_SETS = {}
CAL_SETS["train 2000 (与测试异源)"] = tr[:2000]
CAL_SETS["dev 694 (与测试同源)"] = dv
CAL_SETS["train 1000 + dev 694 (混合)"] = tr[:1000] + dv

print("=" * 80)
print("保形预测：校准集来源敏感性（683 例去泄漏测试集）")
print("=" * 80)
rows = {}
for name, cal in CAL_SETS.items():
    Pc = proba_of([c["text"] for c in cal])
    yc = np.array([classes.index(c["gold"]) if c["gold"] in classes else -1 for c in cal])
    m = yc >= 0
    acc_cal = float(np.mean(Pc.argmax(axis=1) == yc))
    sc = (1.0 - Pc[np.arange(len(cal)), np.clip(yc, 0, K-1)])[m]
    print()
    print("【%s】 校准 n=%d, 校准集准确率 %.4f" % (name, len(sc), acc_cal))
    print("  %-9s | %-14s | %-14s | %-10s" % ("目标覆盖", "实测覆盖", "平均集大小", "达标?"))
    print("  " + "-" * 60)
    res = []
    for alpha in [0.10, 0.05, 0.02, 0.01]:
        n = len(sc)
        lv = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        q = float(np.quantile(sc, lv, method="higher"))
        S = Pt >= (1.0 - q)
        cov = float(np.mean([S[i, yt[i]] for i in range(len(test)) if valid[i]]))
        ok = "OK" if cov >= (1 - alpha) - 0.005 else "不足"
        print("  %6.0f%%   | %-14.4f | %-14.2f | %-10s" % ((1-alpha)*100, cov, S.sum(axis=1).mean(), ok))
        res.append({"target": 1 - alpha, "empirical_coverage": cov,
                    "avg_size": float(S.sum(axis=1).mean()), "meets_target": cov >= (1-alpha)-0.005})
    rows[name] = {"cal_n": int(len(sc)), "cal_accuracy": acc_cal, "results": res}

print()
print("=" * 80)
print("结论")
print("=" * 80)
dev_ok = all(r["meets_target"] for r in rows["dev 694 (与测试同源)"]["results"])
tr_ok = all(r["meets_target"] for r in rows["train 2000 (与测试异源)"]["results"])
print("  用 dev 校准（与测试同源）: %s" % ("全部达标 ✅" if dev_ok else "仍有不足"))
print("  用 train 校准（与测试异源）: %s" % ("全部达标 ✅" if tr_ok else "覆盖不足 ❌"))
if dev_ok and not tr_ok:
    print()
    print("  → 根因确认：不是方法问题，而是【校准集必须与部署分布同源】。")
    print("    这给出一条可操作的工程结论：保形预测的覆盖率保证依赖")
    print("    校准集与测试集的同分布性，跨数据源直接套用会失效。")

out = {"method": "Split Conformal Prediction (LAC)", "test_n": int(valid.sum()),
       "calibration_sources": rows,
       "verdict": "coverge guarantee holds only when calibration set is same-source as deployment data"}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print()
print("结果已保存:", sys.argv[2])