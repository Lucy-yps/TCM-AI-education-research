# -*- coding: utf-8 -*-
"""生成混淆矩阵与逐条预测（683 条统一口径）。"""
import io, json, os, sys
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = sys.argv[1]
sys.path.insert(0, PROJ); os.chdir(PROJ)
import retrieval

LABEL_MAP = {"气滞血瘀证":"气滞血瘀","肝胃不和证":"肝胃不和","肝阳上亢证":"肝阳上亢",
"肝郁气滞证":"肝郁气滞","心血虚证":"心血虚","气血两虚证":"气血两虚","痰湿内阻证":"痰湿内阻",
"肝郁脾虚证":"肝郁脾虚","肝胆湿热证":"肝胆湿热","肾气虚证":"肾气虚","大肠湿热证":"大肠湿热",
"中气下陷证":"中气下陷","脾胃气虚证":"脾胃气虚","阳虚寒凝证":"阳虚寒凝","肺阴虚证":"肺阴虚","肾阳虚证":"肾阳虚"}
def ft(r): return "。".join(x for x in [r.get("chief_complaint",""), r.get("description",""), r.get("detection","")] if x)

test = []
for line in open("external_data/tcm_sd_test.json", encoding="utf-8"):
    if not line.strip(): continue
    r = json.loads(line)
    lab = r.get("norm_syndrome") or r.get("syndrome")
    if lab in LABEL_MAP: test.append(r)
tr = set()
for p in ("external_data/train_dev/train.json","external_data/train_dev/dev.json"):
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if line.strip(): tr.add(ft(json.loads(line)))
clean = [r for r in test if ft(r) not in tr]
print("683 口径测试集:", len(clean))

preds = []
for i, r in enumerate(clean):
    _, _, top = retrieval.match_clf(ft(r) or "", top_k=3)
    p1 = top[0]["syndrome"] if top else ""
    gold = LABEL_MAP[r.get("norm_syndrome") or r.get("syndrome")]
    preds.append({"i": i, "gold": gold, "pred": p1, "correct": p1 == gold,
                  "prob": round(float(top[0]["score"])/100.0, 4) if top else 0.0,
                  "top3": [x["syndrome"] for x in top]})
    if (i+1) % 200 == 0: print("  %d/%d" % (i+1, len(clean)), flush=True)

labels = sorted(set([p["gold"] for p in preds]) | set([p["pred"] for p in preds if p["pred"]]))
cm = {g: Counter() for g in labels}
for p in preds: cm[p["gold"]][p["pred"]] += 1
per = {}
for g in labels:
    tp = cm[g].get(g, 0)
    fn = sum(v for k, v in cm[g].items() if k != g)
    fp = sum(cm[o].get(g, 0) for o in labels if o != g)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    per[g] = {"tp": tp, "fp": fp, "fn": fn, "support": tp+fn,
              "precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4)}
errs = Counter()
for p in preds:
    if not p["correct"] and p["pred"]: errs[(p["gold"], p["pred"])] += 1
top_err = [{"gold": g, "pred": r, "count": c} for (g, r), c in errs.most_common(20)]

n = len(preds); correct = sum(1 for p in preds if p["correct"])
print()
print("Top-1: %d/%d = %.1f%%  Macro-F1: %.1f%%" % (correct, n, correct/n*100,
      sum(v["f1"] for v in per.values())/len(per)*100))
print()
print("=== 最易混淆证型对 ===")
for e in top_err[:10]:
    print("  %-10s → %-10s %d 次" % (e["gold"], e["pred"], e["count"]))

json.dump({"meta": {"n_cases": n, "dedup_rule": "train+dev 拼接文本比对，694→%d" % n,
                    "top1": round(correct/n,4), "macro_f1": round(sum(v["f1"] for v in per.values())/len(per),4),
                    "labels": labels,
                    "source": "TCM-SD test.json 16类对齐去泄漏子集；分类器 tcm_clf.joblib"},
           "confusion_matrix": {g: dict(cm[g]) for g in labels},
           "per_class": per, "top_errors": top_err},
          open("测试案例/eval_confusion.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump({"meta": {"n_cases": n, "labels": labels}, "predictions": preds},
          open("测试案例/eval_predictions.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print()
print("已写出 eval_confusion.json 与 eval_predictions.json")
