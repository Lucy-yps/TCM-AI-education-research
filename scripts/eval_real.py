# -*- coding: utf-8 -*-
"""在 TCM-SD 真实电子病历（精确对齐到我们 16 个证型的子集，694 例）上评测四种检索方法。

数据来源：Borororo/ZY-BERT GitHub 仓库公开 test.json（CC-BY-SA），
         5486 条真实电子病历，148 个标准化证型标签。
本脚本只保留 norm_syndrome 能精确映射到我们知识库证型的子集。
"""
import json
import os
import sys
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/ 的上一级
sys.path.insert(0, BASE_DIR)   # 让脚本能找到项目根目录的 retrieval.py
import retrieval


RAW_FILE = os.path.join(BASE_DIR, "external_data", "tcm_sd_test.json")

# TCM-SD norm_syndrome -> 我们 knowledge_base.json 中的 syndrome
LABEL_MAP = {
    "气滞血瘀证": "气滞血瘀",
    "肝胃不和证": "肝胃不和",
    "肝阳上亢证": "肝阳上亢",
    "肝郁气滞证": "肝郁气滞",
    "心血虚证": "心血虚",
    "气血两虚证": "气血两虚",
    "痰湿内阻证": "痰湿内阻",
    "肝郁脾虚证": "肝郁脾虚",
    "肝胆湿热证": "肝胆湿热",
    "肾气虚证": "肾气虚",
    "大肠湿热证": "大肠湿热",
    "中气下陷证": "中气下陷",
    "脾胃气虚证": "脾胃气虚",
    "阳虚寒凝证": "阳虚寒凝",
    "肺阴虚证": "肺阴虚",
    "肾阳虚证": "肾阳虚",
}


def load_real_cases():
    cases = []
    with open(RAW_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab = r.get("norm_syndrome") or r.get("syndrome")
            if lab not in LABEL_MAP:
                continue
            # 拼输入：主诉 + 现病史 + 四诊
            text = "。".join(
                x for x in [r.get("chief_complaint", ""), r.get("description", ""), r.get("detection", "")] if x
            )
            cases.append({"text": text, "expected": LABEL_MAP[lab], "raw_label": lab})
    return cases


def evaluate(method_fn, cases):
    n = len(cases)
    top1 = top3 = 0
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    per_syndrome = defaultdict(lambda: [0, 0])  # gold -> [correct, total]
    for c in cases:
        _, _, top = method_fn(c["text"], top_k=3)
        preds = [r["syndrome"] for r in top]
        gold = c["expected"]
        per_syndrome[gold][1] += 1
        if preds and preds[0] == gold:
            top1 += 1
            tp[gold] += 1
            per_syndrome[gold][0] += 1
        else:
            fp[preds[0]] += 1 if preds else 0
            fn[gold] += 1
        if gold in preds:
            top3 += 1
    classes = set(tp) | set(fp) | set(fn)
    f1s = []
    for cl in classes:
        p = tp[cl] / (tp[cl] + fp[cl]) if (tp[cl] + fp[cl]) else 0.0
        r = tp[cl] / (tp[cl] + fn[cl]) if (tp[cl] + fn[cl]) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return top1 / n, top3 / n, macro_f1, per_syndrome


def main():
    cases = load_real_cases()
    print(f"真实病历评测集：{len(cases)} 条（精确对齐到 {len(LABEL_MAP)} 个证型）")
    label_dist = Counter(c["expected"] for c in cases)
    print("各证型样本数：")
    for k, v in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
    print("=" * 72)

    rows = []
    for name, fn in retrieval.METHODS.items():
        t1, t3, mf1, per_syn = evaluate(fn, cases)
        rows.append((name, t1, t3, mf1, per_syn))
        print(f"[{name:8s}] Top-1={t1*100:5.1f}%  Top-3={t3*100:5.1f}%  Macro-F1={mf1*100:5.1f}%")

    print("=" * 72)
    print("汇总表（真实电子病历）：")
    print(f"{'method':<10}{'Top-1':>10}{'Top-3':>10}{'Macro-F1':>12}")
    for name, t1, t3, mf1, _ in rows:
        print(f"{name:<10}{t1*100:>9.1f}%{t3*100:>9.1f}%{mf1*100:>11.1f}%")

    # 按证型看 tfidf 方法的表现（找弱项）
    best_name, _, _, _, per_syn = max(rows, key=lambda x: x[1])
    print(f"\n--- 按证型准确率（{best_name} 方法，真实病历）---")
    for syn, (correct, total) in sorted(per_syn.items(), key=lambda x: x[1][0]/max(x[1][1],1)):
        print(f"  {syn:10s}  {correct:4d}/{total:4d}  = {correct/total*100:5.1f}%")

    # 保存
    out = {
        "source": "TCM-SD test.json (Borororo/ZY-BERT, CC-BY-SA)",
        "n_cases": len(cases),
        "mapped_syndromes": list(LABEL_MAP.values()),
        "summary": [
            {"method": n, "top1": round(t1*100, 1), "top3": round(t3*100, 1), "macro_f1": round(mf1*100, 1)}
            for n, t1, t3, mf1, _ in rows
        ],
    }
    with open(os.path.join(BASE_DIR, "测试案例", "eval_real_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写 eval_real_results.json")


if __name__ == "__main__":
    main()
