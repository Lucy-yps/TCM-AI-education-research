# -*- coding: utf-8 -*-
"""离线评测：在 eval_cases.json 上跑四种检索方法，输出 Top-1 / Top-3 / Macro-F1 对比表。

用法：
    python evaluate.py
"""
import json
import os
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/ 的上一级
sys.path.insert(0, BASE_DIR)   # 让脚本能找到项目根目录的 retrieval.py
import retrieval


DATA_DIR = os.path.join(BASE_DIR, "测试案例")
EVAL_FILE = os.path.join(DATA_DIR, "eval_cases.json")


def load_cases():
    with open(EVAL_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def evaluate(method_name, method_fn, cases):
    """返回 (top1_acc, top3_acc, per_class_f1_macro, detail_rows)"""
    n = len(cases)
    top1 = 0
    top3 = 0
    # 用混淆思路算 per-class precision/recall
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    detail = []

    for case in cases:
        text = case["text"]
        gold = case["expected"]
        _, _, top = method_fn(text, top_k=3)
        pred_list = [r["syndrome"] for r in top]
        pred1 = pred_list[0] if pred_list else "(none)"

        if pred1 == gold:
            top1 += 1
            tp[gold] += 1
        else:
            fp[pred1] += 1
            fn[gold] += 1

        if gold in pred_list:
            top3 += 1

        detail.append({
            "text": text, "gold": gold,
            "pred1": pred1, "pred_list": pred_list,
            "hit_top1": pred1 == gold,
            "hit_top3": gold in pred_list,
        })

    # macro F1：对所有出现过的类求平均
    classes = set(tp) | set(fp) | set(fn)
    f1s = []
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return top1 / n, top3 / n, macro_f1, detail


def main():
    cases = load_cases()
    print(f"评测集：{len(cases)} 条教材标准病例")
    print(f"知识库：{len(retrieval.KNOWLEDGE_BASE)} 个证型")
    print("=" * 72)

    rows = []
    for name, fn in retrieval.METHODS.items():
        t1, t3, mf1, detail = evaluate(name, fn, cases)
        rows.append((name, t1, t3, mf1, detail))
        wrong = [d for d in detail if not d["hit_top1"]]
        print(f"[{name:8s}] Top-1={t1*100:5.1f}%  Top-3={t3*100:5.1f}%  Macro-F1={mf1*100:5.1f}%  错例={len(wrong)}")

    print("=" * 72)
    print("汇总表（可直接贴 PPT / README）：")
    print(f"{'method':<10}{'Top-1':>10}{'Top-3':>10}{'Macro-F1':>12}")
    for name, t1, t3, mf1, _ in rows:
        print(f"{name:<10}{t1*100:>9.1f}%{t3*100:>9.1f}%{mf1*100:>11.1f}%")

    # 打印 hybrid 方法的错例，供分析
    print("\n--- hybrid 方法 Top-1 错例（前 10 条）---")
    hybrid_detail = next(d for n, _, _, _, d in rows if n == "hybrid")
    wrong = [d for d in hybrid_detail if not d["hit_top1"]]
    for d in wrong[:10]:
        print(f"  金标={d['gold']:<8s} 预测={d['pred1']:<8s}  | 输入：{d['text'][:50]}")

    # 保存详细结果
    out = {
        "n_cases": len(cases),
        "n_syndromes": len(retrieval.KNOWLEDGE_BASE),
        "summary": [
            {"method": n, "top1": round(t1*100, 1), "top3": round(t3*100, 1), "macro_f1": round(mf1*100, 1)}
            for n, t1, t3, mf1, _ in rows
        ],
        "per_case": {n: d for n, _, _, _, d in rows},
    }
    out_path = os.path.join(DATA_DIR, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已写入：{out_path}")


if __name__ == "__main__":
    main()
