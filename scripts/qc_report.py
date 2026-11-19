# -*- coding: utf-8 -*-
"""从现有题库生成质检报告（不重新出题），并导出人工抽检样本。"""
import io, json, os, random, sys
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ, SAMPLE_N = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, PROJ); os.chdir(PROJ)
import quiz

bank = quiz.bank()
items = list(bank.values())
n = len(items)
if n == 0:
    print("题库为空"); sys.exit(0)

print("=" * 76)
print("辨证训练题库 · 质量质检报告")
print("=" * 76)
print("  题库规模: %d 题" % n)
st = quiz.stats()
print("  出题范围: %s（核心 %d 证型 / 素材 %d 条）" % (st["discipline"], st["core"], st["case_refs"]))

core_keys = ["answer_is_target", "four_options", "all_options_unique", "options_in_kb", "distractors_from_pool", "has_explain"]
extra_keys = ["explain_mentions_typical", "has_distractor_reasons"]

print()
print("[1] 质量校验通过率")
passed = sum(1 for q in items if (q.get("checks") or {}).get("passed"))
print("      核心校验全通过: %d/%d = %.1f%%" % (passed, n, passed / n * 100))
for k in core_keys + extra_keys:
    ok = sum(1 for q in items if (q.get("checks") or {}).get(k))
    print("        [%s] %-26s %4d/%d = %5.1f%%" % ("核心" if k in core_keys else "加分", k, ok, n, ok / n * 100))

print()
print("[2] 正确选项位置分布（防止靠位置猜答案）")
pos = Counter()
for q in items:
    i = q.get("answer_index", -1)
    pos[chr(65 + i) if 0 <= i < 4 else "?"] += 1
mx = 0.0
for k in ["A", "B", "C", "D"]:
    v = pos.get(k, 0)
    pct = v / n * 100
    mx = max(mx, pct)
    print("      %s: %4d 题 (%5.1f%%)  %s" % (k, v, pct, "#" * int(pct / 2)))
print("      最高占比 %.1f%%（理想 25%%，均匀性: %s）" % (mx, "优秀" if mx <= 30 else ("良好" if mx <= 35 else "可接受" if mx <= 45 else "偏斜")))

print()
print("[3] 证型覆盖")
bysyn = Counter(q["syndrome"] for q in items)
core_pool = quiz.data()["pools"]["core"]
print("      已覆盖: %d / %d 核心池 = %.1f%%" % (len(bysyn), len(core_pool), len(bysyn) / len(core_pool) * 100))
print("      单证型题量: 最多 %d 题, 最少 %d 题（越接近越均衡）" % (max(bysyn.values()), min(bysyn.values())))
gaps = [s for s in core_pool if s not in bysyn]
if gaps:
    print("      尚未覆盖: %s" % "、".join(gaps[:10]))

print()
print("[4] 干扰项多样性")
dist = Counter()
for q in items:
    for o in q.get("options", []):
        if o != q["syndrome"]:
            dist[o] += 1
print("      不同干扰项证型数: %d" % len(dist))
print("      最高单项出现率: %.1f%%（越低越多样）" % (dist.most_common(1)[0][1] / n * 100))
print("      使用最多: " + "、".join("%s(%d)" % (k, v) for k, v in dist.most_common(5)))

print()
print("[5] 题干与来源")
lens = [len(str(q.get("stem", ""))) for q in items]
print("      题干长度: 平均 %d 字（%d~%d）" % (sum(lens) / n, min(lens), max(lens)))
print("      涉及疾病: %d 个" % len(set(q.get("lcd_name", "") for q in items)))
sp = Counter(str(q.get("src", "")).split("#")[0] for q in items)
print("      来源划分: " + "、".join("%s %d 题" % (k, v) for k, v in sp.items()))

print()
print("[6] 重复检查")
print("      同一病历重复出题: %d 组" % sum(1 for v in Counter(q.get("src") for q in items).values() if v > 1))
print("      题干重复: %d 组" % sum(1 for v in Counter(str(q.get("stem", ""))[:60] for q in items).values() if v > 1))

print()
print("[7] 选项长度特征（防止靠长度猜答案）")
same_len = 0; ul = 0; us = 0
for q in items:
    opts = [str(o) for o in (q.get("options") or [])]
    if len(opts) < 4:
        continue
    L = [len(o) for o in opts]
    if len(set(L)) == 1:
        same_len += 1; continue
    if L.count(max(L)) == 1 and len(q["answer"]) == max(L):
        ul += 1
    if L.count(min(L)) == 1 and len(q["answer"]) == min(L):
        us += 1
print("      四选项长度相同: %d/%d = %.1f%%" % (same_len, n, same_len / n * 100))
print("      答案为唯一最长: %d/%d = %.1f%%" % (ul, n, ul / n * 100))
print("      答案为唯一最短: %d/%d = %.1f%%" % (us, n, us / n * 100))
print("      → 后两项接近 0 表示不存在可利用的长度规律")

print()
print("[8] 解析充分性")
elen = [len(str(q.get("explain", ""))) for q in items]
dlen = [len(q.get("distractor_reasons") or []) for q in items]
print("      解析长度: 平均 %d 字（%d~%d）" % (sum(elen) / n, min(elen), max(elen)))
print("      干扰项辨析条数: 平均 %.1f 条/题" % (sum(dlen) / n))

print()
print("=" * 76)
print("结论")
print("=" * 76)
print("  核心校验通过率 %.1f%% ｜ 答案位置最高占比 %.1f%% ｜ 覆盖证型 %d 个（%.1f%%）" % (
    passed / n * 100, mx, len(bysyn), len(bysyn) / len(core_pool) * 100))
verdict = "可用于教学练习" if passed / n >= 0.95 and mx <= 40 else "需进一步检查"
print("  自动质检结论: %s" % verdict)
print("  下一步: 人工抽检 %d 题，核对答案与解析的医学正确性" % SAMPLE_N)

# 导出人工抽检样本
rnd = random.Random(20261004)
sample = rnd.sample(items, min(SAMPLE_N, n))
out = []
for k, q in enumerate(sample, 1):
    out.append({
        "序号": k,
        "证型": q["syndrome"],
        "题干": q.get("stem"),
        "选项": q.get("options"),
        "标准答案": q.get("answer"),
        "解析": q.get("explain"),
        "干扰项辨析": q.get("distractor_reasons"),
        "来源": q.get("src"),
        "来源疾病": q.get("lcd_name"),
        "人工判定": "",        # 留空：正确 / 错误 / 存疑
        "备注": "",
    })
path = "docs/quiz_review_sample.json"
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print()
print("人工抽检样本已导出: %s（%d 题）" % (path, len(out)))
