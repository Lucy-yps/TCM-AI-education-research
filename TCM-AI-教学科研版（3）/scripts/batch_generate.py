# -*- coding: utf-8 -*-
"""批量出题 + 质检报告。

主线程串行管理题库（避免竞态），并发执行 LLM 调用以提速。
"""
import io, json, os, sys, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ, TARGET, WORKERS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, PROJ); os.chdir(PROJ)
import quiz

txt = open("server.py", encoding="utf-8").read()
seg = txt[txt.find("deepseek"):txt.find("deepseek") + 400]
a = seg.find("sk-"); b = seg.find(chr(34), a)
KEY = seg[a:b]

def llm(prompt):
    body = json.dumps({"model": "deepseek-chat", "messages": [
        {"role": "system", "content": "你是中医诊断学命题教师，只输出 JSON。"},
        {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 2000}).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                d = json.loads(r.read().decode())
            c = d["choices"][0]["message"]["content"]
            s, e = c.find("{"), c.rfind("}")
            return json.loads(c[s:e+1]) if s >= 0 and e > s else None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2)
    return None

def work(task):
    q = quiz.build_question(task, llm)
    return task, q

bank = quiz.bank()
print("起始题库: %d 题 | 目标: %d 题 | 并发: %d" % (len(bank), TARGET, WORKERS), flush=True)
t0 = time.time()
rounds = 0
while len(bank) < TARGET and rounds < 40:
    rounds += 1
    # 主线程串行取任务（避免竞态）
    need = min(WORKERS, TARGET - len(bank))
    tasks = []
    for _ in range(need):
        t = quiz.next_task()
        if t is None:
            break
        tasks.append(t)
    if not tasks:
        print("素材已用尽", flush=True); break
    # 并发生成
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        results = list(ex.map(work, tasks))
    # 主线程串行入库
    added = 0
    for task, q in results:
        if not q:
            continue
        key = q["src"] + "|" + task["syndrome"]
        if key in bank:
            continue
        bank[key] = q
        added += 1
    quiz.save_bank()
    done = len(bank)
    print("  轮次 %2d: +%d 题 -> 共 %d 题 (%.0fs)" % (rounds, added, done, time.time() - t0), flush=True)

print()
print("=== 生成完成 ===")
print("  题库总量: %d 题 | 耗时 %.0f 秒 | 平均 %.1f 秒/题" % (len(bank), time.time()-t0, (time.time()-t0)/max(len(bank),1)))

# ---------------- 质检报告 ----------------
print()
print("=" * 74)
print("质量质检报告")
print("=" * 74)
items = list(bank.values())
n = len(items)
if n == 0:
    print("  无题目"); sys.exit(0)

# 1 核心校验
core_keys = ["answer_is_target", "four_options", "all_options_unique", "options_in_kb", "distractors_from_pool", "has_explain"]
extra_keys = ["explain_mentions_typical", "has_distractor_reasons"]
print()
print("[1] 质量校验通过率")
passed = sum(1 for q in items if (q.get("checks") or {}).get("passed"))
print("    核心校验全通过: %d/%d = %.1f%%" % (passed, n, passed/n*100))
for k in core_keys + extra_keys:
    ok = sum(1 for q in items if (q.get("checks") or {}).get(k))
    tag = "核心" if k in core_keys else "加分"
    print("      [%s] %-26s %4d/%d = %5.1f%%" % (tag, k, ok, n, ok/n*100))

# 2 答案位置分布
print()
print("[2] 正确选项位置分布")
pos = Counter()
for q in items:
    i = q.get("answer_index", -1)
    pos[chr(65+i) if 0 <= i < 4 else "?"] += 1
for k in ["A", "B", "C", "D", "?"]:
    v = pos.get(k, 0)
    if v == 0 and k == "?":
        continue
    print("      %s: %4d 题 (%5.1f%%)  %s" % (k, v, v/n*100, "#" * int(v/n*50)))
mx = max(pos.get(k, 0) for k in "ABCD") / n * 100
print("      最高占比 %.1f%%（理想 25%%，均匀性: %s）" % (mx, "良好" if mx <= 35 else ("可接受" if mx <= 45 else "偏斜")))

# 3 证型覆盖
print()
print("[3] 证型覆盖")
bysyn = Counter(q["syndrome"] for q in items)
core_pool = quiz.data()["pools"]["core"]
covered = len(bysyn)
print("      已覆盖证型: %d / %d 核心池 = %.1f%%" % (covered, len(core_pool), covered/len(core_pool)*100))
print("      每题证型出现次数: 最多 %d 次, 最少 %d 次" % (max(bysyn.values()), min(bysyn.values())))
top = bysyn.most_common(5)
print("      出题最多: " + ", ".join("%s(%d)" % (k, v) for k, v in top))

# 4 干扰项多样性（是否反复用同几个证型）
print()
print("[4] 干扰项使用频次（检查是否总是同几个证型）")
dist = Counter()
for q in items:
    for o in q.get("options", []):
        if o != q["syndrome"]:
            dist[o] += 1
print("      不同干扰项证型数: %d" % len(dist))
print("      使用最多的干扰项: " + ", ".join("%s(%d)" % (k, v) for k, v in dist.most_common(5)))
if dist:
    top_share = dist.most_common(1)[0][1] / n * 100
    print("      最高单项出现率: %.1f%%（越低越多样）" % top_share)

# 5 题干长度与来源
print()
print("[5] 题干与来源")
lens = [len(str(q.get("stem", ""))) for q in items]
print("      题干长度: 平均 %d 字, 最短 %d, 最长 %d" % (sum(lens)/n, min(lens), max(lens)))
lc = Counter(q.get("lcd_name", "") for q in items)
print("      涉及疾病数: %d" % len(lc))
sp = Counter(str(q.get("src", "")).split("#")[0] for q in items)
print("      来源划分: " + ", ".join("%s %d" % (k, v) for k, v in sp.items()))

# 6 重复检查
print()
print("[6] 重复检查")
srcs = Counter(q.get("src") for q in items)
dup_src = sum(1 for v in srcs.values() if v > 1)
stems = Counter(str(q.get("stem", ""))[:60] for q in items)
dup_stem = sum(1 for v in stems.values() if v > 1)
print("      同一病历重复出题: %d 组" % dup_src)
print("      题干前 60 字重复: %d 组" % dup_stem)

# 7 选项长度特征（检查是否存在可利用的长度规律）
print()
print("[7] 选项长度特征（防止靠长度猜答案）")
same_len = 0
uniq_longest = 0
uniq_shortest = 0
for q in items:
    opts = [str(o) for o in (q.get("options") or [])]
    if len(opts) < 4:
        continue
    lens = [len(o) for o in opts]
    if len(set(lens)) == 1:
        same_len += 1
        continue
    mx, mn = max(lens), min(lens)
    if lens.count(mx) == 1 and len(q["answer"]) == mx:
        uniq_longest += 1
    if lens.count(mn) == 1 and len(q["answer"]) == mn:
        uniq_shortest += 1
print("      四选项长度完全相同: %d/%d = %.1f%%（无长度线索）" % (same_len, n, same_len/n*100))
print("      答案为唯一最长选项: %d/%d = %.1f%%" % (uniq_longest, n, uniq_longest/n*100))
print("      答案为唯一最短选项: %d/%d = %.1f%%" % (uniq_shortest, n, uniq_shortest/n*100))
print("      → 若后两项接近 0，则不存在可利用的长度规律")

print()
print("=" * 74)
print("结论")
print("=" * 74)
ok = (passed/n >= 0.9) and (mx <= 45)
print("  核心校验通过率 %.1f%% | 答案位置最高占比 %.1f%% | 覆盖证型 %d 个" % (passed/n*100, mx, covered))
print("  总体评价: %s" % ("可用（建议再抽检 10~20 题人工复核解析质量）" if ok else "需检查"))
