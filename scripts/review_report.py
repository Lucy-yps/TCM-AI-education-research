# -*- coding: utf-8 -*-
"""汇总人工抽检结果，生成抽检准确率报告。

用法：把填写后的 Excel 另存为 CSV（或直接在 JSON 里填），然后：
    python scripts/review_report.py <项目目录> [填写结果.csv]
若未提供 CSV，则读取 docs/*_review_sample.json 中的「人工判定」列。
"""
import csv, io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = sys.argv[1]
CSV = sys.argv[2] if len(sys.argv) > 2 else None
os.chdir(PROJ)

def rate(ok, total):
    return (ok / total * 100) if total else 0.0

def summarize(name, rows, ok_key, label):
    """rows: [{判定, 备注}]  ok_key: 判定为「正确」视为通过"""
    n = len(rows)
    if n == 0:
        print("  [%s] 无数据" % name)
        return None
    ok = sum(1 for r in rows if r.get(ok_key) == '正确')
    doubt = sum(1 for r in rows if r.get(ok_key) == '存疑')
    bad = sum(1 for r in rows if r.get(ok_key) == '错误')
    print('  [%s] %s' % (name, label))
    print('      抽检 %d 题：正确 %d（%.1f%%）｜存疑 %d（%.1f%%）｜错误 %d（%.1f%%）' % (
          n, ok, rate(ok, n), doubt, rate(doubt, n), bad, rate(bad, n)))
    return {'n': n, 'ok': ok, 'doubt': doubt, 'bad': bad,
            'accuracy': round(ok / n, 4), 'with_doubt': round((ok + doubt) / n, 4)}

# ---------- 读取填写结果 ----------
quiz_rows, formula_rows = [], []
if CSV and os.path.exists(CSV):
    # 约定：CSV 两个段落，或两个文件 xxx_1.csv / xxx_2.csv
    print("读取 CSV:", CSV)
    with open(CSV, encoding='utf-8-sig') as f:
        rd = list(csv.reader(f))
    print("  CSV 行数:", len(rd), "（请按模板结构填写；如需精确解析可改用 JSON 方式）")
else:
    q = json.load(open("docs/quiz_review_sample.json", encoding="utf-8"))
    f_ = json.load(open("docs/formula_review_sample.json", encoding="utf-8"))
    quiz_rows = [{"判定": x.get("人工判定", ""), "备注": x.get("备注", ""), "证型": x.get("证型")} for x in q]
    formula_rows = [{"判定": x.get("人工判定", ""), "备注": x.get("备注", ""), "证型": x.get("证型")} for x in f_]

print("=" * 72)
print("中医专业人工抽检 · 结果汇总")
print("=" * 72)
print()
r1 = summarize('辨证题目', quiz_rows, '判定', '答案+解析+干扰项的综合判定')
r2 = summarize('方剂治法', formula_rows, '判定', '治法+主方的综合判定')

if r1 or r2:
    allr = quiz_rows + formula_rows
    rt = summarize('合计', allr, '判定', '全部抽检题目')
    print()
    print("=" * 72)
    print("结论")
    print("=" * 72)
    if rt:
        print("  抽检 %d 题，判定「正确」%d 题，准确率 %.1f%%" % (rt["n"], rt["ok"], rt["accuracy"] * 100))
        print("  含「存疑」的可用率 %.1f%%（存疑=基本正确但描述不够典型）" % (rt["with_doubt"] * 100))
    # 错误明细
    bads = [r for r in allr if r.get('判定') == '错误']
    if bads:
        print()
        print("  判定为错误的题目：")
        for b in bads:
            print("    · %s  %s" % (b.get("证型"), ("备注：" + b["备注"]) if b.get("备注") else ""))
    # 输出报告
    out = []
    out.append('# 人工抽检结果报告')
    out.append('')
    out.append("> 由中医专业审核者抽检填写，脚本 scripts/review_report.py 汇总生成。")
    out.append('')
    out.append("## 一、抽检结果")
    out.append('')
    out.append("| 类别 | 抽检题数 | 正确 | 存疑 | 错误 | 准确率 |")
    out.append("|---|---|---|---|---|---|")
    for nm, r in (('辨证题目', r1), ('方剂治法', r2), ('合计', rt)):
        if not r: continue
        out.append("| %s | %d | %d | %d | %d | **%.1f%%** |" % (nm, r["n"], r["ok"], r["doubt"], r["bad"], r["accuracy"] * 100))
    out.append('')
    out.append("## 二、结论")
    out.append('')
    if rt:
        out.append("抽检 %d 题中，判定「正确」%d 题，**准确率 %.1f%%**；" % (rt["n"], rt["ok"], rt["accuracy"] * 100))
        out.append("计入「存疑」（基本正确但描述欠典型）后，可用率 **%.1f%%**。" % (rt["with_doubt"] * 100))
    out.append('')
    out.append("## 三、发现的错误")
    out.append('')
    if bads:
        for b in bads:
            out.append("- **%s**：%s" % (b.get("证型"), b.get("备注") or "（未填备注）"))
    else:
        out.append("本次抽检未发现判定为「错误」的题目。")
    out.append('')
    out.append("## 四、说明")
    out.append('')
    out.append("抽检为**随机抽样**，结论仅代表抽检范围内的表现；"
               "不能外推为全部题目的准确率。建议与自动化校验结果（《出题质量验证报告.md》）配合阅读。")
    out.append('')
    open("docs/人工抽检结果报告.md", "w", encoding="utf-8").write("\n".join(out))
    print()
    print("已生成 docs/人工抽检结果报告.md")
else:
    print("尚未填写任何判定。请先填写 docs/人工抽检表_请中医专业审核.xlsx 的「人工判定」列。")