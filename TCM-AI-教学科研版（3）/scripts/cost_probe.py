# -*- coding: utf-8 -*-
"""实测单题 token 消耗：真实调用一次，读取 usage。"""
import io, json, os, sys, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = sys.argv[1]
sys.path.insert(0, PROJ); os.chdir(PROJ)
import quiz

txt = open("server.py", encoding="utf-8").read()
seg = txt[txt.find("deepseek"):txt.find("deepseek") + 400]
a = seg.find("sk-"); b = seg.find(chr(34), a)
KEY = seg[a:b]

# 构造一个和实际出题完全一致的 prompt
task = quiz.next_task(seed=99)
if task is None:
    task = quiz.next_task()
typ = task["typical"][:200]
prompt = (
    "你是中医诊断学教师，为《中医内科学》课程出题。\n"
    "【病例】" + task["text"][:700] + "\n\n"
    "【本题标准答案】" + task["syndrome"] + "\n"
    "【该证型权威典型表现】" + typ + "\n\n"
    "【可选干扰项池（只能从这里选，不得自造证型名）】" + "、".join(task["candidates"]) + "\n\n"
    "要求：\n"
    "1) 题干为精简病例（200字内，保留主诉与舌脉）；\n"
    "2) 四个选项 = 标准答案 + 干扰项池中的 3 个相近证型，顺序任意（服务端会重新打乱），选项要有迷惑性；\n"
    "3) 解析必须引用【典型表现】作为判据，并逐条说明三个干扰项为何不符；\n"
    '4) 只输出 JSON：{"stem":"","options":["","","",""],"answer":"' + task["syndrome"] + '",'
    '"explain":"","key_points":[""],"distractor_reasons":[{"syndrome":"","why":""}]}'
)
print("=== 单题请求特征 ===")
print("  prompt 字符数 : %d" % len(prompt))
print("  system 字符数 : %d" % len("你是中医诊断学命题教师，只输出 JSON。"))
print("  病例字符数     : %d" % len(task["text"][:700]))

body = json.dumps({"model": "deepseek-chat", "messages": [
    {"role": "system", "content": "你是中医诊断学命题教师，只输出 JSON。"},
    {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 2000}).encode()
req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
    headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=150) as r:
    d = json.loads(r.read().decode())
u = d.get("usage", {})
print()
print("=== 实测 usage ===")
for k, v in u.items():
    if not isinstance(v, dict):
        print("  %-22s %s" % (k, v))
print()
print("  模型:", d.get("model"))
print("  回答长度: %d 字" % len(d["choices"][0]["message"]["content"]))
json.dump({"usage": u, "prompt_chars": len(prompt), "model": d.get("model")},
          open("docs/_cost_probe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print()
print("已写出 docs/_cost_probe.json")
