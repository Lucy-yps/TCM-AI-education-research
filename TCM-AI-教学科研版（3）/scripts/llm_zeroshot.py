# -*- coding: utf-8 -*-
"""LLM zero-shot baseline: 分层抽 N 条真实病历，让 DeepSeek 从 16 个证型里选一个。

只跑一次，结果存 external_data/llm_zeroshot.json，避免重复调 API。
"""
import json, os, time, random, re
import urllib.request

random.seed(42)



LABEL_MAP = {
    "气滞血瘀证": "气滞血瘀", "肝胃不和证": "肝胃不和", "肝阳上亢证": "肝阳上亢",
    "肝郁气滞证": "肝郁气滞", "心血虚证": "心血虚", "气血两虚证": "气血两虚",
    "痰湿内阻证": "痰湿内阻", "肝郁脾虚证": "肝郁脾虚", "肝胆湿热证": "肝胆湿热",
    "肾气虚证": "肾气虚", "大肠湿热证": "大肠湿热", "中气下陷证": "中气下陷",
    "脾胃气虚证": "脾胃气虚", "阳虚寒凝证": "阳虚寒凝", "肺阴虚证": "肺阴虚",
    "肾阳虚证": "肾阳虚",
}
ALL_LABELS = list(LABEL_MAP.values())

# DeepSeek 配置（和 server.py 里一致）
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_text(r):
    return "。".join(x for x in [r.get("chief_complaint", ""), r.get("description", ""), r.get("detection", "")] if x)


def call_llm(text: str) -> str:
    prompt = (
        "你是中医辨证专家。请根据以下患者描述，从下面16个证型中选出最符合的一个。"
        "只返回证型名（4个字以内），不要解释，不要加标点。\n\n"
        "证型列表：" + "、".join(ALL_LABELS) + "\n\n"
        f"患者描述：{text}\n\n只返回证型名："
    )
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是中医辨证专家，只返回证型名。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def parse_label(raw: str) -> str:
    """从 LLM 回答里提取证型名。"""
    raw = raw.strip().strip("。.\"'：: ")
    # 精确匹配
    for lab in ALL_LABELS:
        if lab in raw:
            return lab
    return "UNKNOWN"


def main():
    dev = load_jsonl(os.path.join(BASE, "external_data/train_dev/dev.json"))
    aligned = []
    for r in dev:
        lab = r.get("norm_syndrome") or r.get("syndrome")
        if lab in LABEL_MAP:
            aligned.append({"text": make_text(r), "label": LABEL_MAP[lab]})

    # 分层抽样：每个标签至少 2 条，最多 6 条
    by_label = {}
    for item in aligned:
        by_label.setdefault(item["label"], []).append(item)

    samples = []
    for lab, items in by_label.items():
        n = min(6, max(2, len(items) // 30))
        samples.extend(random.sample(items, n))
    random.shuffle(samples)
    print(f"抽样 {len(samples)} 条", flush=True)

    out_path = os.path.join(BASE, "external_data/llm_zeroshot.json")
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"已有 {len(results)} 条结果，续跑", flush=True)
    else:
        results = []

    done_keys = {(r["text"], r["label"]) for r in results}
    for i, item in enumerate(samples):
        if (item["text"], item["label"]) in done_keys:
            continue
        try:
            raw = call_llm(item["text"])
            pred = parse_label(raw)
        except Exception as e:
            raw = f"ERR:{e}"
            pred = "UNKNOWN"
        results.append({"text": item["text"], "gold": item["label"], "raw": raw, "pred": pred})
        print(f"[{i+1}/{len(samples)}] gold={item['label']:6s} pred={pred:6s} raw={raw[:40]}", flush=True)
        # 每 5 条存一次
        if len(results) % 5 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        time.sleep(0.3)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计
    correct = sum(1 for r in results if r["pred"] == r["gold"])
    total = len(results)
    print(f"\n=== LLM zero-shot (DeepSeek) ===")
    print(f"样本数: {total}")
    print(f"Top-1 准确率: {correct/total*100:.1f}%")
    # 按标签
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["gold"]][1] += 1
        if r["pred"] == r["gold"]:
            per[r["gold"]][0] += 1
    print("\n按标签：")
    for lab in sorted(per.keys(), key=lambda x: -per[x][1]):
        c, t = per[lab]
        print(f"  {lab:6s}: {c}/{t} = {c/t*100:.0f}%")


if __name__ == "__main__":
    main()
