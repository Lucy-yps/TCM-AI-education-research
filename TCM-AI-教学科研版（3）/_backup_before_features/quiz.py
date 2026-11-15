# -*- coding: utf-8 -*-
"""辨证训练出题引擎（独立模块，不依赖 retrieval.py 的修改）。

设计原则：生成与判定分离。
  病历 + 证型知识库  ->  决定标准答案（不交给大模型自由发挥）
  大模型             ->  只负责把病例写成题干、并生成解析文字
  白名单 + 候选池    ->  约束选项，杜绝跨体系命名
"""
import json, os, random, threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "docs", "quiz_data.json")
CASES_PATH = os.path.join(BASE_DIR, "docs", "quiz_cases.json")
BANK_PATH = os.path.join(BASE_DIR, "external_data", "quiz_bank.json")

_LAST_ERROR = None

_SPLIT_FILES = {
    "train": os.path.join(BASE_DIR, "external_data", "train_dev", "train.json"),
    "dev":   os.path.join(BASE_DIR, "external_data", "train_dev", "dev.json"),
    "test":  os.path.join(BASE_DIR, "external_data", "tcm_sd_test.json"),
}

# 精简素材包（只含出题池用到的病历，gzip 压缩；68.8 MB -> 5.4 MB）
# 若存在则优先使用，使出题功能不依赖完整的 TCM-SD 原始数据集。
_PACK_DIR = os.path.join(BASE_DIR, "external_data", "quiz_src")
_PACK_CACHE = {}

_LOCK = threading.Lock()
_DATA = None
_CASES = None
_BANK = None


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def data():
    global _DATA
    if _DATA is None:
        _DATA = _load_json(DATA_PATH, {"scope": {}, "pools": {"core": [], "extended": []}, "syndromes": {}})
    return _DATA


def cases():
    global _CASES
    if _CASES is None:
        _CASES = _load_json(CASES_PATH, {})
    return _CASES


def bank():
    """返回题库。每次从磁盘合并最新内容，避免多进程（服务 / 批量脚本）互相覆盖。"""
    global _BANK
    disk = _load_json(BANK_PATH, {})
    if _BANK is None:
        _BANK = disk
    elif disk:
        for k, v in disk.items():
            _BANK.setdefault(k, v)
    return _BANK


def save_bank():
    """写盘前先把磁盘上的新题目合并进来，防止覆盖其它进程的成果。"""
    if _BANK is None:
        return
    disk = _load_json(BANK_PATH, {})
    for k, v in disk.items():
        _BANK.setdefault(k, v)
    os.makedirs(os.path.dirname(BANK_PATH), exist_ok=True)
    with open(BANK_PATH, "w", encoding="utf-8") as f:
        json.dump(_BANK, f, ensure_ascii=False)


def stats():
    d = data()
    b = bank()
    return {
        "discipline": d.get("scope", {}).get("discipline", ""),
        "scope_rule": d.get("scope", {}).get("rule", ""),
        "total_cases": d.get("scope", {}).get("total_cases", 0),
        "internal_cases": d.get("scope", {}).get("internal_cases", 0),
        "syndrome_total": d.get("scope", {}).get("syndrome_total", 0),
        "core": len(d.get("pools", {}).get("core", [])),
        "extended": len(d.get("pools", {}).get("extended", [])),
        "quota": d.get("pools", {}).get("total_quota", 0),
        "case_refs": sum(len(v) for v in cases().values()),
        "bank_size": len(b),
    }


def _render(record):
    return "。".join(x for x in [record.get("chief_complaint", ""), record.get("description", ""),
                                 record.get("detection", "")] if x)


def _load_pack(split):
    """加载精简素材包（gzip），返回 {行号: 记录}。"""
    if split in _PACK_CACHE:
        return _PACK_CACHE[split]
    mapping = {}
    path = os.path.join(_PACK_DIR, split + ".json.gz")
    if os.path.exists(path):
        try:
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for lineno, rec in json.load(f):
                    mapping[lineno] = rec
        except (OSError, ValueError):
            mapping = {}
    _PACK_CACHE[split] = mapping
    return mapping


def read_case(ref):
    """按引用回读病历原文。

    优先使用 external_data/quiz_src/*.json.gz 精简素材包（只含出题池用到的病历）；
    若不存在则回退到完整的 TCM-SD 原始数据集。
    """
    split = ref.get("f")
    target = ref.get("i", -1)

    # 1) 精简素材包
    pack = _load_pack(split)
    if target in pack:
        return _render(pack[target])

    # 2) 回退：完整数据集
    path = _SPLIT_FILES.get(split)
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == target:
                if not line.strip():
                    return ""
                return _render(json.loads(line))
    return ""


def pick_distractors(target, k=9, seed=None):
    """候选干扰项：同为内科证型，与目标共享字眼者优先（提高迷惑性）。"""
    d = data()
    pool = list(d.get("pools", {}).get("core", [])) + list(d.get("pools", {}).get("extended", []))
    chars = set(target)
    scored = [(sum(1 for c in chars if c in s), s) for s in pool if s != target]
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:max(k * 2, k)]]
    rnd = random.Random(seed)
    rnd.shuffle(top)
    return top[:k]


def list_syndromes():
    d = data()
    core = set(d.get("pools", {}).get("core", []))
    out = []
    for name, info in d.get("syndromes", {}).items():
        b = bank()
        done = sum(1 for q in b.values() if q.get("syndrome") == name)
        out.append({
            "syndrome": name,
            "tier": "core" if name in core else "extended",
            "n_cases": info.get("n_cases", 0),
            "quota": info.get("quota", 0),
            "generated": done,
            "typical": info.get("typical", "")[:120],
            "definition": info.get("definition", "")[:120],
        })
    out.sort(key=lambda x: (-x["tier"].__eq__("core"), -x["n_cases"]))
    return out


def next_task(syndrome=None, seed=None):
    """挑一条还没出过题的病历，返回出题任务（含候选干扰项）。"""
    d = data()
    cs = cases()
    b = bank()
    used = set(q.get("src") for q in b.values())
    names = [syndrome] if syndrome else list(d.get("pools", {}).get("core", []))
    rnd = random.Random(seed)
    rnd.shuffle(names)
    for name in names:
        refs = cs.get(name, [])
        fresh = [r for r in refs if "%s#%d" % (r.get("f"), r.get("i", -1)) not in used]
        if not fresh:
            continue
        ref = rnd.choice(fresh)
        text = read_case(ref)
        if len(text) < 60:
            continue
        return {
            "syndrome": name,
            "ref": ref,
            "src": "%s#%d" % (ref.get("f"), ref.get("i", -1)),
            "text": text,
            "typical": d.get("syndromes", {}).get(name, {}).get("typical", ""),
            "definition": d.get("syndromes", {}).get(name, {}).get("definition", ""),
            "candidates": pick_distractors(name, seed=seed),
        }
    return None


def validate(question, task):
    """质量校验：答案、白名单、干扰项来源、解析是否引用典型表现。"""
    d = data()
    kb = set(d.get("syndromes", {}).keys())
    cand = set(task["candidates"])
    opts = [str(o).strip() for o in question.get("options", [])]
    reasons = question.get("distractor_reasons") or []
    typ = task.get("typical", "")
    # 核心校验（决定题目是否可用）
    core = {
        "answer_is_target": str(question.get("answer", "")).strip() == task["syndrome"],
        "four_options": len(opts) == 4,
        "all_options_unique": len(set(opts)) == len(opts),
        "options_in_kb": all(o in kb for o in opts),
        "distractors_from_pool": all(o in cand for o in opts if o != task["syndrome"]),
        "has_explain": len(str(question.get("explain", ""))) >= 30,
    }
    # 质量加分项（记录但不作为通过门槛）
    extra = {
        "explain_mentions_typical": (typ[:6] in str(question.get("explain", ""))) if typ else True,
        "has_distractor_reasons": len(reasons) >= 3,
    }
    checks = dict(core)
    checks.update(extra)
    checks["passed"] = all(core.values())
    return checks


def build_question(task, llm_call):
    """调用大模型生成题目正文；答案由 task 决定，模型不得更改。"""
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
    q = llm_call(prompt)
    if not q:
        return None
    q["syndrome"] = task["syndrome"]
    q["src"] = task["src"]
    q["lcd_name"] = task["ref"].get("d", "")

    # ---- 正确选项位置随机化 ----
    # 提示词要求"标准答案 + 三个干扰项"，模型会习惯性把答案放在首位；
    # 因此不能依赖模型自觉，必须由程序洗牌，并同步更新 answer 字段。
    opts = [str(o).strip() for o in (q.get("options") or [])]
    ans = str(q.get("answer", "")).strip()
    if len(opts) == 4 and ans in opts:
        rnd = random.Random(hash(task["src"]) & 0xFFFFFFFF)
        rnd.shuffle(opts)
        q["options"] = opts
        q["answer_index"] = opts.index(ans)
        q["answer"] = ans          # 答案文本不变，仅位置改变
    else:
        q["options"] = opts
        q["answer_index"] = opts.index(ans) if ans in opts else -1

    q["checks"] = validate(q, task)
    return q


def last_error():
    """返回最近一次 generate_one 的失败原因。"""
    return _LAST_ERROR


def generate_one(syndrome=None, llm_call=None, seed=None, save=True):
    """生成一道题并入库。失败时通过 last_error() 给出具体原因，便于排查。"""
    global _LAST_ERROR
    _LAST_ERROR = None
    if llm_call is None:
        _LAST_ERROR = "生成器未配置（llm_call 为空）"
        return None
    task = next_task(syndrome, seed)
    if task is None:
        _LAST_ERROR = ("该范围内已无可出题素材（可能全部出过题，或病历原文缺失。"
                       "请确认 external_data/quiz_src/*.json.gz 或完整 TCM-SD 数据集存在）")
        return None
    q = build_question(task, llm_call)
    if q is None:
        _LAST_ERROR = "大模型调用或返回解析失败（网络、额度或返回格式问题）"
        return None
    with _LOCK:
        b = bank()
        key = q["src"] + "|" + task["syndrome"]
        b[key] = q
        if save:
            save_bank()
    return q
