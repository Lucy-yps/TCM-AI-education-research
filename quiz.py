# -*- coding: utf-8 -*-
"""辨证训练出题引擎（独立模块，不依赖 retrieval.py 的修改）。

设计原则：生成与判定分离。
  病历 + 证型知识库  ->  决定标准答案（不交给大模型自由发挥）
  大模型             ->  只负责把病例写成题干、并生成解析文字
  白名单 + 候选池    ->  约束选项，杜绝跨体系命名
"""
import json, os, random, threading

try:
    import compare_stats   # 相似证型鉴别的统计显著性检验
except ImportError:
    compare_stats = None

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


# ---------------------------------------------------------------- 相似证型鉴别
_COMPARE_PATH = os.path.join(BASE_DIR, "docs", "syndrome_compare.json")
_COMPARE = None


def compare_data():
    """加载相似证型鉴别数据（离线预计算，含临床层差异特征）。"""
    global _COMPARE
    if _COMPARE is None:
        _COMPARE = _load_json(_COMPARE_PATH, {"meta": {}, "syndromes": {}, "pairs": []})
    return _COMPARE


def compare_syndromes():
    """参与对比的证型列表（含各自特征症状）。"""
    cd = compare_data()
    core = set(data().get("pools", {}).get("core", []))
    out = []
    for name, info in cd.get("syndromes", {}).items():
        out.append({
            "syndrome": name,
            "tier": "core" if name in core else "extended",
            "n_cases": info.get("n_cases", 0),
            "signature": info.get("signature", []),
            "definition": info.get("definition", ""),
            "typical": info.get("typical", ""),
        })
    out.sort(key=lambda x: -x["n_cases"])
    return out


def _flip_shared(shared, flipped):
    """当对比顺序被翻转时，交换 shared 中 a/b 两列，保证与 n_a/n_b 对应。"""
    if not flipped:
        return shared
    return [{"w": x.get("w"), "a": x.get("b"), "b": x.get("a")} for x in shared]


def compare_pair(a, b):
    """返回两个证型的对比详情（含统计显著性）；顺序无关。"""
    cd = compare_data()
    syms = cd.get("syndromes", {})
    if a not in syms or b not in syms:
        return None
    found = None
    for p in cd.get("pairs", []):
        if (p["a"] == a and p["b"] == b) or (p["a"] == b and p["b"] == a):
            found = p
            break
    if found is None:
        return None
    flipped = (found["a"] != a)
    return {
        "a": a, "b": b,
        "synonym": found.get("synonym", False),
        "concept_sim": found.get("concept_sim", 0),
        "n_a": found["n_b"] if flipped else found["n_a"],
        "n_b": found["n_a"] if flipped else found["n_b"],
        "shared": found.get("shared", []),
        "only_a": found.get("only_b", []) if flipped else found.get("only_a", []),
        "only_b": found.get("only_a", []) if flipped else found.get("only_b", []),
        "info_a": syms.get(a, {}),
        "info_b": syms.get(b, {}),
        # 统计显著性：对率已知的共有症状做双比例检验 + BH-FDR 校正
        "stats": (compare_stats.annotate_pair({
            "n_a": found["n_b"] if flipped else found["n_a"],
            "n_b": found["n_a"] if flipped else found["n_b"],
            "shared": _flip_shared(found.get("shared", []), flipped),
        }) if compare_stats else None),
    }


def compare_meta():
    cd = compare_data()
    return cd.get("meta", {})


# ---------------------------------------------------------------- 学习进度追踪
_PROGRESS_PATH = os.path.join(BASE_DIR, "external_data", "quiz_progress.jsonl")


def _progress_write(rec):
    with _LOCK:
        os.makedirs(os.path.dirname(_PROGRESS_PATH), exist_ok=True)
        with open(_PROGRESS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record_answer(syndrome, chosen, correct, src=None, ms=None, kind="quiz"):
    """记录一次答题。返回累计统计。kind: quiz 辨证 | formula 方剂 | tp 舌脉。"""
    import time as _t
    rec = {
        "ts": int(_t.time()),
        "syndrome": syndrome or "",
        "chosen": chosen or "",
        "correct": bool(correct),
        "src": src or "",
        "kind": kind or "quiz",
    }
    if ms is not None:
        rec["ms"] = int(ms)
    _progress_write(rec)
    return progress_stats()


def progress_records():
    """读取全部答题记录。"""
    if not os.path.exists(_PROGRESS_PATH):
        return []
    out = []
    try:
        with open(_PROGRESS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out


def progress_stats():
    """汇总学习进度：总体、按证型、最薄弱、易混淆对、最近趋势。"""
    import time as _t
    recs = progress_records()
    # 按题型分组（旧记录无 kind，标为 legacy，不混入核心统计）
    _km = {}
    for r in recs:
        k = r.get("kind") if r.get("kind") in ("quiz", "formula", "tp") else "legacy"
        r["_kind"] = k
        d = _km.setdefault(k, {"n": 0, "c": 0})
        d["n"] += 1
        if r.get("correct"):
            d["c"] += 1
    by_kind = [{"kind": k, "n": v["n"], "correct": v["c"],
                "accuracy": round(v["c"] / v["n"], 3) if v["n"] else None}
               for k, v in sorted(_km.items(), key=lambda x: -x[1]["n"])]

    total = len(recs)
    quiz_recs = [r for r in recs if r.get("_kind") == "quiz"]
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": None, "by_syndrome": [],
                "weakest": [], "confusions": [], "recent": [], "days": 0,
                "message": "尚无答题记录"}

    right = sum(1 for r in recs if r.get("correct"))

    # 按证型
    agg = {}
    for r in quiz_recs:
        s = r.get("syndrome") or "(未知)"
        d = agg.setdefault(s, {"n": 0, "c": 0})
        d["n"] += 1
        if r.get("correct"):
            d["c"] += 1
    by_syn = [{"syndrome": s, "n": d["n"], "correct": d["c"],
               "accuracy": round(d["c"] / d["n"], 3)} for s, d in agg.items()]
    by_syn.sort(key=lambda x: (x["accuracy"], -x["n"]))

    # 易混淆对：答错时 正确证型 -> 误选证型
    conf = {}
    for r in quiz_recs:
        if r.get("correct"):
            continue
        g = r.get("syndrome") or ""
        c = r.get("chosen") or ""
        if g and c and g != c:
            k = g + " → " + c
            conf[k] = conf.get(k, 0) + 1
    confusions = [{"pair": k, "count": v} for k, v in
                  sorted(conf.items(), key=lambda x: -x[1])[:10]]

    # 最近 7 天
    now = int(_t.time())
    days = {}
    for r in recs:
        day = _t.strftime("%m-%d", _t.localtime(r.get("ts", now)))
        d = days.setdefault(day, {"n": 0, "c": 0})
        d["n"] += 1
        if r.get("correct"):
            d["c"] += 1
    recent = [{"day": k, "n": v["n"], "correct": v["c"],
               "accuracy": round(v["c"] / v["n"], 3)}
              for k, v in sorted(days.items())[-7:]]

    # 薄弱证型（至少答过 3 题）
    weakest = [x for x in by_syn if x["n"] >= 3][:5]

    return {
        "total": total,
        "correct": right,
        "accuracy": round(right / total, 4),
        "by_syndrome": by_syn,
        "weakest": weakest,
        "confusions": confusions,
        "recent": recent,
        "days": len(days),
        "by_kind": by_kind,
        "quiz_total": len(quiz_recs),
        "quiz_correct": sum(1 for r in quiz_recs if r.get("correct")),
        "quiz_accuracy": (round(sum(1 for r in quiz_recs if r.get("correct")) / len(quiz_recs), 4) if quiz_recs else None),
        "distinct_syndromes": len(agg),
    }


def progress_reset():
    """清空学习记录（不可恢复）。"""
    with _LOCK:
        if os.path.exists(_PROGRESS_PATH):
            os.remove(_PROGRESS_PATH)
    return True


# ---------------------------------------------------------------- 方剂治法联想
_FORMULA_PATH = os.path.join(BASE_DIR, "docs", "syndrome_formula.json")
_FORMULA = None


def formula_data():
    """加载方剂治法结构化数据（由知识库抽取，已通过编造检测）。"""
    global _FORMULA
    if _FORMULA is None:
        _FORMULA = _load_json(_FORMULA_PATH, {"meta": {}, "syndromes": {}, "formulas": [], "treatments": []})
    return _FORMULA


def formula_meta():
    return formula_data().get("meta", {})


def formula_concepts():
    """返回可用于练习的证型及其治法/方剂。"""
    fd = formula_data()
    out = []
    for name, e in fd.get("syndromes", {}).items():
        fang = e.get("fang", [])
        zhi = e.get("zhi_fa", [])
        if not fang and not zhi:
            continue
        out.append({
            "syndrome": name,
            "treatments": zhi,
            "formulas": fang,
            "n_diseases": len(e.get("diseases", [])),
        })
    out.sort(key=lambda x: -len(x["formulas"]))
    return out


def _nearest_syndromes(name, k=10):
    """取与该证型最相近的 k 个证型（基于知识库概念相似度 + 临床差异）。

    优先使用相似证型鉴别数据；若无则回退为全部证型。
    """
    cd = compare_data()
    scored = []
    for p in cd.get("pairs", []):
        if p["a"] == name:
            scored.append((p.get("concept_sim", 0), p["b"]))
        elif p["b"] == name:
            scored.append((p.get("concept_sim", 0), p["a"]))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [s for _, s in scored[:k]]


def _pick_pool(fd, key, exclude_syndrome, seed=None, prefer=None):
    """挑选干扰项。

    优先从 prefer（相近证型）中取，使干扰项与正确答案同范畴、更具迷惑性；
    不足时再从全部证型补齐。
    """
    rnd = random.Random(seed)
    syns = fd.get("syndromes", {})
    pool = []
    if prefer:
        for n in prefer:
            if n == exclude_syndrome:
                continue
            for v in syns.get(n, {}).get(key, []):
                if v and v not in pool:
                    pool.append(v)
    rnd.shuffle(pool)
    if len(pool) < 8:
        extra = []
        for n, e in syns.items():
            if n == exclude_syndrome or (prefer and n in prefer):
                continue
            for v in e.get(key, []):
                if v and v not in pool and v not in extra:
                    extra.append(v)
        rnd.shuffle(extra)
        pool.extend(extra)
    return pool


def formula_question(kind=None, syndrome=None, seed=None, distractors=3):
    """生成一道方剂/治法练习题。

    kind: 'syndrome2treatment' | 'syndrome2formula' | 'formula2syndrome'
    """
    fd = formula_data()
    concepts = formula_concepts()
    if not concepts:
        return None
    rnd = random.Random(seed)

    kinds = ["syndrome2treatment", "syndrome2formula", "formula2syndrome"]
    if kind not in kinds:
        kind = rnd.choice(kinds)

    if syndrome:
        cands = [c for c in concepts if c["syndrome"] == syndrome]
        if not cands:
            return None
        item = cands[0]
    else:
        rnd.shuffle(concepts)
        item = None
        for c in concepts:
            if kind == "syndrome2treatment" and not c["treatments"]:
                continue
            if kind == "syndrome2formula" and not c["formulas"]:
                continue
            if kind == "formula2syndrome" and not c["formulas"]:
                continue
            item = c
            break
        if item is None:
            return None

    name = item["syndrome"]

    near = _nearest_syndromes(name, 12)

    if kind == "syndrome2treatment":
        correct = rnd.choice(item["treatments"])
        pool = [v for v in _pick_pool(fd, "zhi_fa", name, seed, prefer=near) if v != correct]
        options = [correct] + pool[:distractors]
        stem = "证型「" + name + "」的治法应为："
        prompt = "治法是针对证候确定的治疗原则，例如「益气活血」「清热利湿」。"
    elif kind == "syndrome2formula":
        correct = rnd.choice(item["formulas"])
        pool = [v for v in _pick_pool(fd, "fang", name, seed, prefer=near) if v != correct]
        options = [correct] + pool[:distractors]
        stem = "证型「" + name + "」的代表方剂是："
        prompt = "方剂为治疗该证候的代表方，如同一治法下不同医家可能选用不同方剂。"
    else:
        correct = name
        correct_formula = rnd.choice(item["formulas"])
        if near:
            others = [s for s in near if s != name and any(
                c["syndrome"] == s for c in concepts)]
        else:
            others = []
        if len(others) < distractors:
            rest = [c["syndrome"] for c in concepts if c["syndrome"] != name and c["syndrome"] not in others]
            rnd.shuffle(rest)
            others.extend(rest)
        options = [correct] + others[:distractors]
        stem = "方剂「" + correct_formula + "」最常用于下列哪个证型？"
        prompt = "同一方剂可能用于多个证型，本题以其主要适应证为准。"

    options = options[:distractors + 1]
    rnd.shuffle(options)
    return {
        "kind": kind,
        "syndrome": name,
        "stem": stem,
        "hint": prompt,
        "options": options,
        "answer": correct,
        "answer_index": options.index(correct),
        "source": {
            "treatments": item["treatments"][:6],
            "formulas": item["formulas"][:6],
            "n_diseases": item["n_diseases"],
        },
        "note": "数据来源：TCM-SD 配套证型知识库的治法方剂文本（经大模型抽取，编造检测通过率 100%）。",
    }


# ---------------------------------------------------------------- 舌脉辨识训练
_TP_PATH = os.path.join(BASE_DIR, "docs", "syndrome_tongue_pulse.json")
_TP = None


def tp_data():
    global _TP
    if _TP is None:
        _TP = _load_json(_TP_PATH, {"meta": {}, "syndromes": {}, "usable": [], "ambiguous": []})
    return _TP


def tp_meta():
    return tp_data().get("meta", {})


def _tp_sig(name):
    """该证型的鉴别性舌脉特征集合。"""
    d = tp_data()
    return set((x["kind"], x["w"]) for x in d.get("syndromes", {}).get(name, {}).get("signature", []))


def _tp_combo(name):
    """舌脉描述文本（每个维度最多 2 个特征）。"""
    d = tp_data()
    s = d.get("syndromes", {}).get(name, {})
    combo = s.get("combo")
    if combo:
        return combo
    first = {"tongue": "舌", "coat": "苔", "pulse": "脉"}
    parts = []
    for kind in ("tongue", "coat", "pulse"):
        ks = [y for y in s.get("signature", []) if y["kind"] == kind]
        if ks:
            parts.append(first[kind] + ks[0]["w"])
    return "，".join(parts)


def tp_syndromes():
    """可用于舌脉训练的证型列表。"""
    d = tp_data()
    usable = d.get("usable", [])
    syms = d.get("syndromes", {})
    out = []
    for n in usable:
        s = syms.get(n, {})
        out.append({
            "syndrome": n,
            "n_cases": s.get("n_cases", 0),
            "combo": s.get("combo", ""),
            "signature": s.get("signature", []),
        })
    out.sort(key=lambda x: -x["n_cases"])
    return out


def tp_question(syndrome=None, seed=None, distractors=3):
    """生成舌脉辨识题。

    题型设计说明（重要）：
      反向题（"看舌脉选证型"）在中医理论上不成立——舌脉在多个证型间高度重叠，
      单独一组舌脉无法唯一确定证型（必须四诊合参）。实测中 27 个可用证型里
      只有 1 个能构成无歧义的反向题。
      因此采用正向题："证型 X 的舌脉表现是？"，正确答案为该证型真实病历中的
      高频舌脉组合，答案唯一且符合教学逻辑。
    """
    d = tp_data()
    usable = d.get("usable", [])
    syms = d.get("syndromes", {})
    if len(usable) < distractors + 1:
        return None
    rnd = random.Random(seed)

    def combo_of(n):
        s = syms.get(n, {})
        c = s.get("combo")
        if c:
            return c
        first = {"tongue": "舌", "coat": "苔", "pulse": "脉"}
        parts = []
        for kind in ("tongue", "coat", "pulse"):
            ks = [y for y in s.get("signature", []) if y["kind"] == kind]
            if ks:
                w = ks[0]["w"]
                pre = first[kind]
                # 特征名里已有该维度字就不重复加前缀（避免"苔少苔"）
                parts.append(w if pre in w else pre + w)
        return "，".join(parts)

    if syndrome:
        if syndrome not in syms or syndrome not in usable:
            return None
        target = syndrome
    else:
        pool = [n for n in usable if combo_of(n)]
        if not pool:
            return None
        rnd.shuffle(pool)
        target = pool[0]

    correct = combo_of(target)
    # 干扰项：舌脉组合与正确答案不同的证型
    pool = []
    for n in usable:
        if n == target:
            continue
        c = combo_of(n)
        if not c or c == correct:
            continue
        pool.append(c)
    # 去重
    seen = []
    for c in pool:
        if c not in seen:
            seen.append(c)
    rnd.shuffle(seen)
    if len(seen) < distractors:
        return None
    options = [correct] + seen[:distractors]
    rnd.shuffle(options)

    sig = syms.get(target, {}).get("signature", [])
    return {
        "kind": "syndrome2tp",
        "syndrome": target,
        "combo": correct,
        "signature": sig,
        "n_cases": syms.get(target, {}).get("n_cases", 0),
        "stem": "证型「" + target + "」的舌脉表现是：",
        "options": options,
        "answer": correct,
        "answer_index": options.index(correct),
        "evidence": [{"w": x["w"], "kind": x["kind"], "rate": x["rate"], "df": x["df"]}
                     for x in sig],
        "note": ("舌脉特征来自 TCM-SD 真实病历统计（该证型病例中的出现率）。"
                 "舌脉需四诊合参，不能单独作为辨证依据；本题用于训练证型与舌脉的对应记忆。"),
    }
