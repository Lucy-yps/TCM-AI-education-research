# -*- coding: utf-8 -*-
"""考试模式：从题库组卷、判分、记录成绩。

要点：
  - 题库里正确选项固定在第一个，组卷时**必须打乱**，否则可被位置规律猜出
  - 下发题目时剥掉 answer / explain，交卷后才返回解析
  - 试卷存内存（不跨重启），成绩存文件
"""
import json
import os
import random
import re
import secrets
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "external_data")
BANK_PATH = os.path.join(DATA_DIR, "quiz_bank.json")

_LOCK = threading.RLock()
_BANK = None
_EXAMS = {}          # exam_id -> {"user":..., "ts":..., "items":[{stem,options,answer,explain,key_points}]}
_EXAM_TTL = 3 * 3600


def bank():
    """加载题库（懒加载 + 缓存）。"""
    global _BANK
    with _LOCK:
        if _BANK is None:
            try:
                with open(BANK_PATH, encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, ValueError):
                raw = {}
            items = []
            for key, v in raw.items():
                if not isinstance(v, dict):
                    continue
                opts = v.get("options") or []
                ans = (v.get("answer") or "").strip()
                stem = (v.get("stem") or "").strip()
                if len(opts) < 2 or not ans or not stem:
                    continue
                if ans not in opts:
                    continue
                meta = key.split("|")
                items.append({
                    "src": key,
                    "syndrome": meta[1] if len(meta) > 1 else "",
                    "stem": stem,
                    "options": list(opts),
                    "answer": ans,
                    "explain": (v.get("explain") or "").strip(),
                    "key_points": v.get("key_points") or [],
                })
            _BANK = items
    return _BANK


def _cleanup():
    now = int(time.time())
    for k in [k for k, v in _EXAMS.items() if now - v["ts"] > _EXAM_TTL]:
        _EXAMS.pop(k, None)


def start(n=10, user=None, syndrome=None):
    """组卷。返回 exam_id 与题目（不含答案）。"""
    items = bank()
    if syndrome:
        pool = [x for x in items if x["syndrome"] == syndrome]
        if len(pool) < 3:
            pool = items
    else:
        pool = items
    if not pool:
        return None, "题库为空"
    n = max(1, min(int(n or 10), 50, len(pool)))
    picked = random.sample(pool, n)

    qs = []
    for i, it in enumerate(picked):
        opts = list(it["options"])
        random.shuffle(opts)          # 关键：打乱，避免答案总在 A
        qs.append({
            "idx": i + 1,
            "stem": it["stem"],
            "options": opts,
            "_answer": it["answer"],   # 暂存服务端，不下发
            "_explain": it["explain"],
            "_key_points": it["key_points"],
        })

    eid = secrets.token_hex(8)
    with _LOCK:
        _cleanup()
        _EXAMS[eid] = {"user": user or "", "ts": int(time.time()),
                       "items": qs, "n": n}

    # 下发时剥掉答案
    public = [{"idx": q["idx"], "stem": q["stem"], "options": q["options"]} for q in qs]
    return {"exam_id": eid, "n": n, "seconds": n * 60, "questions": public}, None


def submit(exam_id, answers, user=None, elapsed=None):
    """判分。answers: {题号: 选项文本} 或 列表。"""
    with _LOCK:
        rec = _EXAMS.get(exam_id)
    if not rec:
        return None, "试卷不存在或已过期，请重新开始"

    items = rec["items"]
    if isinstance(answers, list):
        amap = {str(i + 1): a for i, a in enumerate(answers)}
    elif isinstance(answers, dict):
        amap = {str(k): v for k, v in answers.items()}
    else:
        amap = {}

    details = []
    right = 0
    for q in items:
        mine = (amap.get(str(q["idx"])) or "").strip()
        ok = (mine == q["_answer"])
        if ok:
            right += 1
        details.append({
            "idx": q["idx"],
            "stem": q["stem"],
            "options": q["options"],
            "mine": mine,
            "answer": q["_answer"],
            "correct": ok,
            "explain": q["_explain"],
            "key_points": q["_key_points"],
        })

    n = rec["n"]
    score = round(right / n * 100, 1) if n else 0

    # 记录成绩
    u = user or rec.get("user") or ""
    safe = re.sub(r"[^\w\u4e00-\u9fa5-]", "", u)[:24]
    path = os.path.join(DATA_DIR, ("exams_" + safe + ".jsonl") if safe else "exams.jsonl")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": int(time.time()),
                    "user": safe,
                    "n": n,
                    "right": right,
                    "score": score,
                    "elapsed": int(elapsed) if elapsed else None,
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass

    with _LOCK:
        _EXAMS.pop(exam_id, None)      # 交卷后作废，防重复提交

    return {
        "n": n, "right": right, "wrong": n - right, "score": score,
        "elapsed": int(elapsed) if elapsed else None,
        "details": details,
    }, None


def history(user=None, limit=10):
    """最近考试成绩。"""
    u = re.sub(r"[^\w\u4e00-\u9fa5-]", "", user or "")[:24]
    path = os.path.join(DATA_DIR, ("exams_" + u + ".jsonl") if u else "exams.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
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
    out.sort(key=lambda x: -x.get("ts", 0))
    return out[:limit]
