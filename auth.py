# -*- coding: utf-8 -*-
"""轻量账户系统：用户名 + 密码 + 头像 + 个人信息。

存储：external_data/users.json + tokens.json（无数据库）
密码：werkzeug pbkdf2 哈希，不存明文
会话：随机 token，存文件，重启不丢
"""
import json
import os
import re
import secrets
import threading
import time

from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "external_data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")
TOKENS_PATH = os.path.join(DATA_DIR, "tokens.json")

_LOCK = threading.RLock()

# 预设头像（emoji，避免上传文件）
AVATARS = [
    "🩺", "🌿", "🦉", "🐼", "🦊", "🐯",
    "🍵", "📜", "🔬", "🧪", "🪷", "🐉",
    "🧑‍⚕️", "👩‍🎓", "👨‍🎓", "🦌",
]


def _load(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path, data):
    with _LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)


def _safe_name(u):
    """用户名只允许字母数字下划线中文短横，3-20 位。"""
    u = str(u or "").strip()
    if not re.fullmatch(r"[\w\u4e00-\u9fa5-]{2,20}", u):
        return None
    return u


def users():
    return _load(USERS_PATH, {"users": {}})


def _tokens():
    return _load(TOKENS_PATH, {"tokens": {}})


# ---------------------------------------------------------------- 注册 / 登录

def register(username, password, display=None, avatar=None):
    u = _safe_name(username)
    if not u:
        return False, "用户名需 2-20 位，仅支持中英文、数字、下划线、短横线"
    if not password or len(str(password)) < 4:
        return False, "密码至少 4 位"
    data = users()
    if u in data["users"]:
        return False, "该用户名已被注册"
    av = avatar if avatar in AVATARS else AVATARS[0]
    data["users"][u] = {
        "username": u,
        "display": (str(display or u).strip()[:20]) or u,
        "pwd_hash": generate_password_hash(str(password)),
        "avatar": av,
        "created": int(time.time()),
        "last_login": 0,
    }
    _save(USERS_PATH, data)
    return True, u


def login(username, password):
    u = _safe_name(username)
    if not u:
        return False, "用户名或密码不正确"
    data = users()
    rec = data["users"].get(u)
    if not rec or not check_password_hash(rec.get("pwd_hash", ""), str(password or "")):
        return False, "用户名或密码不正确"
    rec["last_login"] = int(time.time())
    _save(USERS_PATH, data)

    tok = secrets.token_hex(16)
    tk = _tokens()
    tk["tokens"][tok] = {"user": u, "ts": int(time.time())}
    _save(TOKENS_PATH, tk)
    return True, tok


def logout(token):
    tk = _tokens()
    if token in tk["tokens"]:
        del tk["tokens"][token]
        _save(TOKENS_PATH, tk)
    return True


def whoami(token):
    """返回用户信息（不含密码哈希）。"""
    if not token:
        return None
    tk = _tokens()
    rec = tk["tokens"].get(token)
    if not rec:
        return None
    u = rec["user"]
    data = users()
    info = data["users"].get(u)
    if not info:
        return None
    return _public(info)


def _public(info):
    return {
        "username": info.get("username"),
        "display": info.get("display") or info.get("username"),
        "avatar": info.get("avatar") or AVATARS[0],
        "created": info.get("created"),
        "last_login": info.get("last_login"),
    }


# ---------------------------------------------------------------- 个人信息

def update_profile(token, display=None, avatar=None):
    rec = _tokens()["tokens"].get(token)
    if not rec:
        return False, "未登录"
    u = rec["user"]
    data = users()
    info = data["users"].get(u)
    if not info:
        return False, "用户不存在"
    if display is not None:
        d = str(display).strip()[:20]
        if d:
            info["display"] = d
    if avatar is not None and avatar in AVATARS:
        info["avatar"] = avatar
    _save(USERS_PATH, data)
    return True, _public(info)


def change_password(token, old_pwd, new_pwd):
    rec = _tokens()["tokens"].get(token)
    if not rec:
        return False, "未登录"
    u = rec["user"]
    data = users()
    info = data["users"].get(u)
    if not info:
        return False, "用户不存在"
    if not check_password_hash(info.get("pwd_hash", ""), str(old_pwd or "")):
        return False, "原密码不正确"
    if not new_pwd or len(str(new_pwd)) < 4:
        return False, "新密码至少 4 位"
    info["pwd_hash"] = generate_password_hash(str(new_pwd))
    _save(USERS_PATH, data)
    # 改密后清掉该用户所有 token（强制重新登录）
    tk = _tokens()
    for k in [k for k, v in tk["tokens"].items() if v.get("user") == u]:
        del tk["tokens"][k]
    _save(TOKENS_PATH, tk)
    return True, "密码已更新，请重新登录"


def stats_for(user):
    """该用户的答题统计（供个人信息页用）。"""
    if not user:
        return {}
    path = os.path.join(DATA_DIR, "quiz_progress_" + re.sub(r"[^\w\u4e00-\u9fa5-]", "", user) + ".jsonl")
    if not os.path.exists(path):
        return {"total": 0, "correct": 0, "days": 0}
    total = 0
    right = 0
    days = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                total += 1
                if r.get("correct"):
                    right += 1
                if r.get("ts"):
                    days.add(time.strftime("%Y-%m-%d", time.localtime(r["ts"])))
    except OSError:
        pass
    return {"total": total, "correct": right, "days": len(days),
            "accuracy": round(right / total, 4) if total else None}
