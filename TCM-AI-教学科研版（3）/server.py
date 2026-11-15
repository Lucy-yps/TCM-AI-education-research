# -*- coding: utf-8 -*-
"""TCM pattern-matching research demo. Not a diagnostic or prescribing tool.

本版在检索层之上增加了两条能力：
  1) 输入风格路由：口语描述走检索、电子病历体走训练分类器（语料实测两者互补）；
  2) 按准确率校准的置信度与拒答：置信度来自参考分布上的实测准确率，而非分数绝对值。
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor

import retrieval
import quiz

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

AI_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "api_url": os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "zhipu": {
        "name": "智谱",
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "api_url": os.getenv("ZHIPU_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
        "model": os.getenv("ZHIPU_MODEL", "glm-4-flash"),
    },
    "modelscope": {
        "name": "ModelScope",
        "api_key": os.getenv("MODELSCOPE_API_KEY", ""),
        "api_url": os.getenv("MODELSCOPE_API_URL", "https://api-inference.modelscope.cn/v1/chat/completions"),
        "model": os.getenv("MODELSCOPE_MODEL", "Qwen/Qwen3.8-Flash-Next"),
    },
}

# 启动时预热 BM25 / TF-IDF 索引，避免首次请求卡顿
retrieval.match_tfidf("预热", top_k=1)
retrieval.match_bm25("预热", top_k=1)


@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.html"), encoding="utf-8") as f:
        return f.read()


def build_analysis_prompt(symptoms, analysis):
    """只把检索证据交给模型做教学性摘要，并明确告知证据强度与降级状态。"""
    matches = analysis["answer_matches"][:5]
    evidence = [{"syndrome": r["syndrome"], "score": r["score"], "matched": r["matched_symptoms"]}
                for r in matches if r["score"] > 0]
    conf = analysis["answer_confidence"]
    route = analysis["route"]
    return f"""你是面向中医专业教学与科研场景的知识助手，不是医生，不做诊断。只基于下方研究知识库条目，做教学科研用途的证据摘要。
用户原文：{symptoms}

系统判定的输入风格：{analysis['style']['style']}（{analysis['style']['reason']}）
采用方法：{analysis['answer_method']}（{route['rationale']}）
置信等级：{conf['label']}（{conf['reason']}）
是否降级候选：{'是——首选方法置信度不足，以下仅为回落候选' if analysis['answer_is_fallback'] else '否'}
检索证据：{json.dumps(evidence, ensure_ascii=False)}

规则：明确说明这是词条匹配而非诊断；不得声称确诊、开方、给剂量或替代就医；证据不足时指出缺少的信息，不得补造舌脉或病史；若为降级候选或置信不足，必须在摘要开头说明结果不可作为结论；只总结证据、分歧和需要由专业人员核实的问题。中文，简洁分点。"""


def call_model(provider_id, provider, prompt):
    if not provider["api_key"]:
        return {
            "provider": provider_id, "name": provider["name"], "model": provider["model"],
            "analysis": None, "status": "未配置 API Key",
        }
    try:
        res = requests.post(provider["api_url"], headers={"Authorization": f"Bearer {provider['api_key']}"}, json={
            "model": provider["model"],
            "messages": [{"role": "system", "content": "你是谨慎的中医药知识教学助手，服务教学科研场景，不能诊断或治疗。"},
                         {"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": 1000,
        }, timeout=60)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回内容为空")
        return {
            "provider": provider_id, "name": provider["name"], "model": provider["model"],
            "analysis": content.strip(), "status": "已生成教学性证据摘要",
        }
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
        return {
            "provider": provider_id, "name": provider["name"], "model": provider["model"],
            "analysis": None, "status": f"模型服务暂不可用：{type(exc).__name__} {detail[:150]}",
        }


def compare_models(symptoms, analysis):
    prompt = build_analysis_prompt(symptoms, analysis)
    with ThreadPoolExecutor(max_workers=len(AI_PROVIDERS)) as executor:
        futures = [executor.submit(call_model, provider_id, provider, prompt)
                   for provider_id, provider in AI_PROVIDERS.items()]
        return [future.result() for future in futures]


@app.get("/api/health")
def health():
    cov = retrieval.classifier_coverage()
    return jsonify({
        "status": "ok",
        "knowledge_count": len(retrieval.KNOWLEDGE_BASE),
        "methods": list(retrieval.METHODS.keys()),
        "models_configured": {pid: bool(p["api_key"]) for pid, p in AI_PROVIDERS.items()},
        "mode": "research_demo",
        "routing": {k: v["primary"] for k, v in retrieval.STYLE_ROUTE.items()},
        "classifier": {"trained_count": cov["trained_count"], "uncovered_count": cov["uncovered_count"]},
        "calibration": {
            "colloquial_reference_n": retrieval._calibration_table().get("n_cases", 0),
            "record_reference_n": retrieval._record_calibration().get("n_cases", 0),
        },
    })


@app.get("/api/calibration")
def calibration():
    """把校准表原样暴露：每个分数档到底对应多高的实测准确率。"""
    return jsonify({
        "code": 200,
        "colloquial": retrieval.calibration_summary(),
        "record": retrieval._record_calibration(),
        "classifier": {
            "reliable": retrieval.RECORD_CONF_RELIABLE,
            "hint": retrieval.RECORD_CONF_HINT,
            "stop": retrieval.RECORD_CONF_STOP,
        },
        "coverage": retrieval.classifier_coverage(),
        "note": "置信度按参考分布上的实测准确率分级，不是概率意义上的置信区间，也不代表临床有效性。",
    })


@app.get("/api/syndromes")
def syndromes():
    return jsonify({"code": 200, "data": [
        {"name": x["syndrome"], "symptoms": x["symptoms"], "category": x.get("category", "")}
        for x in retrieval.KNOWLEDGE_BASE]})


@app.get("/api/graph")
def graph():
    nodes, links, node_map = [], [], {}

    def node(name, category, size):
        if name not in node_map:
            node_map[name] = len(nodes)
            nodes.append({"id": node_map[name], "name": name, "category": category, "symbolSize": size})
        return node_map[name]

    for item in retrieval.KNOWLEDGE_BASE:
        sid = node(item["syndrome"], 0, 48)
        for symptom in item["symptoms"]:
            links.append({"source": sid, "target": node(symptom, 1, 24)})
        detail = item.get("detail", {})
        formula = detail.get("formula", {}) if isinstance(detail, dict) else {}
        if isinstance(formula, dict) and formula.get("name"):
            links.append({"source": sid, "target": node(formula["name"], 2, 34)})
    return jsonify({"nodes": nodes, "links": links,
                    "total_nodes": len(nodes), "total_links": len(links)})


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("symptoms"), str):
        return jsonify({"code": 400, "message": "请以文本形式提交症状描述"}), 400
    symptoms = payload["symptoms"].strip()
    if not symptoms:
        return jsonify({"code": 400, "message": "请输入症状信息"}), 400
    if len(symptoms) > 2000:
        return jsonify({"code": 400, "message": "输入过长，请控制在2000字以内"}), 400

    # 一次调用完成：风格判定 -> 路由 -> 五方法评估 -> 分通道置信度 -> 拒答/降级
    analysis = retrieval.route_query(symptoms)

    primary = analysis["answer_matches"]
    ai_comparison = compare_models(symptoms, analysis)
    deepseek_result = next((r for r in ai_comparison if r["provider"] == "deepseek"), None)
    available_models = [r["name"] for r in ai_comparison if r["analysis"]]
    ai_status = ("已完成模型对照：" + "、".join(available_models)) if available_models \
        else "未配置可用模型 API；显示本地可解释匹配结果"

    return jsonify({"code": 200, "data": {
        "input_symptoms": symptoms,

        # ---- 新增：风格判定、路由、回答口径 ----
        "style": analysis["style"],
        "route": analysis["route"],
        "status": analysis["status"],
        "status_note": analysis["status_note"],
        "limitations": analysis["limitations"],
        "answer_method": analysis["answer_method"],
        "answer_matches": analysis["answer_matches"],
        "answer_confidence": analysis["answer_confidence"],
        "answer_is_fallback": analysis["answer_is_fallback"],

        # ---- 兼容旧字段 ----
        "observed_symptoms": analysis["observed_symptoms"],
        "unmatched_text": analysis["unmatched_text"],
        "confidence": analysis["confidence"]["level"],
        "confidence_detail": analysis["confidence"],
        "local_matches": primary,
        "method_results": analysis["method_results"],
        "ai_comparison": ai_comparison,
        "ai_analysis": deepseek_result["analysis"] if deepseek_result else None,
        "ai_status": ai_status,
        "use_ai": bool(available_models),
    }})


# ---------------------------------------------------------------- 辨证训练（出题引擎）
# 说明：出题逻辑封装在 quiz.py（独立模块），此处只做接口编排，不改动既有检索/分析逻辑。
def _quiz_llm(prompt):
    """调用 DeepSeek 生成题目正文；答案由 quiz.py 给定，模型不得更改。"""
    provider = AI_PROVIDERS["deepseek"]
    api_key = str(provider.get("api_key") or "").strip()
    if not api_key:
        return None
    try:
        res = requests.post(provider["api_url"], headers={"Authorization": f"Bearer {api_key}"}, json={
            "model": provider["model"],
            "messages": [{"role": "system", "content": "你是中医诊断学命题教师，只输出 JSON。"},
                         {"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 2000,
        }, timeout=150)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        return json.loads(content[start:end + 1])
    except Exception:
        return None


@app.get("/quiz.html")
def quiz_page():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quiz.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/quiz/stats")
def quiz_stats():
    try:
        return jsonify({"code": 200, "data": quiz.stats()})
    except Exception as exc:
        return jsonify({"code": 500, "message": f"出题引擎不可用：{type(exc).__name__}"}), 500


@app.get("/api/quiz/syndromes")
def quiz_syndromes():
    try:
        return jsonify({"code": 200, "data": quiz.list_syndromes()})
    except Exception as exc:
        return jsonify({"code": 500, "message": f"出题引擎不可用：{type(exc).__name__}"}), 500


@app.post("/api/quiz/generate")
def quiz_generate():
    payload = request.get_json(silent=True) or {}
    syndrome = payload.get("syndrome") if isinstance(payload.get("syndrome"), str) else None
    if syndrome == "":
        syndrome = None
    try:
        question = quiz.generate_one(syndrome, llm_call=_quiz_llm)
    except Exception as exc:
        return jsonify({"code": 500, "message": f"出题失败：{type(exc).__name__} {exc}"}), 500
    if question is None:
        return jsonify({"code": 404, "message": quiz.last_error() or "生成失败，请稍后重试"}), 404
    return jsonify({"code": 200, "data": {
        "syndrome": question.get("syndrome"),
        "stem": question.get("stem"),
        "options": question.get("options"),
        "answer": question.get("answer"),
        "explain": question.get("explain"),
        "key_points": question.get("key_points"),
        "distractor_reasons": question.get("distractor_reasons"),
        "source": {"split": question.get("src", "").split("#")[0],
                   "line": question.get("src", "").split("#")[-1],
                   "lcd_name": question.get("lcd_name", "")},
        "quality": question.get("checks", {}),
    }})


@app.get("/api/quiz/bank")
def quiz_bank():
    b = quiz.bank()
    return jsonify({"code": 200, "data": {
        "size": len(b),
        "passed": sum(1 for q in b.values() if (q.get("checks") or {}).get("passed")),
        "by_syndrome": {},
    }})



# ---------------------------------------------------------------- 相似证型鉴别
@app.get("/compare.html")
def compare_page():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/quiz/compare/syndromes")
def quiz_compare_syndromes():
    try:
        return jsonify({"code": 200, "data": quiz.compare_syndromes(),
                        "meta": quiz.compare_meta()})
    except Exception as exc:
        return jsonify({"code": 500, "message": "对比数据不可用：" + type(exc).__name__}), 500


@app.get("/api/quiz/compare")
def quiz_compare():
    a = (request.args.get("a") or "").strip()
    b = (request.args.get("b") or "").strip()
    if not a or not b:
        return jsonify({"code": 400, "message": "需要参数 a 与 b（两个证型名）"}), 400
    if a == b:
        return jsonify({"code": 400, "message": "请选择两个不同的证型"}), 400
    try:
        r = quiz.compare_pair(a, b)
    except Exception as exc:
        return jsonify({"code": 500, "message": "对比失败：" + type(exc).__name__}), 500
    if r is None:
        return jsonify({"code": 404, "message": "未找到该证型对（请确认两者都在核心出题池内）"}), 404
    return jsonify({"code": 200, "data": r})



# ---------------------------------------------------------------- 学习进度追踪
@app.post("/api/quiz/record")
def quiz_record():
    payload = request.get_json(silent=True) or {}
    syndrome = str(payload.get("syndrome") or "").strip()
    chosen = str(payload.get("chosen") or "").strip()
    if not syndrome or not chosen:
        return jsonify({"code": 400, "message": "需要 syndrome 与 chosen 两个字段"}), 400
    try:
        stats = quiz.record_answer(syndrome, chosen, syndrome == chosen,
                                   src=payload.get("src"), ms=payload.get("ms"))
    except Exception as exc:
        return jsonify({"code": 500, "message": "记录失败：" + type(exc).__name__}), 500
    return jsonify({"code": 200, "data": stats})


@app.get("/api/quiz/progress")
def quiz_progress():
    try:
        return jsonify({"code": 200, "data": quiz.progress_stats()})
    except Exception as exc:
        return jsonify({"code": 500, "message": "读取失败：" + type(exc).__name__}), 500


@app.post("/api/quiz/progress/reset")
def quiz_progress_reset():
    try:
        quiz.progress_reset()
    except Exception as exc:
        return jsonify({"code": 500, "message": "重置失败：" + type(exc).__name__}), 500
    return jsonify({"code": 200, "data": quiz.progress_stats()})


@app.get("/progress.html")
def progress_page():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.html"), encoding="utf-8") as f:
        return f.read()



# ---------------------------------------------------------------- 方剂治法联想
@app.get("/api/quiz/formula/meta")
def quiz_formula_meta():
    try:
        return jsonify({"code": 200, "data": quiz.formula_meta(),
                        "concepts": quiz.formula_concepts()})
    except Exception as exc:
        return jsonify({"code": 500, "message": "方剂数据不可用：" + type(exc).__name__}), 500


@app.post("/api/quiz/formula")
def quiz_formula():
    payload = request.get_json(silent=True) or {}
    kind = payload.get("kind")
    syndrome = payload.get("syndrome") or None
    if kind not in (None, "", "syndrome2treatment", "syndrome2formula", "formula2syndrome"):
        return jsonify({"code": 400, "message": "kind 取值不合法"}), 400
    try:
        q = quiz.formula_question(kind=kind or None, syndrome=syndrome)
    except Exception as exc:
        return jsonify({"code": 500, "message": "出题失败：" + type(exc).__name__}), 500
    if q is None:
        return jsonify({"code": 404, "message": "该证型暂无治法方剂数据"}), 404
    return jsonify({"code": 200, "data": q})


@app.get("/formula.html")
def formula_page():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "formula.html"), encoding="utf-8") as f:
        return f.read()



# ---------------------------------------------------------------- 舌脉辨识训练
@app.get("/api/quiz/tp/meta")
def quiz_tp_meta():
    try:
        return jsonify({"code": 200, "data": quiz.tp_meta(),
                        "syndromes": quiz.tp_syndromes()})
    except Exception as exc:
        return jsonify({"code": 500, "message": "舌脉数据不可用：" + type(exc).__name__}), 500


@app.post("/api/quiz/tp")
def quiz_tp():
    payload = request.get_json(silent=True) or {}
    syndrome = payload.get("syndrome") or None
    try:
        q = quiz.tp_question(syndrome=syndrome)
    except Exception as exc:
        return jsonify({"code": 500, "message": "出题失败：" + type(exc).__name__}), 500
    if q is None:
        return jsonify({"code": 404, "message": "该证型暂无足够的舌脉数据出题"}), 404
    return jsonify({"code": 200, "data": q})


@app.get("/tongue.html")
def tongue_page():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tongue.html"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    configured = [p["name"] for p in AI_PROVIDERS.values() if p["api_key"]]
    cov = retrieval.classifier_coverage()
    print(f"TCM-AI research demo | {len(retrieval.KNOWLEDGE_BASE)} syndromes | methods: {', '.join(retrieval.METHODS.keys())}")
    print(f"AI providers: {', '.join(configured) or 'none'}")
    print(f"routing: {retrieval.STYLE_ROUTE['colloquial']['primary']} (colloquial) / "
          f"{retrieval.STYLE_ROUTE['record']['primary']} (record)")
    print(f"classifier covers {cov['trained_count']}/{cov['knowledge_count']} syndromes")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
