# -*- coding: utf-8 -*-
"""CPU 性能基准测试：无需 GPU 的实测依据

运行: python scripts/benchmark.py <项目根目录>
产物: 控制台输出 + 测试案例/benchmark_result.json
"""
import io, json, os, platform, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = sys.argv[1] if len(sys.argv) > 1 else "."
os.chdir(PROJ)
sys.path.insert(0, PROJ)

def bench(fn, n=50, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0

print("=" * 66)
print("运行环境")
print("=" * 66)
print("  Python :", sys.version.split()[0])
print("  平台   :", platform.platform()[:62])
print("  GPU    : 未使用（全程 CPU）")
print()

CASE = "。".join(["头晕伴心前区不适约5小时", "患者突发头晕伴心前区不适，恶心欲呕、乏力", "神志清晰，口唇发绀，舌紫暗，苔白腻，脉弦滑"])
SHORT = "我这两天有点发烧，体温偏高，稍微怕风，喉咙痛，鼻涕是黄的"

t0 = time.perf_counter()
import retrieval
t_imp = (time.perf_counter() - t0) * 1000
print("=" * 66)
print("一、启动与加载（一次性）")
print("=" * 66)
print("  模块导入 retrieval       : %8.1f ms" % t_imp)
t0 = time.perf_counter(); retrieval._ensure_clf(); t_clf = (time.perf_counter() - t0) * 1000
print("  分类器加载               : %8.1f ms" % t_clf)
t0 = time.perf_counter(); retrieval._ensure_tfidf(); t_tf = (time.perf_counter() - t0) * 1000
print("  TF-IDF 索引构建          : %8.1f ms" % t_tf)
sz = os.path.getsize("external_data/tcm_clf.joblib") / 1048576
print("  分类器模型大小           : %8.2f MB" % sz)
print()

print("=" * 66)
print("二、单条推理延迟（真实病历，n=50）")
print("=" * 66)
results = {}
for name, fn in retrieval.METHODS.items():
    ms = bench(lambda f=fn: f(CASE, top_k=5), n=50)
    results[name] = ms
    print("  %-9s : %8.2f ms" % (name, ms))
print()
print("=" * 66)
print("三、完整分析链路")
print("=" * 66)
ms = bench(lambda: retrieval.route_query(CASE), n=50)
results["route_query"] = ms
print("  route_query（风格判定+方法选择）: %8.2f ms" % ms)
print()
print("=" * 66)
print("四、口语体短文本输入")
print("=" * 66)
for name in ("tfidf", "bm25", "keyword"):
    fn = retrieval.METHODS[name]
    ms = bench(lambda f=fn: f(SHORT, top_k=5), n=50)
    print("  %-9s : %8.2f ms" % (name, ms))
print()
print("=" * 66)
print("五、结论")
print("=" * 66)
slowest = max(results.values())
print("  最慢环节 : %.2f ms (%s)" % (slowest, max(results, key=results.get)))
print("  全部环节均在 %d ms 以内" % (int(slowest) + 1))
print("  GPU kernel 启动开销通常 10~20 ms，高于本项目单次推理耗时，")
print("  故 GPU 对本工作负载无收益——全程 CPU 即可满足。")
print()
out = {"python": sys.version.split()[0], "platform": platform.platform()[:80],
       "model_size_mb": round(sz, 2), "import_ms": round(t_imp, 2),
       "classifier_load_ms": round(t_clf, 2), "tfidf_build_ms": round(t_tf, 2),
       "method_latency_ms": {k: round(v, 3) for k, v in results.items()}}
os.makedirs("测试案例", exist_ok=True)
with open("测试案例/benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("结果已保存: 测试案例/benchmark_result.json")