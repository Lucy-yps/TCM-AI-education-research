# -*- coding: utf-8 -*-
"""训练 TF-IDF + LogisticRegression 中医证型分类器，并在 dev 694 条上评测。

用法：
    python classifier.py            # 训练 + 评测
    训练好的模型保存到 external_data/tcm_clf.joblib
"""
import json
import os
import pickle
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report



LABEL_MAP = {
    "气滞血瘀证": "气滞血瘀", "肝胃不和证": "肝胃不和", "肝阳上亢证": "肝阳上亢",
    "肝郁气滞证": "肝郁气滞", "心血虚证": "心血虚", "气血两虚证": "气血两虚",
    "痰湿内阻证": "痰湿内阻", "肝郁脾虚证": "肝郁脾虚", "肝胆湿热证": "肝胆湿热",
    "肾气虚证": "肾气虚", "大肠湿热证": "大肠湿热", "中气下陷证": "中气下陷",
    "脾胃气虚证": "脾胃气虚", "阳虚寒凝证": "阳虚寒凝", "肺阴虚证": "肺阴虚",
    "肾阳虚证": "肾阳虚",
}


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


def main():
    train = load_jsonl(os.path.join(BASE, "external_data/train_dev/train.json"))
    dev = load_jsonl(os.path.join(BASE, "external_data/train_dev/dev.json"))

    def filter_align(rows):
        X, y = [], []
        for r in rows:
            lab = r.get("norm_syndrome") or r.get("syndrome")
            if lab in LABEL_MAP:
                X.append(make_text(r))
                y.append(LABEL_MAP[lab])
        return X, y

    X_train, y_train = filter_align(train)
    X_test, y_test = filter_align(dev)
    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
    print("训练标签分布:", dict(Counter(y_train).most_common()))

    # 和检索方法同样的特征：char_wb 1-2 gram，保证公平对比
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), sublinear_tf=True, max_features=50000)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    ])
    clf.fit(X_train, y_train)

    # 评测
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    classes = clf.classes_

    # Top-1 / Top-3
    top1 = sum(int(p == g) for p, g in zip(pred, y_test)) / len(y_test)
    top3 = 0
    for i, g in enumerate(y_test):
        top3_labels = classes[proba[i].argsort()[::-1][:3]]
        if g in top3_labels:
            top3 += 1
    top3 /= len(y_test)

    print(f"\n[classifier] Top-1={top1*100:.1f}%  Top-3={top3*100:.1f}%")
    print("\n按类别：")
    report = classification_report(y_test, pred, zero_division=0)
    print(report)

    # 保存模型
    model_path = os.path.join(BASE, "external_data/tcm_clf.joblib")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()
