"""Сравнение моделей-кандидатов и фиксация результата финального ансамбля."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


def compute_metrics(y_true, y_pred, y_score):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    m = compute_metrics(y_test, preds, proba)
    m["model_name"] = name
    return m


df = pd.read_csv("german.csv", sep=";")
y = df["Creditability"].to_numpy()
X = df.drop(columns=["Creditability"]).to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

seed = 42
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=3,
    class_weight="balanced", n_jobs=-1, random_state=seed,
)
hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, l2_regularization=1.0,
    early_stopping=False, random_state=seed,
)

candidates = {
    "LogisticRegression": lr,
    "RandomForest": rf,
    "HistGradientBoosting": hgb,
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=seed,
    ),
    "SoftVoting(LR+RF+HGB)": VotingClassifier(
        [("lr", lr), ("rf", rf), ("hgb", hgb)], voting="soft", n_jobs=-1,
    ),
    "Stacking(LR+RF+HGB -> LR)": StackingClassifier(
        [("lr", lr), ("rf", rf), ("hgb", hgb)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=5, n_jobs=-1,
    ),
}

print("=== Сравнение моделей-кандидатов ===")
results = []
for name, model in candidates.items():
    m = evaluate(name, model, X_train, y_train, X_test, y_test)
    results.append(m)
    print(f"{name:28s} AUC={m['roc_auc']:.4f}  acc={m['accuracy']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

final = run_ensemble_experiment(EnsembleInput(
    train_features=X_train, train_target=y_train,
    test_features=X_test, test_target=y_test,
    random_seed=42,
))

print("\n=== Финальная модель (ensemble_models.run_ensemble_experiment) ===")
print(f"model_name: {final.model_name}")
print(f"ROC-AUC: {final.roc_auc:.4f}")
print(f"Accuracy: {final.accuracy:.4f}")
print(f"Precision: {final.precision:.4f}")
print(f"Recall: {final.recall:.4f}")
print(f"F1: {final.f1:.4f}")

with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_name": final.model_name,
        "roc_auc": final.roc_auc,
        "accuracy": final.accuracy,
        "precision": final.precision,
        "recall": final.recall,
        "f1": final.f1,
    }, f, ensure_ascii=False, indent=2)

with open("experiments_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nСохранено: experiment_results.json, experiments_comparison.json")