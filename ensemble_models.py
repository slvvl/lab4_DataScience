"""Эксперимент с ансамблем на german.csv."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def _to_matrix(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _to_vector(y) -> np.ndarray:
    return np.asarray(y).reshape(-1)


def _binarize(y_train, y_test):
    """Приводим метки к 0/1, если пришли строки или другие классы."""
    classes = np.unique(y_train)
    if classes.size == 2 and not np.array_equal(classes.astype(int), [0, 1]):
        pos = classes[1]
        return (y_train == pos).astype(int), (y_test == pos).astype(int)
    return y_train.astype(int), y_test.astype(int)


def _build_model(seed: int) -> StackingClassifier:
    lr_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    return StackingClassifier(
        estimators=[("lr", lr_pipe), ("rf", rf), ("hgb", hgb)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=5,
        n_jobs=-1,
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    X_train = _to_matrix(data.train_features)
    y_train = _to_vector(data.train_target)
    X_test = _to_matrix(data.test_features)
    y_test = _to_vector(data.test_target)
    seed = int(data.random_seed)

    y_train_bin, y_test_bin = _binarize(y_train, y_test)

    model = _build_model(seed)
    model.fit(X_train, y_train_bin)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Stacking(LR + RandomForest + HistGB -> LR meta)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test_bin, probabilities)),
        accuracy=float(accuracy_score(y_test_bin, predictions)),
        precision=float(precision_score(y_test_bin, predictions, zero_division=0)),
        recall=float(recall_score(y_test_bin, predictions, zero_division=0)),
        f1=float(f1_score(y_test_bin, predictions, zero_division=0)),
    )