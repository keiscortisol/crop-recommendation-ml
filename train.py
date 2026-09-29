"""
train.py — Crop Recommendation Classifier
==========================================
End-to-end pipeline: load data -> EDA artifacts -> preprocess -> train
baseline (Logistic Regression) and main model (Random Forest) -> evaluate
-> save model + metrics + figures.

Run:
    python train.py

Outputs:
    figures/*.png                  EDA and evaluation plots
    report/metrics_summary.json    headline metrics for both models
    report/classification_report.json   per-class precision/recall/F1
    models/rf_crop_model.joblib    trained Random Forest
    models/scaler.joblib           fitted StandardScaler (for the LR baseline)
    models/label_encoder.joblib    fitted LabelEncoder (crop name <-> class id)
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RANDOM_STATE = 42

for d in ["figures", "models", "report"]:
    (ROOT / d).mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    q1, q3 = df[FEATURES].quantile(0.25), df[FEATURES].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = {c: int(((df[c] < lo[c]) | (df[c] > hi[c])).sum()) for c in FEATURES}
    return {
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "n_classes": int(df["label"].nunique()),
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_balance": df["label"].value_counts().to_dict(),
        "iqr_outlier_counts": outliers,
    }


def make_eda_plots(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    df["label"].value_counts().plot(kind="bar", color="#3b7a57")
    plt.title("Class distribution (crops)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/class_balance.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#4c72b0")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/feature_distributions.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 6))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/correlation_heatmap.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, FEATURES):
        sns.boxplot(y=df[col], ax=ax, color="#dd8452")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/outlier_boxplots.png", dpi=130)
    plt.close()


def train_and_evaluate(df: pd.DataFrame):
    X, y = df[FEATURES], df["label"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Baseline: Logistic Regression (needs scaled features) ---
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average="macro")

    # --- Main model: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X, y_enc, cv=skf, scoring="accuracy")

    report = classification_report(
        y_test, rf_pred, target_names=le.classes_, output_dict=True
    )
    with open(ROOT / "report/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Random Forest — Confusion Matrix (test set)")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/confusion_matrix.png", dpi=130)
    plt.close()

    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, color="#3b7a57")
    plt.title("Random Forest feature importance")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/feature_importance.png", dpi=130)
    plt.close()

    X_test_df = X_test.copy()
    X_test_df["true"] = le.inverse_transform(y_test)
    X_test_df["pred"] = le.inverse_transform(rf_pred)
    errors = X_test_df[X_test_df["true"] != X_test_df["pred"]]

    joblib.dump(rf, ROOT / "models/rf_crop_model.joblib")
    joblib.dump(scaler, ROOT / "models/scaler.joblib")
    joblib.dump(le, ROOT / "models/label_encoder.joblib")

    metrics_summary = {
        "logistic_regression": {"test_accuracy": lr_acc, "macro_f1": lr_f1},
        "random_forest": {
            "test_accuracy": rf_acc,
            "macro_f1": rf_f1,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
        },
        "misclassified_count": int(len(errors)),
        "test_set_size": int(len(X_test_df)),
        "feature_importances": importances.to_dict(),
    }
    with open(ROOT / "report/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def main():
    df = load_data()
    dq = data_quality_report(df)
    with open(ROOT / "report/data_quality_report.json", "w") as f:
        json.dump(dq, f, indent=2)
    make_eda_plots(df)
    metrics = train_and_evaluate(df)

    print("=== Data quality ===")
    print(json.dumps(dq, indent=2))
    print("\n=== Model performance ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
