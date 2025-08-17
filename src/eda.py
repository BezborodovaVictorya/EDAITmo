import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from .config import PATHS

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

def _start_report_folder() -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = PATHS.reports_dir / f"EDA_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_basic_stats(df: pd.DataFrame, out_dir: Path):
    desc = df.describe(include="all").transpose()
    desc.to_csv(out_dir / "describe.csv")
    df.isna().sum().sort_values(ascending=False).to_csv(out_dir / "missing_values.csv")

def class_balance(df: pd.DataFrame, out_dir: Path):
    plt.figure()
    sns.countplot(x="is_fraud", data=df)
    plt.title("Fraud vs Non-Fraud Count")
    plt.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close()
    fraud_rate = df["is_fraud"].mean()
    with open(out_dir / "fraud_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"Fraud rate: {fraud_rate:.4%}\nImbalance ratio: {(1-fraud_rate)/fraud_rate:.2f}\n")

def plot_top_categories(df: pd.DataFrame, col: str, out_dir: Path, top_n: int = 10):
    if col not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    top = df[col].value_counts().head(top_n)
    sns.barplot(x=top.index.astype(str), y=top.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} {col} by count")
    plt.tight_layout()
    plt.savefig(out_dir / f"top_{col}.png", dpi=150)
    plt.close()

    fraud_rate = df.groupby(col)["is_fraud"].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=fraud_rate.index.astype(str), y=fraud_rate.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} {col} by fraud rate")
    plt.tight_layout()
    plt.savefig(out_dir / f"fraud_rate_top_{col}.png", dpi=150)
    plt.close()

def correlation_heatmap(df: pd.DataFrame, out_dir: Path):
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

def temporal_patterns(df: pd.DataFrame, out_dir: Path):
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "dayofweek" not in df.columns:
        df["dayofweek"] = df["timestamp"].dt.dayofweek

    for col, label in [("hour", "Hour of Day"), ("dayofweek", "Day of Week (0=Mon)")]:
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=col, y="is_fraud", data=df, estimator="mean")
        plt.title(f"Fraud rate by {label}")
        plt.savefig(out_dir / f"fraud_by_{col}.png", dpi=150)
        plt.close()

def quick_feature_importance(df: pd.DataFrame, out_dir: Path):
    # Черновой LightGBM для оценки важности признаков
    num_df = df.select_dtypes(include=[np.number])
    if "is_fraud" not in num_df.columns:
        return
    X = num_df.drop(columns=["is_fraud"]).fillna(0)
    y = num_df["is_fraud"]
    if y.nunique() != 2:
        return
    model = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp.values[:20], y=imp.index[:20])
    plt.title("Top 20 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()
    imp.to_csv(out_dir / "feature_importance.csv")

def generate_summary(df: pd.DataFrame, out_dir: Path):
    fraud_rate = df["is_fraud"].mean()
    summary = [
        f"# EDA Summary",
        f"Дата анализа: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Всего транзакций: {len(df):,}",
        f"- Доля мошеннических: {fraud_rate:.2%}",
        f"- Кол-во стран: {df['country'].nunique()}",
        f"- Кол-во валют: {df['currency'].nunique()}",
        f"- Топ вендор категорий: {', '.join(df['vendor_category'].value_counts().head(5).index.astype(str))}",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

def run_eda(df: pd.DataFrame) -> Path:
    out_dir = _start_report_folder()
    logger.info(f"Сохраняю EDA отчёты в {out_dir}")

    save_basic_stats(df, out_dir)
    class_balance(df, out_dir)
    for col in ["vendor_category", "country", "currency", "device", "channel"]:
        plot_top_categories(df, col, out_dir)

    correlation_heatmap(df, out_dir)
    temporal_patterns(df, out_dir)
    quick_feature_importance(df, out_dir)
    generate_summary(df, out_dir)

    logger.info("EDA завершён.")
    return out_dir
