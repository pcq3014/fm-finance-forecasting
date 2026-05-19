"""
Visualization: LEAF Regime Analysis + Model Comparison
=======================================================
Output: outputs/figures/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "data" / "results"
FIG_DIR = ROOT_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

MODEL_COLORS = {
    "ARIMA":     "#6c757d",
    "LSTM":      "#0d6efd",
    "TimesFM":   "#198754",
    "Chronos-2": "#dc3545",
    "Moirai-2":  "#fd7e14",
    "LEAF":      "#6f42c1",
    "Equal-W":   "#adb5bd",
}

REGIME_COLORS = {"Bull": "#198754", "Sideway": "#fd7e14", "Bear": "#dc3545"}


# ─── helpers ─────────────────────────────────────────────────────────────────

def load_summary(fname, ic_col="ic", da_col="dir_acc", rmse_col="rmse"):
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col0 = df.columns[0]
    return df[df[col0].astype(str) != "OVERALL"].copy()


def overall_metrics(fname, ic_col="ic", da_col="dir_acc", rmse_col="rmse"):
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col0 = df.columns[0]
    row = df[df[col0].astype(str) == "OVERALL"]
    if len(row) == 0:
        row = df.tail(1)
    return row.iloc[0]


# ─── Figure 1: Overall model comparison bar chart ────────────────────────────

def fig_model_comparison():
    data = [
        ("ARIMA",     overall_metrics("arima_summary.csv")),
        ("LSTM",      overall_metrics("lstm_mv_summary.csv")),
        ("TimesFM",   overall_metrics("timesfm_summary.csv")),
        ("Chronos-2", overall_metrics("chronos2_summary_v3.csv")),
        ("Moirai-2",  overall_metrics("moirai2_summary_v3.csv")),
    ]
    # Add LEAF if available
    leaf_path = RESULTS_DIR / "leaf_model_comparison.csv"
    if leaf_path.exists():
        ldf = pd.read_csv(leaf_path)
        for _, row in ldf.iterrows():
            name = str(row["model"])
            if name in ("LEAF", "Equal-Weight"):
                label = "LEAF" if "LEAF" in name else "Equal-W"
                data.append((label, row))

    models = [d[0] for d in data if d[1] is not None]
    metrics = {
        "RMSE": [float(d[1].get("rmse", np.nan)) for d in data if d[1] is not None],
        "MAE":  [float(d[1].get("mae",  np.nan)) for d in data if d[1] is not None],
        "IC":   [float(d[1].get("ic",   np.nan)) for d in data if d[1] is not None],
        "DA":   [float(d[1].get("dir_acc", d[1].get("da", np.nan))) for d in data if d[1] is not None],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Comparison — Overall Metrics (Test Set)", fontweight="bold")

    for ax, (metric, vals) in zip(axes, metrics.items()):
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        if metric == "IC":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        if metric == "DA":
            ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="Random")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    path = FIG_DIR / "fig1_model_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 2: IC heatmap per ticker × model ─────────────────────────────────

def fig_ic_heatmap():
    sources = [
        ("ARIMA",     "arima_summary.csv",        "ic"),
        ("LSTM",      "lstm_mv_summary.csv",      "ic"),
        ("TimesFM",   "timesfm_summary.csv",      "ic"),
        ("Chronos-2", "chronos2_summary_v3.csv",  "ic"),
        ("Moirai-2",  "moirai2_summary_v3.csv",   "ic"),
    ]

    frames = {}
    for model, fname, col in sources:
        df = load_summary(fname)
        if df is None:
            continue
        col0 = df.columns[0]
        frames[model] = df.set_index(col0)[col]

    if not frames:
        return

    heat = pd.DataFrame(frames).T
    heat = heat[[c for c in heat.columns if c != "OVERALL"]]

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        heat.astype(float), ax=ax,
        cmap="RdYlGn", center=0, vmin=-0.15, vmax=0.15,
        annot=True, fmt=".3f", linewidths=0.5,
        cbar_kws={"label": "IC"},
    )
    ax.set_title("Information Coefficient (IC) — per Ticker × Model", fontweight="bold")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Model")
    plt.tight_layout()
    path = FIG_DIR / "fig2_ic_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 3: LEAF regime weight distribution ───────────────────────────────

def fig_regime_weights():
    pred_path = RESULTS_DIR / "leaf_predictions.csv"
    if not pred_path.exists():
        print("SKIP fig3: leaf_predictions.csv not found")
        return

    df = pd.read_csv(pred_path)
    if "trend_regime" not in df.columns:
        print("SKIP fig3: no trend_regime column")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LEAF — Dynamic Weight Distribution by Regime", fontweight="bold")

    # Subplot 1: avg weights per regime (bar)
    ax = axes[0]
    regimes = ["Bull", "Sideway", "Bear"]
    w_cols = ["w_timesfm", "w_chronos2", "w_moirai2"]
    labels = ["TimesFM", "Chronos-2", "Moirai-2"]
    x = np.arange(len(regimes))
    width = 0.25

    for i, (col, label) in enumerate(zip(w_cols, labels)):
        means = [df[df["trend_regime"] == r][col].mean() for r in regimes]
        bars = ax.bar(x + i * width, means, width, label=label,
                      color=list(MODEL_COLORS.values())[i + 2], edgecolor="white")
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Average Weight")
    ax.set_title("Avg Model Weights per Regime")
    ax.legend()
    ax.set_ylim(0, 0.7)

    # Subplot 2: weight box plots per regime
    ax = axes[1]
    regime_order = ["Bull", "Sideway", "Bear"]
    plot_data = []
    for regime in regime_order:
        sub = df[df["trend_regime"] == regime]
        for col, label in zip(w_cols, labels):
            for v in sub[col].values:
                plot_data.append({"Regime": regime, "Model": label, "Weight": v})

    plot_df = pd.DataFrame(plot_data)
    if len(plot_df):
        palette = {l: list(MODEL_COLORS.values())[i + 2] for i, l in enumerate(labels)}
        sns.boxplot(data=plot_df, x="Regime", y="Weight", hue="Model",
                    order=regime_order, ax=ax, palette=palette, linewidth=0.8)
        ax.set_title("Weight Distribution per Regime")
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "fig3_regime_weights.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 4: Weight timeline (rolling mean) ────────────────────────────────

def fig_weight_timeline():
    pred_path = RESULTS_DIR / "leaf_predictions.csv"
    if not pred_path.exists():
        print("SKIP fig4: leaf_predictions.csv not found")
        return

    df = pd.read_csv(pred_path, parse_dates=["date"])
    if "w_timesfm" not in df.columns:
        return

    # Average across tickers per date
    daily = df.groupby("date")[["w_timesfm", "w_chronos2", "w_moirai2", "trend_regime"]].agg(
        w_timesfm=("w_timesfm", "mean"),
        w_chronos2=("w_chronos2", "mean"),
        w_moirai2=("w_moirai2", "mean"),
        trend_regime=("trend_regime", lambda x: x.mode()[0]),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("LEAF — Weight Dynamics Over Time", fontweight="bold")

    roll = 10
    for col, label, color in zip(
        ["w_timesfm", "w_chronos2", "w_moirai2"],
        ["TimesFM", "Chronos-2", "Moirai-2"],
        [MODEL_COLORS["TimesFM"], MODEL_COLORS["Chronos-2"], MODEL_COLORS["Moirai-2"]],
    ):
        ax1.plot(daily["date"], daily[col].rolling(roll).mean(),
                 label=label, color=color, linewidth=1.5)

    ax1.set_ylabel("Weight (rolling 10-day avg)")
    ax1.legend(loc="upper right")
    ax1.axhline(1/3, color="gray", linestyle="--", linewidth=0.8, label="Equal weight")
    ax1.set_ylim(0, 1)

    # Regime background
    for _, row in daily.iterrows():
        c = REGIME_COLORS.get(row["trend_regime"], "#ffffff")
        ax2.axvspan(row["date"], row["date"] + pd.Timedelta(days=1),
                    alpha=0.4, color=c, linewidth=0)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k, alpha=0.6)
                  for k, v in REGIME_COLORS.items()]
    ax2.legend(handles=legend_els, loc="upper right", fontsize=9)
    ax2.set_ylabel("Regime")
    ax2.set_yticks([])
    ax2.set_xlabel("Date")

    plt.tight_layout()
    path = FIG_DIR / "fig4_weight_timeline.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 5: IC per regime across models ───────────────────────────────────

def fig_ic_per_regime():
    pred_path = RESULTS_DIR / "leaf_predictions.csv"
    if not pred_path.exists():
        print("SKIP fig5: leaf_predictions.csv not found")
        return

    df = pd.read_csv(pred_path)
    required = {"y_true", "leaf_pred", "equal_pred",
                "w_timesfm", "w_chronos2", "w_moirai2", "trend_regime"}
    if not required.issubset(df.columns):
        print(f"SKIP fig5: missing columns {required - set(df.columns)}")
        return

    regimes = ["Bull", "Sideway", "Bear"]
    models = ["LEAF", "Equal-W", "TimesFM", "Chronos-2", "Moirai-2"]
    pred_cols = ["leaf_pred", "equal_pred"]

    # need base FM predictions merged
    fm_files = {
        "TimesFM":   ("timesfm_predictions.csv",     "y_pred"),
        "Chronos-2": ("chronos2_predictions_v3.csv", "y_pred"),
        "Moirai-2":  ("moirai2_predictions_v3.csv",  "y_pred"),
    }
    for name, (fname, col) in fm_files.items():
        p = RESULTS_DIR / fname
        if p.exists():
            fdf = pd.read_csv(p)[["date", "ticker", col]].rename(columns={col: f"pred_{name}"})
            df = df.merge(fdf, on=["date", "ticker"], how="left")

    ic_data = {m: [] for m in models}
    n_data = []

    for regime in regimes:
        sub = df[df["trend_regime"] == regime]
        n_data.append(len(sub))
        y = sub["y_true"].values
        for model, pred_col in zip(
            ["LEAF", "Equal-W"],
            ["leaf_pred", "equal_pred"],
        ):
            ic = np.corrcoef(y, sub[pred_col].values)[0, 1] if sub[pred_col].std() > 0 else 0
            ic_data[model].append(ic)
        for name in ["TimesFM", "Chronos-2", "Moirai-2"]:
            col = f"pred_{name}"
            if col in sub.columns:
                ic = np.corrcoef(y, sub[col].fillna(0).values)[0, 1] if sub[col].std() > 0 else 0
                ic_data[name].append(ic)
            else:
                ic_data[name].append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(regimes))
    width = 0.15

    for i, model in enumerate(models):
        vals = ic_data[model]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=model, color=MODEL_COLORS.get(model, "#888"),
                      edgecolor="white")

    ax.set_xticks(x)
    labels_with_n = [f"{r}\n(n={n})" for r, n in zip(regimes, n_data)]
    ax.set_xticklabels(labels_with_n)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("IC")
    ax.set_title("Information Coefficient by Market Regime", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "fig5_ic_per_regime.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...\n")
    fig_model_comparison()
    fig_ic_heatmap()
    fig_regime_weights()
    fig_weight_timeline()
    fig_ic_per_regime()
    print(f"\nAll figures saved to: {FIG_DIR}")
