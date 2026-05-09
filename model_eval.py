import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import glob
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CYAN   = "#00b4d8"
CORAL  = "#e07a5f"
GREEN  = "#3d9970"
AMBER  = "#f4a261"


# ─── Cargar modelo y datos ────────────────────────────────────
def load_everything():
    model    = joblib.load("model/price_model.joblib")
    encoders = joblib.load("model/encoders.joblib")
    features = joblib.load("model/features.joblib")
    stats    = joblib.load("model/model_stats.joblib")

    files = glob.glob("data/dallas_listings_*.csv")
    df    = pd.read_csv(max(files))
    df    = df[df["price"] > 10_000].copy()
    p99   = df["price"].quantile(0.99)
    df    = df[df["price"] <= p99].copy()
    df    = df.dropna(subset=["sqft", "bedrooms", "bathrooms",
                               "year_built", "property_type", "zipcode"])
    df["zipcode"]       = df["zipcode"].astype(str)
    df["lot_size_acres"] = df["lot_size_acres"].fillna(
                           df["lot_size_acres"].median())

    # Encodear
    for col in ["zipcode", "property_type"]:
        le       = encoders[col]
        known    = set(le.classes_)
        df[col]  = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col]  = le.transform(df[col])

    return model, encoders, features, stats, df


# ─── Gráfico 1 — Predicción vs Real ──────────────────────────
def chart_pred_vs_real(model, df, features):
    X = df[features]
    y = df["price"]
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y / 1000, y_pred / 1000,
               alpha=0.35, s=18, color=CYAN, edgecolors="none")

    # Línea perfecta
    max_val = max(y.max(), y_pred.max()) / 1000
    ax.plot([0, max_val], [0, max_val],
            color=CORAL, lw=1.8, linestyle="--", label="Perfect prediction")

    ax.set_xlabel("Actual price (thousands USD)")
    ax.set_ylabel("Predicted price (thousands USD)")
    ax.set_title("Predicted vs Actual Price")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}k"))
    ax.legend(fontsize=10)

    path = os.path.join(OUTPUT_DIR, "06_pred_vs_real.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {path}")
    plt.close()


# ─── Gráfico 2 — Distribución de errores ─────────────────────
def chart_error_distribution(model, df, features):
    X      = df[features]
    y      = df["price"]
    y_pred = model.predict(X)
    errors = (y_pred - y) / 1000  # error en miles

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(errors, bins=60, color=CYAN,
            edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.axvline(0,           color=CORAL, lw=2,
               linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color=AMBER, lw=1.5,
               linestyle="--",
               label=f"Mean error: ${errors.mean():,.0f}k")

    ax.set_xlabel("Prediction error (thousands USD)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Prediction Errors")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}k"))
    ax.legend(fontsize=10)

    path = os.path.join(OUTPUT_DIR, "07_error_distribution.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {path}")
    plt.close()


# ─── Gráfico 3 — Feature importance ──────────────────────────
def chart_feature_importance(model, features):
    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values()

    labels = {
        "sqft":           "Square footage",
        "zipcode":        "Zip code",
        "lot_size_acres": "Lot size (acres)",
        "year_built":     "Year built",
        "bedrooms":       "Bedrooms",
        "bathrooms":      "Bathrooms",
        "property_type":  "Property type"
    }
    importance.index = [labels.get(f, f) for f in importance.index]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [CORAL if v == importance.max() else CYAN
              for v in importance.values]
    bars   = ax.barh(importance.index, importance.values,
                     color=colors, height=0.6)

    for bar, val in zip(bars, importance.values):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — What Drives Dallas Home Prices?")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, importance.max() * 1.2)

    path = os.path.join(OUTPUT_DIR, "08_feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {path}")
    plt.close()


# ─── Gráfico 4 — Error por rango de precio ───────────────────
def chart_error_by_price_range(model, df, features):
    X      = df[features].copy()
    y      = df["price"].copy()
    y_pred = model.predict(X)

    results = pd.DataFrame({
        "actual":    y.values,
        "predicted": y_pred,
        "pct_error": abs(y_pred - y.values) / y.values * 100
    })

    bins   = [0, 300_000, 600_000, 1_000_000, 2_000_000, 99_000_000]
    labels = ["<$300k", "$300k–$600k", "$600k–$1M", "$1M–$2M", ">$2M"]
    results["range"] = pd.cut(results["actual"], bins=bins, labels=labels)

    avg_error = results.groupby("range", observed=True)["pct_error"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [GREEN if v < 20 else AMBER if v < 35 else CORAL
              for v in avg_error.values]
    bars   = ax.bar(avg_error.index, avg_error.values,
                    color=colors, width=0.55)

    for bar, val in zip(bars, avg_error.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11,
                fontweight="500")

    ax.axhline(20, color=GREEN, linestyle="--",
               lw=1.2, alpha=0.6, label="20% error threshold")
    ax.set_xlabel("Price range")
    ax.set_ylabel("Average % error")
    ax.set_title("Model Accuracy by Price Range")
    ax.legend(fontsize=10)
    ax.set_ylim(0, avg_error.max() * 1.3)

    path = os.path.join(OUTPUT_DIR, "09_error_by_price_range.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {path}")
    plt.close()


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Dallas House Price Predictor — Model Evaluation")
    print(" " + "─"*48)

    print("\n[1/5] Loading model and data...")
    model, encoders, features, stats, df = load_everything()
    print(f"  R²: {stats['r2']:.3f} | MAE: ${stats['mae']:,.0f} | "
          f"RMSE: ${stats['rmse']:,.0f}")

    print("\n[2/5] Chart 1 — Predicted vs Actual...")
    chart_pred_vs_real(model, df, features)

    print("\n[3/5] Chart 2 — Error distribution...")
    chart_error_distribution(model, df, features)

    print("\n[4/5] Chart 3 — Feature importance...")
    chart_feature_importance(model, features)

    print("\n[5/5] Chart 4 — Error by price range...")
    chart_error_by_price_range(model, df, features)

    print("\n✓ Week 2 complete. 4 charts saved to outputs/\n")