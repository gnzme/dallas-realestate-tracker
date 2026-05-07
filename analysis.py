import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import glob

# ─── Configuración visual ────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
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
PURPLE = "#7c6fcd"


# ─── 1. Cargar datos ─────────────────────────────────────────
def load_data():
    files = glob.glob("data/dallas_listings_*.csv")
    if not files:
        raise FileNotFoundError("No se encontró el CSV. Corre scraper.py primero.")
    latest = max(files)
    df = pd.read_csv(latest)
    print(f"  Archivo cargado: {latest}")
    print(f"  Filas originales: {len(df):,}")
    return df


# ─── 2. Limpiar datos ────────────────────────────────────────
def clean_data(df):
    original = len(df)

    # Remover precios inválidos
    df = df[df["price"] > 10_000].copy()

    # Remover outliers extremos (sobre percentil 99)
    p99 = df["price"].quantile(0.99)
    df  = df[df["price"] <= p99].copy()

    # Remover filas sin sqft o días en mercado
    df = df.dropna(subset=["sqft", "days_on_market"])

    # Limpiar property_type
    df["property_type"] = df["property_type"].str.replace(
        "singleFamily", "Single Family"
    ).str.replace(
        "condo", "Condo"
    ).str.replace(
        "townhome", "Townhome"
    ).str.replace(
        "multiFamily", "Multi-Family"
    ).str.replace(
        "land", "Land"
    ).str.replace(
        "manufactured", "Manufactured"
    )

    # Columna auxiliar: tiene rebaja?
    df["has_price_cut"] = df["price_change"].notna() & (df["price_change"] < 0)

    removed = original - len(df)
    print(f"  Filas eliminadas (outliers/nulos): {removed}")
    print(f"  Filas limpias para análisis: {len(df):,}")
    return df


# ─── 3. Gráfico 1 — Distribución de precios ──────────────────
def chart_price_distribution(df):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(df["price"] / 1_000, bins=50, color=CYAN,
            edgecolor="white", linewidth=0.4, alpha=0.9)

    median = df["price"].median() / 1_000
    mean   = df["price"].mean()   / 1_000
    ax.axvline(median, color=CORAL,  linestyle="--", lw=1.8,
               label=f"Mediana: ${median:,.0f}k")
    ax.axvline(mean,   color=AMBER,  linestyle="--", lw=1.8,
               label=f"Promedio: ${mean:,.0f}k")

    ax.set_title("Distribución de precios — Dallas TX (For Sale)")
    ax.set_xlabel("Precio (miles USD)")
    ax.set_ylabel("Cantidad de propiedades")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}k"))
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_price_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Guardado: {path}")
    plt.close()


# ─── 4. Gráfico 2 — Precio promedio por zip code ─────────────
def chart_price_by_zipcode(df):
    top_zips = (
        df.groupby("zipcode")["price"]
        .agg(["mean", "count"])
        .query("count >= 10")          # solo zips con 10+ listings
        .sort_values("mean", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    colors  = [CYAN if v < df["price"].median() else CORAL
               for v in top_zips["mean"]]
    bars    = ax.barh(top_zips.index.astype(str),
                      top_zips["mean"] / 1_000,
                      color=colors, height=0.65)

    # Etiquetas de valor
    for bar, count in zip(bars, top_zips["count"]):
        w = bar.get_width()
        ax.text(w + 15, bar.get_y() + bar.get_height() / 2,
                f"${w:,.0f}k  ({count})",
                va="center", ha="left", fontsize=9,
                color="#555")

    ax.axvline(df["price"].median() / 1_000, color=AMBER,
               linestyle="--", lw=1.5,
               label=f"Mediana general: ${df['price'].median()/1000:,.0f}k")
    ax.set_title("Precio promedio por zip code — Top 15 (min. 10 listings)")
    ax.set_xlabel("Precio promedio (miles USD)")
    ax.set_ylabel("Zip code")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}k"))
    ax.legend(fontsize=10)
    ax.set_xlim(0, top_zips["mean"].max() / 1_000 * 1.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_price_by_zipcode.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Guardado: {path}")
    plt.close()


# ─── 5. Gráfico 3 — Precio vs sqft ───────────────────────────
def chart_price_vs_sqft(df):
    d = df[df["sqft"] < 8_000].copy()    # excluir mansiones extremas

    type_colors = {
        "Single Family": CYAN,
        "Condo":         CORAL,
        "Townhome":      GREEN,
        "Multi-Family":  AMBER,
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for ptype, color in type_colors.items():
        subset = d[d["property_type"] == ptype]
        if len(subset) == 0:
            continue
        ax.scatter(subset["sqft"], subset["price"] / 1_000,
                   color=color, alpha=0.5, s=22,
                   label=f"{ptype} ({len(subset)})", edgecolors="none")

    ax.set_title("Precio vs Superficie — Dallas TX")
    ax.set_xlabel("Superficie (sqft)")
    ax.set_ylabel("Precio (miles USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:,.0f}k"))
    ax.legend(fontsize=10, markerscale=1.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_price_vs_sqft.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Guardado: {path}")
    plt.close()


# ─── 6. Gráfico 4 — Días en mercado por tipo ─────────────────
def chart_days_by_type(df):
    types_order = (
        df.groupby("property_type")["days_on_market"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df[df["property_type"].isin(types_order)],
        x="property_type", y="days_on_market",
        order=types_order,
        palette=[CYAN, GREEN, CORAL, AMBER, PURPLE],
        width=0.5, linewidth=1.2,
        flierprops=dict(marker="o", markersize=3, alpha=0.3),
        ax=ax
    )

    ax.set_title("Días en mercado por tipo de propiedad")
    ax.set_xlabel("")
    ax.set_ylabel("Días en mercado")
    ax.set_ylim(0, df["days_on_market"].quantile(0.95) + 10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_days_by_type.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Guardado: {path}")
    plt.close()


# ─── 7. Gráfico 5 — Rebaja de precio por zip code ────────────
def chart_price_cuts_by_zip(df):
    top_zips = (
        df.groupby("zipcode")
        .filter(lambda x: len(x) >= 10)
        ["zipcode"].unique()
    )
    d = df[df["zipcode"].isin(top_zips)].copy()

    cut_rate = (
        d.groupby("zipcode")["has_price_cut"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [CORAL if v >= 40 else CYAN for v in cut_rate.values]
    ax.barh(cut_rate.index.astype(str)[::-1],
            cut_rate.values[::-1],
            color=colors[::-1], height=0.65)

    ax.axvline(cut_rate.mean(), color=AMBER, linestyle="--",
               lw=1.5, label=f"Promedio: {cut_rate.mean():.1f}%")
    ax.set_title("% de propiedades con rebaja de precio — Top 15 zip codes")
    ax.set_xlabel("% con rebaja")
    ax.set_ylabel("Zip code")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_price_cuts_by_zip.png")
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Guardado: {path}")
    plt.close()


# ─── 8. Resumen ejecutivo ────────────────────────────────────
def print_insights(df):
    fastest = (df.groupby("property_type")["days_on_market"]
               .median().idxmin())
    most_cuts = (df.groupby("zipcode")["has_price_cut"]
                 .mean().idxmax())
    priciest_zip = (df.groupby("zipcode")["price"]
                    .mean().idxmax())

    print("\n" + "="*54)
    print("  INSIGHTS CLAVE — DALLAS REAL ESTATE")
    print("="*54)
    print(f"  Precio mediano          : ${df['price'].median():,.0f}")
    print(f"  Precio/sqft mediano     : ${df['price_per_sqft'].median():,.0f}")
    print(f"  Días en mercado mediano : {df['days_on_market'].median():.0f} días")
    print(f"  Tipo más rápido en vender: {fastest}")
    print(f"  Zip con más rebajas     : {most_cuts}")
    print(f"  Zip más caro (promedio) : {priciest_zip}")
    print(f"  Propiedades con rebaja  : {df['has_price_cut'].mean()*100:.1f}%")
    print("="*54)


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Dallas Real Estate — Semana 2: Análisis")
    print(" " + "─"*42)

    print("\n[1/7] Cargando datos...")
    df = load_data()

    print("\n[2/7] Limpiando datos...")
    df = clean_data(df)

    print("\n[3/7] Gráfico 1 — Distribución de precios...")
    chart_price_distribution(df)

    print("\n[4/7] Gráfico 2 — Precio por zip code...")
    chart_price_by_zipcode(df)

    print("\n[5/7] Gráfico 3 — Precio vs sqft...")
    chart_price_vs_sqft(df)

    print("\n[6/7] Gráfico 4 — Días en mercado por tipo...")
    chart_days_by_type(df)

    print("\n[7/7] Gráfico 5 — Rebajas por zip code...")
    chart_price_cuts_by_zip(df)

    print_insights(df)

    print("\n✓ Semana 2 completada.")
    print(f"  5 gráficos guardados en la carpeta outputs/\n")