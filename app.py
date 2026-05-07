import streamlit as st
import pandas as pd
import plotly.express as px
import glob
 
# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title            = "Dallas Real Estate Tracker",
    page_icon             = "🏠",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)
 
st.markdown("""
<style>
  [data-testid="metric-container"] { background:#0f1e35; border-radius:8px; padding:12px; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)
 
 
# ─── Load & clean data ────────────────────────────────────────
@st.cache_data
def load_data():
    sale_files = glob.glob("data/dallas_listings_*.csv")
    rent_files = glob.glob("data/dallas_rentals_*.csv")
    dfs = []
 
    if sale_files:
        df_sale = pd.read_csv(max(sale_files))
        df_sale["listing_type"] = "For Sale"
        dfs.append(df_sale)
 
    if rent_files:
        df_rent = pd.read_csv(max(rent_files))
        df_rent["listing_type"] = "For Rent"
        dfs.append(df_rent)
 
    if not dfs:
        st.error("No CSV file found. Please run scraper.py first.")
        st.stop()
 
    df = pd.concat(dfs, ignore_index=True)
 
    df = df[df["price"] > 500].copy()
    p99 = df[df["listing_type"] == "For Sale"]["price"].quantile(0.99)
    df  = df[
        (df["listing_type"] == "For Rent") |
        (df["price"] <= p99)
    ].copy()
    df = df.dropna(subset=["sqft", "days_on_market"])
 
    replacements = {
        "singleFamily": "Single Family",
        "condo":        "Condo",
        "townhome":     "Townhome",
        "multiFamily":  "Multi-Family",
        "land":         "Land",
        "manufactured": "Manufactured"
    }
    df["property_type"] = df["property_type"].replace(replacements)
    df["has_price_cut"] = df["price_change"].notna() & (df["price_change"] < 0)
    df["price_M"]       = (df["price"] / 1_000).round(1)
    df["listing_url"]   = df["zpid"].apply(
        lambda x: f"https://www.zillow.com/homedetails/{int(x)}_zpid/"
        if pd.notna(x) else ""
    )
    return df
 
df_full = load_data()
 
 
# ─── Sidebar filters ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")
    st.markdown("---")
 
    listing_types    = sorted(df_full["listing_type"].unique().tolist())
    sel_listing_type = st.multiselect(
        "Listing type",
        options = listing_types,
        default = listing_types
    )
 
    all_types = sorted(df_full["property_type"].dropna().unique().tolist())
    sel_types = st.multiselect(
        "Property type",
        options = all_types,
        default = ["Single Family", "Condo", "Townhome"]
    )
 
    min_p = int(df_full["price"].min())
    max_p = int(df_full["price"].max())
    price_range = st.slider(
        "Price range (USD)",
        min_value = min_p,
        max_value = max_p,
        value     = (min_p, 3_000_000),
        step      = 50_000,
        format    = "$%,d"
    )
 
    max_dom = int(df_full["days_on_market"].max())
    dom_max = st.slider(
        "Max days on market",
        min_value = 1,
        max_value = max_dom,
        value     = max_dom
    )
 
    all_zips = sorted(df_full["zipcode"].dropna().unique().tolist())
    sel_zips = st.multiselect(
        "Zip codes (empty = all)",
        options = all_zips,
        default = []
    )
 
    st.markdown("---")
    st.markdown("📅 Data: May 2026")
    st.markdown("🔗 [GitHub](https://github.com/gnzme)")
 
 
# ─── Apply filters ────────────────────────────────────────────
df = df_full.copy()
 
if sel_listing_type:
    df = df[df["listing_type"].isin(sel_listing_type)]
 
if sel_types:
    df = df[df["property_type"].isin(sel_types)]
 
df = df[
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1]) &
    (df["days_on_market"] <= dom_max)
]
 
if sel_zips:
    df = df[df["zipcode"].isin(sel_zips)]
 
 
# ─── Header ──────────────────────────────────────────────────
st.markdown("# 🏠 Dallas Real Estate Market Tracker")
st.markdown(
    f"Showing **{len(df):,}** properties · "
    f"Live data via Zillow API · Dallas, TX"
)
st.markdown("---")
 
 
# ─── KPIs ────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
 
k1.metric("Total listings", f"{len(df):,}")
 
price_label = (
    "Median price / month"
    if sel_listing_type == ["For Rent"]
    else "Median price"
)
k2.metric(price_label, f"${df['price'].median()/1000:,.0f}k")
k3.metric("Median price / sqft",  f"${df['price_per_sqft'].median():,.0f}")
k4.metric("Median days on market", f"{df['days_on_market'].median():.0f} days")
k5.metric("With price cut",        f"{df['has_price_cut'].mean()*100:.1f}%")
 
st.markdown("---")
 
 
# ─── Map ─────────────────────────────────────────────────────
st.markdown("#### 🗺️ Property Map")
 
map_df = df[["latitude", "longitude", "price", "address",
             "zipcode", "bedrooms", "days_on_market"]].dropna(
    subset=["latitude", "longitude"]
).copy()
 
if len(map_df) > 0:
    fig_map = px.scatter_mapbox(
        map_df,
        lat                    = "latitude",
        lon                    = "longitude",
        color                  = "price",
        size                   = "price",
        size_max               = 12,
        zoom                   = 10,
        center                 = {"lat": 32.7767, "lon": -96.7970},
        color_continuous_scale = ["#00b4d8", "#f4a261", "#e07a5f"],
        mapbox_style           = "carto-positron",
        hover_name             = "address",
        hover_data             = {
            "price":          ":$,.0f",
            "zipcode":        True,
            "bedrooms":       True,
            "days_on_market": True,
            "latitude":       False,
            "longitude":      False
        },
        labels = {
            "price":          "Price",
            "zipcode":        "Zip",
            "bedrooms":       "Bedrooms",
            "days_on_market": "Days on market"
        },
        height = 500
    )
    fig_map.update_layout(
        margin             = dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar = dict(
            title      = "Price",
            tickprefix = "$",
            tickformat = ",.0f"
        ),
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No coordinates available to display the map.")
 
st.markdown("---")
 
 
# ─── Row 1: Price distribution + Price by zip ────────────────
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("#### Price Distribution")
    fig = px.histogram(
        df, x="price_M", nbins=50,
        labels={"price_M": "Price (thousands USD)", "count": "Properties"},
        color_discrete_sequence=["#00b4d8"]
    )
    fig.add_vline(
        x=df["price"].median()/1000,
        line_dash="dash", line_color="#e07a5f", line_width=2,
        annotation_text=f"Median ${df['price'].median()/1000:,.0f}k",
        annotation_position="top right"
    )
    fig.update_layout(
        showlegend    = False,
        margin        = dict(t=20, b=20, l=20, r=20),
        height        = 350,
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    fig.update_xaxes(tickprefix="$", ticksuffix="k")
    st.plotly_chart(fig, use_container_width=True)
 
with col2:
    st.markdown("#### Average Price by Zip Code (top 12)")
    zip_avg = (
        df.groupby("zipcode")["price"]
        .agg(["mean", "count"])
        .query("count >= 2")
        .sort_values("mean", ascending=False)
        .head(12)
        .reset_index()
    )
    if len(zip_avg) == 0:
        st.info("Not enough data to display this chart with the current filters.")
    else:
        zip_avg["label"]   = zip_avg.apply(
            lambda r: f"${r['mean']/1000:,.0f}k ({int(r['count'])})", axis=1
        )
        zip_avg["zipcode"] = zip_avg["zipcode"].astype(str)
        fig2 = px.bar(
            zip_avg, x="mean", y="zipcode",
            orientation="h",
            text="label",
            labels={"mean": "Average price (USD)", "zipcode": "Zip"},
            color="mean",
            color_continuous_scale=["#00b4d8", "#e07a5f"]
        )
        fig2.update_traces(textposition="outside", textfont_size=10)
        fig2.update_layout(
            coloraxis_showscale = False,
            margin              = dict(t=20, b=20, l=20, r=120),
            height              = 350,
            plot_bgcolor        = "rgba(0,0,0,0)",
            paper_bgcolor       = "rgba(0,0,0,0)",
            yaxis               = dict(type="category")
        )
        fig2.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig2, use_container_width=True)
 
 
# ─── Row 2: Scatter + Boxplot ─────────────────────────────────
col3, col4 = st.columns(2)
 
with col3:
    st.markdown("#### Price vs Square Footage")
    d_scatter = df[df["sqft"] < 8000].copy()
    fig3 = px.scatter(
        d_scatter,
        x="sqft", y="price_M",
        color="property_type",
        opacity=0.55,
        labels={
            "sqft":          "Square footage (sqft)",
            "price_M":       "Price (thousands USD)",
            "property_type": "Type"
        },
        color_discrete_sequence=["#00b4d8","#e07a5f","#3d9970","#f4a261","#7c6fcd"],
        hover_data=["zipcode", "bedrooms", "days_on_market"]
    )
    fig3.update_layout(
        margin        = dict(t=20, b=20, l=20, r=20),
        height        = 350,
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        legend        = dict(orientation="h", y=-0.2)
    )
    fig3.update_yaxes(tickprefix="$", ticksuffix="k")
    st.plotly_chart(fig3, use_container_width=True)
 
with col4:
    st.markdown("#### Days on Market by Property Type")
    type_order = (
        df.groupby("property_type")["days_on_market"]
        .median().sort_values().index.tolist()
    )
    fig4 = px.box(
        df[df["property_type"].isin(type_order)],
        x="property_type", y="days_on_market",
        category_orders={"property_type": type_order},
        labels={
            "property_type":  "Type",
            "days_on_market": "Days on market"
        },
        color="property_type",
        color_discrete_sequence=["#00b4d8","#3d9970","#e07a5f","#f4a261","#7c6fcd"]
    )
    fig4.update_layout(
        showlegend    = False,
        margin        = dict(t=20, b=20, l=20, r=20),
        height        = 350,
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        yaxis_range   = [0, df["days_on_market"].quantile(0.95) + 10]
    )
    st.plotly_chart(fig4, use_container_width=True)
 
 
# ─── Row 3: Price cuts + Table ───────────────────────────────
col5, col6 = st.columns([1, 1])
 
with col5:
    st.markdown("#### % with Price Cut by Zip Code")
    cut_rate = (
        df.groupby("zipcode")
        .filter(lambda x: len(x) >= 5)
        .groupby("zipcode")["has_price_cut"]
        .mean().mul(100)
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    cut_rate.columns = ["zipcode", "pct_cut"]
    avg_cut = cut_rate["pct_cut"].mean()
    cut_rate["zipcode"] = cut_rate["zipcode"].astype(str)
    
    fig5 = px.bar(
        cut_rate.sort_values("pct_cut"),
        x="pct_cut", y="zipcode",
        orientation="h",
        color="pct_cut",
        color_continuous_scale=["#00b4d8", "#e07a5f"],
        labels={"pct_cut": "% with price cut", "zipcode": "Zip"}
    )
    fig5.add_vline(
        x=avg_cut, line_dash="dash",
        line_color="#f4a261", line_width=1.5,
        annotation_text=f"Avg {avg_cut:.1f}%",
        annotation_position="top right"
    )
    fig5.update_layout(
        coloraxis_showscale = False,
        margin              = dict(t=20, b=20, l=20, r=20),
        height              = 400,
        plot_bgcolor        = "rgba(0,0,0,0)",
        paper_bgcolor       = "rgba(0,0,0,0)",
        yaxis               = dict(type="category")
    )
    fig5.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig5, use_container_width=True)
 
with col6:
    st.markdown("#### Listings Table")
    base_cols  = ["address", "zipcode", "price", "bedrooms",
                  "sqft", "days_on_market", "property_type", "has_price_cut"]
    extra_cols = [c for c in ["listing_url"] if c in df.columns]
    cols_show  = base_cols + extra_cols
    table      = df[cols_show].copy()
    table["price"] = table["price"].apply(lambda x: f"${x:,.0f}")
    table = table.rename(columns={
        "address":        "Address",
        "zipcode":        "Zip",
        "price":          "Price",
        "bedrooms":       "Beds",
        "sqft":           "Sqft",
        "days_on_market": "Days",
        "property_type":  "Type",
        "has_price_cut":  "Price cut",
        **({ "listing_url": "View on Zillow" } if "listing_url" in df.columns else {})
    })
    st.dataframe(
        table,
        column_config={
            **({ "View on Zillow": st.column_config.LinkColumn(
                "View on Zillow", display_text="🔗 Open"
            )} if "listing_url" in df.columns else {})
        },
        use_container_width=True,
        height=370
    )
 
 
# ─── Property Gallery ─────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📷 Property Gallery")
 
n_cards   = st.slider("Properties to display", 3, 12, 6, step=3)
gallery_df = df[
    df["photo_url"].notna() &
    df["listing_url"].notna() &
    (df["photo_url"] != "") &
    (df["listing_url"] != "")
].head(n_cards)
 
if len(gallery_df) == 0:
    st.info("No photos available for the current filters.")
else:
    cols_per_row = 3
    rows = [gallery_df.iloc[i:i+cols_per_row]
            for i in range(0, len(gallery_df), cols_per_row)]
 
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (_, prop) in zip(cols, row.iterrows()):
            with col:
                try:
                    st.image(prop["photo_url"], use_container_width=True)
                except:
                    st.markdown("📷 *Photo not available*")
 
                price_fmt = f"${prop['price']:,.0f}"
                beds  = int(prop["bedrooms"])  if pd.notna(prop["bedrooms"])  else "—"
                baths = int(prop["bathrooms"]) if pd.notna(prop["bathrooms"]) else "—"
                sqft  = int(prop["sqft"])      if pd.notna(prop["sqft"])      else "—"
                dom   = int(prop["days_on_market"])
 
                st.markdown(f"**{price_fmt}**")
                st.markdown(
                    f"🛏 {beds} beds · 🚿 {baths} baths · "
                    f"📐 {sqft} sqft · 📅 {dom} days"
                )
                st.markdown(f"📍 {prop['address']}, {prop['zipcode']}")
                st.markdown(
                    f'<a href="{prop["listing_url"]}" target="_blank">'
                    f'<button style="width:100%;padding:6px;'
                    f'background:#00b4d8;color:white;border:none;'
                    f'border-radius:4px;cursor:pointer;font-size:13px;">'
                    f'View on Zillow →</button></a>',
                    unsafe_allow_html=True
                )
                st.markdown("---")
 
 
# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Developed by **Gonzalo Medina C.** · Dallas, TX · "
    "[LinkedIn](https://www.linkedin.com/in/gonzalo-medina09/) · "
    "[GitHub](https://github.com/gnzme)"
)