import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─── Configuración ───────────────────────────────────────────
API_KEY  = os.getenv("RAPIDAPI_KEY")
BASE_URL = "https://private-zillow.p.rapidapi.com/search/byaddress"
HEADERS  = {
    "x-rapidapi-key":  API_KEY,
    "x-rapidapi-host": "private-zillow.p.rapidapi.com"
}
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Fetch listings ──────────────────────────────────────────
def fetch_listings(location="Dallas, TX", pages=2):
    """
    Obtiene hasta 1000 listings reales de Dallas.
    500 por página x 2 páginas = 2 requests del cupo free.
    """
    all_results = []

    for page in range(1, pages + 1):
        params = {
            "location":      location,
            "listingStatus": "For_Sale",
            "pageSize":      500,
            "page":          page,
            "sortOrder":     "Homes_for_you"
        }

        try:
            print(f"  Fetching página {page}/{pages}...", end=" ")
            response = requests.get(
                BASE_URL, headers=HEADERS,
                params=params, timeout=15
            )

            if response.status_code != 200:
                print(f"✗ Error {response.status_code}: {response.text[:200]}")
                break

            data    = response.json()
            results = data.get("searchResults", [])
            all_results.extend(results)
            total   = data.get("resultsCount", {}).get("totalMatchingCount", 0)
            print(f"✓ {len(results)} listings  (total disponible: {total:,})")

            time.sleep(2)  # respeta rate limit

        except requests.exceptions.Timeout:
            print("✗ Timeout — reintentando en 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"✗ Error: {e}")
            break

    return all_results


# ─── Limpiar y estructurar datos ─────────────────────────────
def clean_listings(raw_results):
    """
    Extrae los campos relevantes de cada property.
    Estructura real: result["property"]["campo"]
    """
    cleaned = []

    for result in raw_results:
        if result.get("resultType") != "property":
            continue

        p       = result.get("property", {})
        addr    = p.get("address", {})
        loc     = p.get("location", {})
        price   = p.get("price", {})
        est     = p.get("estimates", {})
        tax     = p.get("taxAssessment", {})
        lot     = p.get("lotSizeWithUnit", {})
        photo   = p.get("media", {}).get("propertyPhotoLinks", {})
        hdp     = p.get("hdpView", {})

        price_val  = price.get("value")
        sqft       = p.get("livingArea")
        price_psf  = price.get("pricePerSquareFoot")

        # calcular price/sqft si no viene
        if not price_psf and price_val and sqft and sqft > 0:
            price_psf = round(price_val / sqft, 2)

        cleaned.append({
            # Identificación
            "zpid":              p.get("zpid"),
            # Ubicación
            "address":           addr.get("streetAddress", "N/A"),
            "zipcode":           addr.get("zipcode", "N/A"),
            "city":              addr.get("city", "Dallas"),
            "state":             addr.get("state", "TX"),
            "latitude":          loc.get("latitude"),
            "longitude":         loc.get("longitude"),
            # Precio
            "price":             price_val,
            "price_per_sqft":    price_psf,
            "price_change":      price.get("priceChange"),
            "zestimate":         est.get("zestimate"),
            "rent_zestimate":    est.get("rentZestimate"),
            # Características
            "bedrooms":          p.get("bedrooms"),
            "bathrooms":         p.get("bathrooms"),
            "sqft":              sqft,
            "year_built":        p.get("yearBuilt"),
            "lot_size_acres":    lot.get("lotSize"),
            "property_type":     p.get("propertyType", "N/A"),
            # Mercado
            "days_on_market":    p.get("daysOnZillow"),
            "tax_assessed":      tax.get("taxAssessedValue"),
            "tax_year":          tax.get("taxAssessmentYear"),
            # Media
            "photo_url":         photo.get("mediumSizeLink", ""),
            # Meta
            "fetched_at":        datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    return cleaned


# ─── Guardar CSV ─────────────────────────────────────────────
def save_csv(df):
    date_str = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(OUTPUT_DIR, f"dallas_listings_{date_str}.csv")
    df.to_csv(filepath, index=False)
    print(f"\n  ✓ Guardado en: {filepath}")
    return filepath


# ─── Resumen ─────────────────────────────────────────────────
def print_summary(df):
    d = df.dropna(subset=["price"])

    print("\n" + "="*52)
    print("   DALLAS REAL ESTATE — RESUMEN")
    print("="*52)
    print(f"  Total listings scrapeados  : {len(df):,}")
    print(f"  Con precio válido          : {len(d):,}")
    print(f"  Precio promedio            : ${d['price'].mean():,.0f}")
    print(f"  Precio mediano             : ${d['price'].median():,.0f}")
    print(f"  Precio mínimo              : ${d['price'].min():,.0f}")
    print(f"  Precio máximo              : ${d['price'].max():,.0f}")
    print(f"  Precio/sqft promedio       : ${d['price_per_sqft'].mean():,.0f}")
    print(f"  Días en mercado (promedio) : {d['days_on_market'].mean():.0f}")

    print(f"\n  Top 5 zip codes por cantidad:")
    top_zips = df["zipcode"].value_counts().head(5)
    for zip_code, count in top_zips.items():
        avg = df[df["zipcode"] == zip_code]["price"].mean()
        print(f"    {zip_code}  →  {count:>3} listings  |  avg ${avg:,.0f}")

    print(f"\n  Distribución por tipo de propiedad:")
    for ptype, count in df["property_type"].value_counts().items():
        print(f"    {ptype:<20} {count:>4}")

    print(f"\n  Listings con rebaja de precio:")
    cuts = df[df["price_change"] < 0]
    print(f"    {len(cuts)} propiedades ({len(cuts)/len(df)*100:.1f}%)")
    if len(cuts) > 0:
        print(f"    Rebaja promedio: ${cuts['price_change'].mean():,.0f}")
    print("="*52)


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Dallas Real Estate Tracker — Semana 1")
    print(" " + "─"*40)

    print("\n[1/3] Fetching listings reales de Zillow...")
    raw = fetch_listings(location="Dallas, TX", pages=2)
    print(f"\n  Total raw results: {len(raw)}")

    print("\n[2/3] Limpiando y estructurando datos...")
    cleaned = clean_listings(raw)
    df = pd.DataFrame(cleaned)
    print(f"  DataFrame: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    print("\n[3/3] Guardando CSV...")
    save_csv(df)

    print_summary(df)

    print("\n  Vista previa:")
    cols = ["address", "zipcode", "price", "bedrooms", "sqft", "days_on_market"]
    print(df[cols].head(8).to_string(index=False))

    print("\n✓ Semana 1 completada — datos reales de Dallas listos.\n")