import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://private-zillow.p.rapidapi.com/search/byaddress"

params = {
    "location": "Dallas, TX",
    "listingStatus": "For_Sale",
    "pageSize": 5  # solo 5 para el test
}

headers = {
    "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
    "x-rapidapi-host": "private-zillow.p.rapidapi.com"
}

print("Conectando a private-zillow API...")
response = requests.get(url, headers=headers, params=params)
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    total    = data.get("resultsCount", {}).get("totalMatchingCount", 0)
    results  = data.get("searchResults", [])
    print(f"✓ API funcionando correctamente")
    print(f"  Total listings en Dallas : {total:,}")
    print(f"  Listings en este test    : {len(results)}")
    if results:
        p = results[0].get("property", {})
        print(f"\n  Ejemplo:")
        print(f"  Dirección  : {p.get('address',{}).get('streetAddress','N/A')}")
        print(f"  Zip Code   : {p.get('address',{}).get('zipcode','N/A')}")
        print(f"  Precio     : ${p.get('price',{}).get('value',0):,}")
        print(f"  Dormitorios: {p.get('bedrooms','N/A')}")
        print(f"  Sqft       : {p.get('livingArea','N/A')}")
        print(f"  Días en mkt: {p.get('daysOnZillow','N/A')}")
        print(f"  Zestimate  : ${p.get('estimates',{}).get('zestimate',0):,}")
else:
    print(f"✗ Error: {response.status_code}")
    print(response.text)