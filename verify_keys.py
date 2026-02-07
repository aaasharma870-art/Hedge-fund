import os
import requests
from dotenv import load_dotenv

load_dotenv()

poly_key = os.environ.get("POLYGON_API_KEY")
fmp_key = os.environ.get("FMP_API_KEY")

print(f"Loaded Polygon Key: {poly_key[:4]}...{poly_key[-4:] if poly_key else ''}")
print(f"Loaded FMP Key: {fmp_key[:4]}...{fmp_key[-4:] if fmp_key else ''}")

# Test Polygon
if poly_key:
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-05?apiKey={poly_key}"
    r = requests.get(url)
    print(f"Polygon Test: Status {r.status_code}")
    if r.status_code != 200:
        print(f"Polygon Response: {r.text}")
else:
    print("re: Polygon Key missing")

# Test FMP
if fmp_key:
    url = f"https://financialmodelingprep.com/api/v3/profile/AAPL?apikey={fmp_key}"
    r = requests.get(url)
    print(f"FMP Test: Status {r.status_code}")
    if r.status_code != 200:
        print(f"FMP Response: {r.text}")
else:
    print("re: FMP Key missing")
