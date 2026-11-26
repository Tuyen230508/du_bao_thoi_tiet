import requests
import pandas as pd

url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=21.0285&longitude=105.8542"
    "&start_date=2024-01-01&end_date=2025-12-31"
    "&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
)
data = requests.get(url, timeout=20).json()["hourly"]
df = pd.DataFrame(data)
df.to_csv("hanoi_weather.csv", index=False)
print("Đã lưu xong:", df.shape)