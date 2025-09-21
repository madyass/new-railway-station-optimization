import json
import folium
import random

# --- JSON dosyalarını oku ---
with open("/home/saydam/github_projects/metro_optimization/data/ordered_stations.json", "r", encoding="utf-8") as f:
    line_to_stations = json.load(f)

with open("/home/saydam/github_projects/metro_optimization/data/station_id_to_coords.json", "r", encoding="utf-8") as f:
    station_id_to_coords = json.load(f)

# --- Başlangıç haritası (ortalama koordinatlara zoom) ---
all_coords = list(station_id_to_coords.values())
avg_lat = sum(c[0] for c in all_coords) / len(all_coords)
avg_lon = sum(c[1] for c in all_coords) / len(all_coords)

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles="CartoDB positron")

# --- Hatları çiz ---
for line, stations in line_to_stations.items():
    coords = [station_id_to_coords[str(s)] for s in stations]

    # Renkleri random seçelim
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Hat çizgisi
    folium.PolyLine(
        locations=coords,
        color=color,
        weight=4,
        opacity=0.8,
        tooltip=f"Hat: {line}"
    ).add_to(m)

    # İstasyonları marker olarak ekle
    for s in stations:
        lat, lon = station_id_to_coords[str(s)]
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"İstasyon ID: {s}"
        ).add_to(m)

# --- Haritayı kaydet ---
m.save("metro_map.html")
print("Harita kaydedildi: metro_map.html")
