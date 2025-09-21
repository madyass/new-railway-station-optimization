import json
from shapely.geometry import Polygon, Point
import folium

# -----------------------------
# 1. JSON’dan poligonları yükle
# -----------------------------
def load_polygons_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    city_polygon = Polygon(data["city_polygon"])
    exclude_polygons = [Polygon(p) for p in data.get("exclude_polygons", [])]
    return city_polygon, exclude_polygons

# -----------------------------
# 2. Grid oluştur
# -----------------------------
def generate_grid(polygon, lat_offset=0.01, lon_offset=0.01):
    min_lat = min(lat for lat, lon in polygon.exterior.coords)
    max_lat = max(lat for lat, lon in polygon.exterior.coords)
    min_lon = min(lon for lat, lon in polygon.exterior.coords)
    max_lon = max(lon for lat, lon in polygon.exterior.coords)

    grid_points = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            point = Point(lat + lat_offset/2, lon + lon_offset/2)
            if polygon.contains(point):
                grid_points.append([point.x, point.y])
            lon += lon_offset
        lat += lat_offset
    return grid_points

# -----------------------------
# 3. Engel poligonlarla filtrele
# -----------------------------
def filter_stations(candidate_stations, exclude_polygons):
    filtered = []
    for s in candidate_stations:
        point = Point(s[0], s[1])
        if not any(poly.contains(point) for poly in exclude_polygons):
            filtered.append(s)
    return filtered

# -----------------------------
# 4. Harita görselleştirme
# -----------------------------
def visualize_map(city_polygon, exclude_polygons, stations, output_file="map.html"):
    avg_lat = sum(lat for lat, lon in stations) / len(stations)
    avg_lon = sum(lon for lat, lon in stations) / len(stations)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10, tiles="CartoDB positron")

    # İstanbul poligonu
    folium.Polygon(
        locations=list(city_polygon.exterior.coords),
        color="blue",
        weight=3,
        fill=True,
        fill_opacity=0.2,
        popup="İstanbul Sınırı"
    ).add_to(m)

    # Engel poligonlar
    for poly in exclude_polygons:
        folium.Polygon(
            locations=list(poly.exterior.coords),
            color="red",
            weight=2,
            fill=True,
            fill_opacity=0.3,
            popup="Engel Alan"
        ).add_to(m)

    # Filtrelenmiş aday istasyonlar
    for s in stations:
        folium.CircleMarker(
            location=[s[0], s[1]],
            radius=3,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.7
        ).add_to(m)

    m.save(output_file)
    print(f"Harita kaydedildi: {output_file}")

# -----------------------------
# 5. Ana akış
# -----------------------------
if __name__ == "__main__":
    city_poly, exclude_polys = load_polygons_from_json("/home/saydam/github_projects/metro_optimization/src/core/polygons.json")
    candidates = generate_grid(city_poly)
    filtered = filter_stations(candidates, exclude_polys)
    print(f"Aday istasyon sayısı (engel sonrası): {len(filtered)}")
    visualize_map(city_poly, exclude_polys, filtered, "istanbul_filtered_stations.html")
