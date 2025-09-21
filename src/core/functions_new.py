import pandas as pd
import json

# uploas csv
df = pd.read_csv("/home/saydam/github_projects/metro_optimization/data/stations.csv")

# extact line name
def extract_line_code(proje_adi):
    return proje_adi.split()[0].lower()

df["line_code"] = df["PROJE_ADI"].apply(extract_line_code)

# 1. line -> stations
line_to_stations = df.groupby("line_code").apply(lambda x: x.index.tolist()).to_dict()

# 2. station id -> [lat, lon]
station_id_to_coords = df[["lat", "lon"]].T.to_dict("list")

import math

# calculate distance
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def order_stations(line_to_stations, station_id_to_coords):
    ordered_lines = {}

    for line, stations in line_to_stations.items():
        if len(stations) <= 2:
            ordered_lines[line] = stations
            continue

        start = min(stations, key=lambda s: station_id_to_coords[s][1])

        ordered = [start]
        remaining = set(stations) - {start}

        while remaining:
            last = ordered[-1]
            next_station = min(
                remaining,
                key=lambda s: haversine(station_id_to_coords[last], station_id_to_coords[s])
            )
            ordered.append(next_station)
            remaining.remove(next_station)

        ordered_lines[line] = ordered

    return ordered_lines

ordered = order_stations(line_to_stations, station_id_to_coords)

with open("ordered_stations.json", "w", encoding="utf-8") as f:
    json.dump(ordered, f, ensure_ascii=False, indent=4)

with open("station_id_to_coords.json", "w", encoding="utf-8") as f:
    json.dump(station_id_to_coords, f, ensure_ascii=False, indent=4)

