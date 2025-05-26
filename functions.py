import numpy as np
import math
import pandas as pd
from collections import defaultdict
import folium
from shapely.geometry import Point 

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    return distance


def calculate_arrived_population(lat1 , lon1 , population_df : pd.DataFrame , k = 1 ):
    """
    k is perimeter parameter (km)
    """
    result = 0
    for index , row in population_df.iterrows():
        lat2  = row['latitude']
        lon2 = row['longitude']
        
        distance = haversine(lat1 , lon1 , lat2 , lon2)

        if distance < k:
            result +=row['population']

    return result

def calculate_arrived_neighoorhood_ids( lat1 , lon1 , 
                                        population_df : pd.DataFrame , 
                                        k = 1):

    ids = set()
    for index , row in population_df.iterrows():
        lat2  = row['latitude']
        lon2 = row['longitude']
        
        distance = haversine(lat1 , lon1 , lat2 , lon2)
        
        
        if distance < k:
            ids.add(row['neighborhood_id'])

    return ids

def create_grid(lat_min , lon_min , lat_max , lon_max , num_lat_grids = 30 , num_lon_grids = 30):
    """
    Input: lateral and longlitude (min and max) coordinates and grid number
    Output: grid_list : numpy.array , candidat_stations : numpy.array
    """
    
    lat_step = (lat_max - lat_min) / num_lat_grids
    lon_step = (lon_max - lon_min) / num_lon_grids
    
    grid_list = []
    candidate_stations = []

    for i in range(num_lat_grids):
        for j in range(num_lon_grids):
            lat_start = lat_min + (i*lat_step)
            lat_end = lat_start + lat_step
            lon_start = lon_min + (j * lon_step)
            lon_end = lon_start + lon_step

            index = (i , j)

            grid_list.append({
            'grid_id' : index,
            'lat_start' : lat_start,
            'lat_end' : lat_end,
            'lon_start' : lon_start,
            'lon_end' : lon_end
            })

            x = (lat_start + lat_end) / 2
            y = (lon_start + lon_end) / 2

            candidate_stations.append({
            'station_id' : index,
            'lat' : x,
            'lon' : y
            })

    return grid_list , candidate_stations


def create_grid_for_polygon(polygon, n = 1000):
    candidate_stations = []

    index = 1

    # 3. Bounding box sınırları
    minx, miny, maxx, maxy = polygon.bounds

    # 4. Poligon içine n adet nokta üret
    while len(candidate_stations) < n:
        random_points = np.column_stack([
            np.random.uniform(minx, maxx, n),
            np.random.uniform(miny, maxy, n)
        ])
        for x, y in random_points:
            if polygon.contains(Point(x, y)):
                candidate_stations.append({
                    'station_id' : index,
                    'lat' : x,
                    'lon' : y
                })

                index += 1
            if len(candidate_stations) >= n:
                break

    return candidate_stations

def calculate_population_per_station(candidate_stations_df : pd.DataFrame , population_df : pd.DataFrame , k = 1):
    """
    Input: candidate stations DataFrame and population DataFrame

    calculates population arrived for each candidate station and returns this DataFrame 
    """
    
    result = []
    for _ , row in candidate_stations_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        pop = calculate_arrived_population(lat, lon, population_df , k=k)
        result.append(pop)

    candidate_stations_df['arrived_population'] = result

    return candidate_stations_df

def calculate_arrived_neighborhood_per_station(candidate_stations_df : pd.DataFrame,
                                               population_df : pd.DataFrame ,
                                               k = 1):
    result = []
    for _ , row in candidate_stations_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        neighbors = calculate_arrived_neighoorhood_ids(lat , lon , population_df , k=k)
        result.append(neighbors)
    
    candidate_stations_df['arrived_neighborhoods'] = result

    return candidate_stations_df

def calculate_connectivity_dict(all_stations_pop : pd.DataFrame , MIN_DIST_KM = 1 , MAX_DIST_KM = 3):
    station_list = list(zip(all_stations_pop['station_id'], all_stations_pop['lat'], all_stations_pop['lon']))

    connectivity_dict = defaultdict(list)

    for i, (id1, lat1, lon1) in enumerate(station_list):
        for j in range(i + 1, len(station_list)):
            id2, lat2, lon2 = station_list[j]
            dist = haversine(lat1, lon1, lat2, lon2)
            if MIN_DIST_KM <= dist <= MAX_DIST_KM:
                connectivity_dict[id1].append(id2)
                connectivity_dict[id2].append(id1)  

    connectivity_dict = dict(connectivity_dict)

    return connectivity_dict


def visualize_chromosome(chromosome, stations_df):
    m = folium.Map(location=[41.015137, 28.979530], zoom_start=11 , tiles='CartoDB positron')

    color_palette = [
    '#1f77b4',  # koyu mavi
    '#ff7f0e',  # turuncu
    '#2ca02c',  # koyu yeşil
    '#d62728',  # kırmızı
    '#9467bd',  # mor
    '#8c564b',  # kahverengi
    '#e377c2',  # pembe
    '#7f7f7f',  # gri
    '#bcbd22',  # zeytin
    '#17becf',  # cam göbeği
    '#393b79',  # lacivert
    '#637939',  # zeytin yeşili
    '#8c6d31',  # koyu altın
    '#843c39',  # bordo
    '#7b4173',  # koyu pembe
    '#5254a3',  # orta mavi
    '#9c9ede'   # açık mor ama yeterince koyu
                ]
    line_colors = {}

    for i, (line, station_ids) in enumerate(chromosome.items()):
        color = color_palette[i % len(color_palette)]
        line_colors[line] = color
        line_coords = []

        for station_id in station_ids:
            row = stations_df[stations_df['station_id'] == station_id]
            if row.empty:
                continue
            lat = row.iloc[0]['lat']
            lon = row.iloc[0]['lon']
            line_coords.append((lat, lon))

            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f"{line}: {station_id}"
            ).add_to(m)

        

    return m

