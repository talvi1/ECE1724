import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon
from itertools import product
import folium
import pygeos
import random
import math


def zone_division(distance):
    response = requests.get("https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information/")
    stations = response.json()

    station = {}

    for x in range(len(stations['data']['stations'])):
        station[int(stations['data']['stations'][x]['station_id'])] = [stations['data']['stations'][x]['lat'], stations['data']['stations'][x]['lon']]

    df = pd.DataFrame(station)

    df = df.transpose()

    df.reset_index(inplace=True)
    df.set_axis(['station','lat', 'lon'], axis='columns', inplace=True)
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lat', 'lon'], axis=1)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    gdf.to_crs(crs="EPSG:3857", inplace=True)

    bounds = gdf.total_bounds
    x_coords = np.arange(bounds[0] + distance/2, bounds[2], distance)
    y_coords = np.arange(bounds[1] + distance/2, bounds[3], distance)
    combinations = np.array(list(product(x_coords, y_coords)))
    squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(distance / 2, cap_style=3)

    zones = gpd.GeoDataFrame(geometry=gpd.GeoSeries(squares), crs="EPSG:3857")

    zones_w_station = gpd.sjoin(zones, gdf, how='right')
    zones_w_station.to_csv("out.csv")
    station_zones = {}
    zone_list = []
    zones_w_station = zones_w_station.dropna(subset='index_left')
    for station, zone in zip(zones_w_station['station'], zones_w_station['index_left']):
        station_zones[station] = int(zone)
        zone_list.append(int(zone))
    zone_list = sorted(list(set(zone_list)))
    zone_dict = {}
    for x in range(len(zone_list)):
        zone_dict[zone_list[x]] = x

    return station_zones, zone_dict, zones

def process_bike_share():
    df = pd.read_csv('./2022-09.csv')
    print(len(df))
    
    rand_day = random.randrange(1, 30)
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%m/%d/%Y %H:%M')
    df['End Time'] = pd.to_datetime(df['End Time'], format='%m/%d/%Y %H:%M')
    #print(df.loc[pd.to_datetime(df['Start Time'] == '2022.09.01')])
    mask = (df['Start Time'] >= '2022-09-01 08:00:00') & (df['Start Time'] <= '2022-09-01 08:59:00')
    station_zones, zone_dict, y = zone_division(300)
    demand_vec = np.zeros(len(zone_dict))
    arrival_vec = np.zeros(len(zone_dict))
    start_end = {}
    for start_st, end_st in zip(df['Start Station Id'].loc[mask], df['End Station Id'].loc[mask]):
        if start_st in station_zones and end_st in station_zones:
            start_zone = station_zones[start_st]
            demand_vec[zone_dict[start_zone]] +=1
            end_zone = station_zones[end_st]
            arrival_vec[zone_dict[end_zone]] +=1
            if start_zone in start_end:
                if end_zone not in start_end[start_zone]:
                    start_end[start_zone].update({end_zone: 1})
                else:
                    start_end[start_zone][end_zone] += 1
            else:
                start_end[start_zone] = {end_zone: 1} 
    print(start_end)
       
    return demand_vec, arrival_vec

def max_zone_capacity():
    response = requests.get("https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information/")
    stations = response.json()

    station = {}

    for x in range(len(stations['data']['stations'])):
        station[int(stations['data']['stations'][x]['station_id'])] = [stations['data']['stations'][x]['capacity']]
    station_zones, zone_dict, y = zone_division(300)

    max_capacity = np.zeros(len(zone_dict))

    for key, val in station.items():
        if key in station_zones:
            zone = station_zones[key]
        max_capacity[zone_dict[zone]] += val
    return max_capacity
    
def initialize_random_supply():
    max_supply = max_zone_capacity()
    rand_supply = np.zeros(len(max_supply))
    for x in range(len(max_supply)):
        rand_supply[x] = random.randrange(0, max_supply[x])
    return rand_supply

def find_zone_distance():
    station_zones, zone_dict, zone_df = zone_division(300)
    a = zone_df.iloc[63]['geometry'].centroid
    b = zone_df.iloc[1193]['geometry'].centroid
    print(a.distance(b))
    zone_dist = {}
    temp_dist = {}
    for a, b in zone_dict.items():
        a_point = zone_df['geometry'].loc[a].centroid
        for c, d in zone_dict.items():
            c_point = zone_df['geometry'].loc[c].centroid
            dist = a_point.distance(c_point)
            if dist < 1500 and a != c:
                temp_dist[c] = int(dist)
        zone_dist[a] = temp_dist
        temp_dist = {}
    return zone_dist

def fulfilled_demand():
    max_capacity = max_zone_capacity()
    supply = initialize_random_supply()
    demand, arrival = process_bike_share()
    station_zones, zone_dict, zones = zone_division(300)
    action_price = np.zeros(len(max_capacity))
    for x in range(len(max_capacity)):
        action_price[x] = round(random.uniform(0.0, 4.0), 2)
    extra_demand = np.subtract(supply, demand)
    lost_demand = np.zeros(len(max_capacity))
    zone_dict_rev = {y: x for x, y in zone_dict.items()}
    zone_dist = find_zone_distance()
    print(extra_demand)
    for x in range(len(extra_demand)):
        if extra_demand[x] < 0:
            zon = zone_dict_rev[x]
            nearest_zones = zone_dist[zon]
            if len(nearest_zones) == 0: #no nearby zones 
                lost_demand[x] = abs(extra_demand[x])
            else: 
                filled = False
                visited = []
                while not filled: 
                    nearest = min(nearest_zones, key=nearest_zones.get)
                    if extra_demand[zone_dict[nearest]] <= 0:
                        nearest_zones.pop(nearest)
                        if len(nearest_zones) == 0:
                            lost_demand[x] = abs(extra_demand[x])
                            break
                    elif extra_demand[zone_dict[nearest]] >= abs(extra_demand[x]):
                        filled = True 
                        visited.append([nearest_zones[nearest], abs(extra_demand[x])])
                    else: 
                        a = extra_demand[zone_dict[nearest]]
                        extra_demand[x] += a
                        visited.append([nearest_zones[nearest], a])
                        nearest_zones.pop(nearest)
                if filled:
                    for y in range(len(visited)):
                        if user_reject(visited[y][0], action_price[x]):
                            lost_demand[x] += visited[y][1] 
    print(lost_demand.sum())
    print(np.subtract(demand, lost_demand).sum())

def find_fulfilled_arrivals(self, demand, arrivals, lost_demand):
        #find fulfilled arrivals based on overall arrivals, minus the lost demand
    for x in range(self.num_zones):
        

    return np.zeros(self.num_zones

def user_reject(distance, price):
    max_distance = 2000
    return ((distance/max_distance)**2)*12 > price
    
process_bike_share()