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
    print(len(zone_list))
    print(len(set(zone_list)))
    zone_list = sorted(list(set(zone_list)))
    zone_dict = {}
    for x in range(len(zone_list)):
        zone_dict[zone_list[x]] = x

    return station_zones, zone_dict

def process_bike_share():
    df = pd.read_csv('./2022-09.csv')
    print(len(df))
    
    rand_day = random.randrange(1, 30)
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%m/%d/%Y %H:%M')
    #print(df.loc[pd.to_datetime(df['Start Time'] == '2022.09.01')])
    print(df['Start Time'].head(10)) 


process_bike_share()