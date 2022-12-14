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
import datetime


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
    dat = pd.read_csv('./2022-09.csv')

    
    dat['Start Time'] = pd.to_datetime(dat['Start Time'], format='%m/%d/%Y %H:%M')
    dat['End Time'] = pd.to_datetime(dat['End Time'], format='%m/%d/%Y %H:%M')
    station_zones, zone_dict, zones = zone_division(300)
    zone_dict_rev = {y: x for x, y in zone_dict.items()}
    #create start and end time for each timestep
    t_start = 17
    #t_end = self.curr_step + self.time_step
    start_step = pd.Timestamp(datetime.datetime(2022, 9, 1, hour=t_start))
    end_step = pd.Timestamp(datetime.datetime(2022, 9, 1, hour=t_start, minute=59))
    #mask for start time
    mask = (dat['Start Time'] >= start_step) & (dat['Start Time'] <= end_step)
    
    #add demand and arrivals to each zone for the chosen timestep
    demand_vec = np.zeros(len(zone_dict))
    arrival_vec = np.zeros(len(zone_dict))
    trips = {}
    
    for start_st, end_st in zip(dat['Start Station Id'].loc[mask], dat['End Station Id'].loc[mask]):
        if start_st in station_zones and end_st in station_zones:
            start_zone = station_zones[start_st]
            end_zone = station_zones[end_st]
            start_ind = zone_dict[start_zone]
            end_ind = zone_dict[end_zone]
            if start_ind in trips:
                if end_ind in trips[start_ind]:
                    trips[start_ind][end_ind] += 1
                else:
                    trips[start_ind].update({end_ind: 1})
            else:
                trips[start_ind] = {end_ind: 1}

    for x in range(len(zone_dict)):
        if fulfilled_demand[x] > 0 and x in trips:
            z = fulfilled_demand[x]
            while z > 0 and sum(trips[x].values()) > 0:
                for key in trips[x]:
                    trips[x][key] -=1
                    z -=1
                    if z == 0:
                        break        
    
    arrivals = np.zeros(len(zone_dict))
    for x, y in trips.items():
        for a, b in trips[x].items():
            arrivals[a] += b
                


        # if start_st not in start_end:
        #     if end_st not in start_end[start_st]:
        #         start_end[start_st].update({end_st: 1})
        #     else:
        #         start_end[start_st][end_st] += 1
        # else:
        #     start_end[start_st] = {end_st: 1}   
    #print(trips)

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
            if dist < 2000 and a != c:
                temp_dist[c] = int(dist)
        zone_dist[a] = temp_dist
        temp_dist = {}
    return zone_dist

def fulfilled_demand():
    max_capacity = max_zone_capacity()
    supply = initialize_random_supply()
    demand, arrival = process_bike_share()
    station_zones, zone_dict, zones = zone_division(300)
    action = np.zeros(len(max_capacity))
    shifted_demand = np.zeros(len(max_capacity))
    for x in range(len(max_capacity)):
        action[x] = round(random.uniform(0, 5.0), 2)
    extra_demand = np.subtract(supply, demand)
    lost_demand = np.zeros(len(max_capacity))
    zone_dict_rev = {y: x for x, y in zone_dict.items()}
    zone_dist = find_zone_distance()
    curr_budget = 500
    copy_demand = demand
    for x in range(len(extra_demand)):
        if extra_demand[x] < 0:
            zon = zone_dict_rev[x]
            nearest_zones = zone_dist[zon]
            if len(nearest_zones) == 0: #no nearby zones 
                lost_demand[x] = abs(extra_demand[x])
            else: 
                visited = []
                extra = abs(extra_demand[x])
                print('Index is:', x)
                print('Extra: ', extra)
                while extra > 0 and len(nearest_zones) > 0: 
                    nearest = min(nearest_zones, key=nearest_zones.get)
                    avail = extra_demand[zone_dict[nearest]]
                    if avail >= extra:
                        used = extra
                        extra = 0
                    elif avail >= 0 and avail < extra:
                        used = avail 
                        extra = extra - avail
                    else: 
                        used = 0
                    visited.append([nearest, nearest_zones[nearest], used])
                    nearest_zones.pop(nearest)
                print(visited)
                for y in range(len(visited)):
                    dist = visited[y][1]
                    user_reject = (2.2685090*math.pow(10, -6)*(dist**2) - 0.000645645*dist + 0.84182) < (action[x]) 
                    #bool value to check if user cost for the walking distance to the nearest zone is greater than the offered price
                    # if so, user rejects the offered price, since it offers negative utility. If user rejects, demand is lost.  
                    if user_reject or curr_budget <= 0:
                        lost_demand[x] += visited[y][2]
                       # demand[zone_dict[visited[y][0]]] -= abs(extra_demand[x])
                    else:
                        curr_budget = curr_budget - (5 - (action[x]))
                        shifted_demand[zone_dict[visited[y][0]]] += visited[y][2]
                       # demand[zone_dict[visited[y][0]]] += visited[y][2]
                print('Lost: ', lost_demand[x])
                print('Budget:', curr_budget)
    z = np.zeros(len(extra_demand))
    for x in range(len(extra_demand)):
        if extra_demand[x] < 0:
            z[x] = abs(extra_demand[x])
           # demand[x] = demand[x] - abs(extra_demand[x])

    print(demand.sum())
    print(lost_demand)
    fulfilled = np.add(demand, shifted_demand)
    print(fulfilled)
    print(fulfilled.sum())
    print(z.sum())
    print(lost_demand.sum())
    print((z.sum()-lost_demand.sum())/z.sum())
    print(curr_budget/500)
    print(supply)
    print(np.subtract(supply, fulfilled))
    return np.subtract(demand, lost_demand) 

def find_fulfilled_arrivals(self, demand, arrivals, lost_demand):
        #find fulfilled arrivals based on overall arrivals, minus the lost demand
    return np.zeros(self.num_zones)

def user_reject(distance, price):
    max_distance = 2000
    return ((distance/max_distance)**2)*12 > price
    
process_bike_share()