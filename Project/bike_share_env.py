import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import geopandas as gpd
import random
import math
from shapely.geometry import Point, Polygon
import pandas as pd

class BikeShareEnv(gym.Env):
    #time_step: timestep in minutes 
    #zone_size: edge size of square grid zone in metres to be used for balancing supply (recommend: 500 m)
    def __init__(self, df, size): 
        super(BikeShareEnv, self).__init__()
        self.time_step = 60 #step simulation time in minutes 
        self.ride_data = df #dataframe containing bike share trip data for a month
        self.zone_size = size
        self.max_distance = 1500
        self.station_zones, self.zone_dict, self.zones = self.zone_division()
        self.num_zones = len(self.zone_dict) # number of zones in the environment 
        self.max_capacity = self.max_zone_capacity() #maximum capacity in each zone, numpy vector 
        self.zone_dist = self.find_zone_distance()
        

        #spaces
        self.action_space = spaces.Box(low=np.zeros(num_zones), high=self.max_capacity), dtype=np.float32)
        self.observation_space = spaces.Dict(
                            {'demand': spaces.Box(low=np.zeros(num_zones), high=self.max_capacity), 
                              'arrival': spaces.Box(low=np.zeros(num_zones), high=self.max_capacity), 
                              'supply': spaces.Box(low=np.zeros(num_zones), high=self.max_capacity)
                            })
        #episode 
        self.max_length = 24*60 #minutes in one day
        self.curr_step = None
        self.done = None
        self.prev_supply = None
        self.reward_history = None
        self.history = None
        self.start_day = None
        self.reset()
        

    def reset(self, options=None):
        self.done = False
        self.curr_step = 0
        self.reward_history = []
        self.history = {}
        random_supply = self.initialize_random_supply() #randomly sample supply from the overall supply capacity
        obs =  {'demand': np.zeros(self.num_zones), 
                'arrival': np.zeros(self.num_zones), 
                'supply': random_supply}

        self.prev_supply = random_supply

        self.start_day = self.np_random.integers(0, 30, dtype=np.int64)
        #to-do: initialize a random list based on capacity for each zone 
        return obs

    def step(self, action):
        self.done = False

        if self.curr_step == self.max_length:
            self.done = True
        
        demand, arrivals = self.process_bike_share()
        fulfilled_demand = self.find_fulfilled_demand(demand, action)
        fulfilled_arrivals = self.find_fulfilled_arrivals(arrivals)
        temp = np.subtract(self.prev_supply, fulfilled_demand) #Prev supply updated with fulfilled demand removed
        updated_supply = np.add(temp, fulfilled_arrivals)
        self.prev_supply = updated_supply

        step_award = fulfilled_demand.sum()/demand.sum()
        self.current_step += self.time_step
        obs = {
                'demand': fulfilled_demand,
                'arrival': fulfilled_arrival,
                'supply': self.prev_supply
                }
        return obs, step_reward, self.done, info

    def render(self, mode="console"):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
    
    def find_fulfilled_demand(self, demand, action):
        #find fulfilled demand based on overall demand and user cost model 
        supply = self.prev_supply
        extra_demand = np.subtract(prev_supply, demand)
        lost_demand = np.zeros(self.num_zones)
        zone_dict_rev = {y: x for x, y in self.zone_dict.items()}
        
        for x in range(len(extra_demand)):
            if extra_demand[x] < 0:
                zon = zone_dict_rev[x]
                nearest_zones = self.zone_dist[zon]
                if len(nearest_zones) == 0: 
                    lost_demand[x] = abs(extra_demand[x]) #extra demand is lost if no nearby zones available
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
                            user_reject = ((visited[y][0]/self.max_distance)**2)*12 > action_price[x] 
                            #bool value to check if user cost for the walking distance to the nearest zone is greater than the offered price
                            # if so, user rejects the offered price, since it offers negative utility. If user rejects, demand is lost.  
                            if user_reject:
                                lost_demand[x] += visited[y][1] 
        return np.subtract(demand, lost_demand) #fulfilled demand is overall demand subtracted by lost_demand

    def find_fulfilled_arrivals(arrivals):
        #find fulfileld arrivals based on overall arrivals and user cost model


    def zone_division(self):
        response = requests.get("https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information/")
        stations = response.json()

        station = {}

        #get station id and corresponding geo coordinates
        for x in range(len(stations['data']['stations'])):
            station[int(stations['data']['stations'][x]['station_id'])] = [stations['data']['stations'][x]['lat'], stations['data']['stations'][x]['lon']]

        df = pd.DataFrame(station)

        df = df.transpose()

        #add station geometry into a geopandas data frame
        df.reset_index(inplace=True)
        df.set_axis(['station','lat', 'lon'], axis='columns', inplace=True)
        geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
        df = df.drop(['lat', 'lon'], axis=1)
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
        gdf.to_crs(crs="EPSG:3857", inplace=True)

        #calculate bounds for the station coordinates to create a polygon containing all stations
        #split polygon into even square grid of zone_size x zone_size, geometry data stored in geopandas dataframe
        bounds = gdf.total_bounds
        x_coords = np.arange(bounds[0] + distance/2, bounds[2], self.zone_size)
        y_coords = np.arange(bounds[1] + distance/2, bounds[3], self.zone_size)
        combinations = np.array(list(product(x_coords, y_coords)))
        squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(distance / 2, cap_style=3)

        
        zones = gpd.GeoDataFrame(geometry=gpd.GeoSeries(squares), crs="EPSG:3857") #df containing coordinates for each zone

        
        zones_w_station = gpd.sjoin(zones, gdf, how='right') #all zones that contain at least one bike station
        station_zones = {}
        zone_list = []
        zones_w_station = zones_w_station.dropna(subset='index_left')
        
        for station, zone in zip(zones_w_station['station'], zones_w_station['index_left']):
            station_zones[station] = int(zone) #links each station id to a zone id 
            zone_list.append(int(zone))
        
        zone_list = sorted(list(set(zone_list)))
        
        zone_dict = {}
        for x in range(len(zone_list)):
            zone_dict[zone_list[x]] = x #a dict with each zone id corresponding to an index of 0 to num_zones

        return station_zones, zone_dict, zones
        
    def max_zone_capacity(self):
        response = requests.get("https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information/")
        stations = response.json()

        station = {}

        #get max capacity for each station
        for x in range(len(stations['data']['stations'])):
            station[int(stations['data']['stations'][x]['station_id'])] = [stations['data']['stations'][x]['capacity']]

        max_capacity = np.zeros(self.num_zones)

        #calculate max capacity for each zone
        for key, val in station.items():
            if key in self.station_zones:
                zone = station_zones[key]
            max_capacity[self.zone_dict[zone]] += val 
        return max_capacity

    def initialize_random_supply(self):
        rand_supply = np.zeros(self.num_zones)
        for x in range(self.num_zones):
            rand_supply[x] = random.randrange(0, self.max_capacity[x])
        return rand_supply

    def process_bike_share(self):
        df = self.ride_data
        
        #convert dataframe to datetime objects
        df['Start Time'] = pd.to_datetime(df['Start Time'], format='%m/%d/%Y %H:%M')
        df['End Time'] = pd.to_datetime(df['End Time'], format='%m/%d/%Y %H:%M')

        #create start and end time for each timestep
        t_start = self.curr_step
        t_end = self.curr_step + self.time_step - 1
        t_s_str = str(datetime.timedelta(minutes=t_start))
        t_e_str = str(datetime.timedelta(minutes=t_end))
        start_step = '2022-09-{day} {time}'.format(day=self.start_day, time=t_s_str)
        end_step = '2022-09-{day} {time}'.format(day=self.start_day, time=t_s_str)

        #mask for start time
        mask = (df['Start Time'] >= start_step) & (df['Start Time'] <= end_step)
        
        #add demand and arrivals to each zone for the chosen timestep
        demand_vec = np.zeros(self.num_zones)
        arrival_vec = np.zeros(self.num_zones)
        for start_st, end_st in zip(df['Start Station Id'].loc[mask], df['End Station Id'].loc[mask]):
            if start_st in self.station_zones:
                start_zone = self.station_zones[start_st]
            if end_st in self.station_zones:
                end_zone = self.station_zones[end_st]
            demand_vec[self.zone_dict[start_zone]] +=1
            arrival_vec[self.zone_dict[end_zone]] +=1
        
        return demand_vec, arrival_vec

    def find_zone_distance(self):
        zone_dist = {}
        temp_dist = {}
        for a, b in self.zone_dict.items():
            a_point = self.zones['geometry'].loc[a].centroid
            for c, d in self.zone_dict.items():
                c_point = self.zones['geometry'].loc[c].centroid
                dist = a_point.distance(c_point)
                if dist < self.max_distance and a != c:
                    temp_dist[c] = int(dist)
            zone_dist[a] = temp_dist
            temp_dist = {}
        return zone_dist
