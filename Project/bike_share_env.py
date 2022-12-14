import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import geopandas as gpd
import random
import math
from itertools import product
from shapely.geometry import Point, Polygon
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3, PPO, A2C, SAC, DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from copy import deepcopy

class BikeShareEnv(gym.Env):
    #time_step: timestep in minutes 
    #zone_size: edge size of square grid zone in metres to be used for balancing supply (recommend: 500 m)
    def __init__(self, data, size): 
        super(BikeShareEnv, self).__init__()
        self.time_step = 1 #step simulation time in hours 
        self.ride_data = data #dataframe containing bike share trip data for a month
        self.zone_size = size
        self.max_distance = 2000
        self.station_zones, self.zone_dict, self.zones = self.zone_division()
        self.num_zones = len(self.zone_dict) # number of zones in the environment 
        self.max_capacity = self.max_zone_capacity() #maximum capacity in each zone, numpy vector 
        self.zone_dist = self.find_zone_distance()
        self.max_budget = 3000.0

        #spaces
        self.action_space = spaces.Box(low=-2.5, high=2.5, shape=(self.num_zones,), dtype=np.float32)
        self.observation_space = spaces.Dict(
                            {'demand': spaces.Box(low=np.zeros(self.num_zones), high=self.max_capacity, shape=(self.num_zones,), dtype=np.float64), 
                              'fulfilled_demand': spaces.Box(low=np.zeros(self.num_zones), high=self.max_capacity, shape=(self.num_zones,), dtype=np.float64), 
                              'supply': spaces.Box(low=np.zeros(self.num_zones), high=self.max_capacity, shape=(self.num_zones,), dtype=np.float64),
                              'budget': spaces.Box(low =0 , high=self.max_budget, shape=(1,))
                            }) 
        # #spaces.Box(low=np.zeros(self.num_zones), high=self.max_capacity, shape=(self.num_zones,), dtype=np.float64)
                        
        # #episode 
        self.max_length = 20 #hours in one day, 0 indexed
        self.curr_step = None
        self.done = None
        self.prev_supply = None
        self.reward_history = None
        self.history = None
        self.start_day = None
        self.reset()
        self.curr_budget = None
        self.total_step = None
        self.reward = None
        

    def reset(self, options=None):
        self.done = False
        self.curr_step = random.randrange(6, 20)
        self.reward_history = []
        self.history = []
        
        self.curr_budget = self.max_budget
        random_supply = self.initialize_random_supply() #randomly sample supply from the overall supply capacity
        obs = {
                'demand': np.zeros(self.num_zones), 
                'fulfilled_demand': np.zeros(self.num_zones), 
                'supply': random_supply,
                'budget': np.array(self.max_budget)
                }#
        
        self.prev_supply = random_supply
        self.delay_arrivals = np.zeros(self.num_zones)
        self.start_day = random.randrange(1, 30)
        self.reward = 0
        self.total_step = 0
        info = {}
        return obs

    def step(self, action):
        self.done = False

        if self.curr_step == self.max_length:
            self.curr_step = random.randrange(6, 20)
        if self.total_step == 24:
            self.reset()
            self.done = True
        res = sum(self.reward_history[-10:])
        # if self.total_step > 25 and res < 1:
        #     self.reset()
        #     self.done = True
        self.total_step += 1

        demand = self.process_bike_demand()
        
        fulfilled_demand, extra_filled = self.find_fulfilled_demand(demand, action)
        

        self.reward = extra_filled
        self.reward_history.append(self.reward)
        updated_supply = np.subtract(self.prev_supply, fulfilled_demand)

       # print('Current Supply:', updated_supply, 'Max Capacity: ', self.max_capacity)
        print('Step: ', self.total_step, 'Reward: ', self.reward, 'Demand: ', demand.sum(), 'Fulfill: ', fulfilled_demand.sum(), 'Supply: ', updated_supply.sum(), 'Budget: ', self.curr_budget)

        obs = {
                'demand': demand,
                'fulfilled_demand': fulfilled_demand,
                'supply': updated_supply,
                'budget': np.array(self.curr_budget)
                }
        arrivals = self.process_bike_arrivals(fulfilled_demand)
       # print('Arrivals: ', arrivals[200:210])

        fulfilled_arrivals = self.find_fulfilled_arrivals(arrivals)
       # print('Fuflilled Arrivals ', fulfilled_arrivals[200:210])
        
        self.prev_supply = np.add(self.prev_supply, fulfilled_arrivals) #Prev supply updated with fulfilled demand removed
        
        self.curr_step += self.time_step


        info = {}
        self.history.append(self.reward)
        return obs, self.reward, self.done, info
    
    def norm_obs(self, obs):
        new_obs = np.zeros(len(obs))
        for x in range(len(obs)):
            new_obs[x] = obs[x]/self.max_capacity[x]
        return new_obs

    def render(self, mode="console"):
        if mode != 'console':
            raise NotImplementedError()        

    def close(self):
        pass
    
    def return_history(self):
        return self.history
    
    def find_fulfilled_demand(self, demand, action):
        #find fulfilled demand based on overall demand and user cost model 
        extra_demand = np.subtract(self.prev_supply, demand)
        lost_demand = np.zeros(self.num_zones)
        shifted_demand = np.zeros(self.num_zones)
        zone_dict_rev = {y: x for x, y in self.zone_dict.items()}
        demand_copy = deepcopy(demand)        
        for x in range(len(extra_demand)):
            if extra_demand[x] < 0:
                zon = zone_dict_rev[x]
                nearest_zones = deepcopy(self.zone_dist[zon])
                if len(nearest_zones) == 0: 
                    lost_demand[x] = abs(extra_demand[x]) #extra demand is lost if no nearby zones available
                else: 
                    filled = False
                    visited = []
                    extra = abs(extra_demand[x])
                    while extra > 0 and len(nearest_zones) > 0: 
                        nearest = min(nearest_zones, key=nearest_zones.get)
                        avail = extra_demand[self.zone_dict[nearest]]
                        if avail >= extra:
                            used = extra
                            extra = 0
                        elif avail > 0 and avail < extra:
                            used = avail 
                            extra = extra - avail
                        else: 
                            used = 0
                        visited.append([nearest, nearest_zones[nearest], used])
                        nearest_zones.pop(nearest)
                        
                    for y in range(len(visited)):
                        dist = visited[y][1]
                        user_accept = (2.26850903717*math.pow(10, -6)*(dist**2) - 0.000645645*dist + 0.84182) < (action[x] + 2.5) 
                        #bool value to check if user cost for the walking distance to the nearest zone is greater than the offered price
                        # if so, user rejects the offered price, since it offers negative utility. If user rejects, demand is lost.  
                        if user_accept and self.curr_budget > 0:
                            self.curr_budget = self.curr_budget - (5 - (action[x]+2.5))
                            shifted_demand[self.zone_dict[visited[y][0]]] += visited[y][2]
                        else:
                            lost_demand[x] += visited[y][2] 
                            
        for x in range(self.num_zones):
            if extra_demand[x] < 0:
                demand_copy[x] = demand_copy[x] + (extra_demand[x])
        # print('Demand: ', demand[200:210])
        # print('Supply: ', self.prev_supply[200:210])
        # print('Extra Demand: ', extra_demand[200:210])
        # print('Demand Copy: ',demand_copy[200:210])
        # print('Shifted Demand:', shifted_demand[200:210])
        fulfilled = np.add(demand_copy, shifted_demand)
        print('Lost: ', lost_demand.sum())
        
        return fulfilled, shifted_demand.sum() #fulfilled demand is overall demand subtracted by lost_demand

    def find_fulfilled_arrivals(self, arrivals):
        arrival_capacity = np.subtract(self.max_capacity, self.prev_supply)
        extra_arrivals = np.subtract(arrival_capacity, arrivals)
        lost_arrivals = np.zeros(self.num_zones)
        shifted_arrivals = np.zeros(self.num_zones)
        zone_dict_rev = {y: x for x, y in self.zone_dict.items()}
        arrivals_copy = deepcopy(arrivals)        
        for x in range(len(extra_arrivals)):
            if extra_arrivals[x] < 0:
                zon = zone_dict_rev[x]
                nearest_zones = deepcopy(self.zone_dist[zon])
                if len(nearest_zones) == 0: 
                    lost_arrivals[x] = abs(extra_arrivals[x]) #extra demand is lost if no nearby zones available
                else: 
                    visited = []
                    extra = abs(extra_arrivals[x])
                    while extra > 0 and len(nearest_zones) > 0: 
                        nearest = min(nearest_zones, key=nearest_zones.get)
                        avail = extra_arrivals[self.zone_dict[nearest]]
                        if avail >= extra:
                            used = extra
                            extra = 0
                        elif avail > 0 and avail < extra:
                            used = avail 
                            extra = extra - avail
                        else: 
                            used = 0
                        visited.append([nearest, nearest_zones[nearest], used])
                        nearest_zones.pop(nearest)
                        
                    for y in range(len(visited)):
                        shifted_arrivals[self.zone_dict[visited[y][0]]] += visited[y][2] 
                            
        for x in range(self.num_zones):
            if extra_arrivals[x] < 0:
                arrivals_copy[x] = arrivals_copy[x] - abs(extra_arrivals[x])
        
        fulfilled = np.add(arrivals_copy, shifted_arrivals)
        
        return fulfilled

    def process_bike_arrivals(self, fulfilled_demand):
        dat = self.ride_data
        
        #convert dataframe to datetime objects
        dat['Start Time'] = pd.to_datetime(dat['Start Time'], format='%m/%d/%Y %H:%M')
        dat['End Time'] = pd.to_datetime(dat['End Time'], format='%m/%d/%Y %H:%M')

        #create start and end time for each timestep
        t_start = self.curr_step
        #t_end = self.curr_step + self.time_step
        start_step = pd.Timestamp(datetime.datetime(2022, 9, self.start_day, hour=t_start))
        end_step = pd.Timestamp(datetime.datetime(2022, 9, self.start_day, hour=t_start, minute=59))
        #mask for start time
        mask = (dat['Start Time'] >= start_step) & (dat['Start Time'] <= end_step)
        
        trips = {}
        for start_st, end_st in zip(dat['Start Station Id'].loc[mask], dat['End Station Id'].loc[mask]):
            if start_st in self.station_zones and end_st in self.station_zones:
                    start_zone = self.station_zones[start_st]
                    end_zone = self.station_zones[end_st]
                    start_ind = self.zone_dict[start_zone]
                    end_ind = self.zone_dict[end_zone]
                    if start_ind in trips:
                        if end_ind in trips[start_ind]:
                            trips[start_ind][end_ind] += 1
                        else:
                            trips[start_ind].update({end_ind: 1})
                    else:
                        trips[start_ind] = {end_ind: 1}

            for x in range(self.num_zones):
                if fulfilled_demand[x] > 0 and x in trips:
                    z = fulfilled_demand[x]
                    while z > 0 and sum(trips[x].values()) > 0:
                        for key in trips[x]:
                            if trips[x][key] == 0:
                                continue
                            trips[x][key] -=1
                            z -=1
                            if z == 0:
                                break        
            
            arrivals = np.zeros(self.num_zones)
            for x, y in trips.items():
                for a, b in trips[x].items():
                    arrivals[a] += b  
                
        return arrivals

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
        x_coords = np.arange(bounds[0] + self.zone_size/2, bounds[2], self.zone_size)
        y_coords = np.arange(bounds[1] + self.zone_size/2, bounds[3], self.zone_size)
        combinations = np.array(list(product(x_coords, y_coords)))
        squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(self.zone_size / 2, cap_style=3)

        
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
                zone = self.station_zones[key]
            max_capacity[self.zone_dict[zone]] += val 
        return max_capacity

    def initialize_random_supply(self):
        rand_supply = np.zeros(self.num_zones)
        for x in range(self.num_zones):
            y = max(1, self.max_capacity[x] - 0)
            rand_supply[x] = random.randrange(0, y)
        return rand_supply

    def process_bike_demand(self):
        dat = self.ride_data
        
        #convert dataframe to datetime objects
        dat['Start Time'] = pd.to_datetime(dat['Start Time'], format='%m/%d/%Y %H:%M')
        dat['End Time'] = pd.to_datetime(dat['End Time'], format='%m/%d/%Y %H:%M')

        #create start and end time for each timestep
        t_start = self.curr_step
        #t_end = self.curr_step + self.time_step
        start_step = pd.Timestamp(datetime.datetime(2022, 9, self.start_day, hour=t_start))
        end_step = pd.Timestamp(datetime.datetime(2022, 9, self.start_day, hour=t_start, minute=59))
        #mask for start time
        mask = (dat['Start Time'] >= start_step) & (dat['Start Time'] <= end_step)
        
        #add demand and arrivals to each zone for the chosen timestep
        demand_vec = np.zeros(self.num_zones)
        arrival_vec = np.zeros(self.num_zones)
        start_end = {}
        for start_st, end_st in zip(dat['Start Station Id'].loc[mask], dat['End Station Id'].loc[mask]):
            if start_st in self.station_zones and end_st in self.station_zones:
                start_zone = self.station_zones[start_st]
                demand_vec[self.zone_dict[start_zone]] +=1

        return demand_vec

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

if __name__ == "__main__":
    tmp_path = "./tmp/"
    new_logger = configure(tmp_path, ["stdout", "csv"])
    data = pd.read_csv('./2022-09.csv')
    env = BikeShareEnv(data, 300)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
    # Because we use parameter noise, we should use a MlpPolicy with layer normalization
    model = SAC("MultiInputPolicy", env, learning_rate=0.0007, gamma=0.99, action_noise=action_noise,  batch_size=24, learning_starts=48,  tensorboard_log=tmp_path, verbose=1) #learning_rate=0.0003, action_noise=action_noise, gamma=0.49

    model.set_logger(new_logger)
    model.learn(total_timesteps=100000, log_interval=1, progress_bar=True, tb_log_name='log1')

    vec_env = model.get_env()
    obs = vec_env.reset()
    
    for i in range(10):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(action)
        print(reward)
    