import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class BikeShareEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    #ride_data: ride share trip data passed as a pandas dataframe
    #time_step: timestep in minutes 
    #zone_size: edge size of square grid zone in metres to be used for balancing supply (recommend: 500 m)
    def __init__(self, ride_data, time_step, zone_size): 
        assert ride_data.ndim == 2

        self.seed()
        self.ride_data = ride_data
        self.time_step = time_step #step simulation time in minutes 
        self.zones, self.num_zones = self.process_data(zone_size)
        self.shape = self.data_features.shape
        self.zone_states = []
        self.max_capacity = #add numpy array for capacity for each zone (zones are indexed)
        self.prev_supply = None

        #spaces
        self.action_space = spaces.Box(low=np.zeros(num_zones), high=self.max_capacity), dtype=np.float32)
        self.observation_space = spaces.Dict(
                            {'demand': spaces.Box(low=np.zeros(num_zones), high=self.max_capacity), 
                              'arrival': spaces.Box(low=np.zeros(num_zones), high=self.max_capacity), 
                              'supply': spaces.Box(low=np.zeros(num_zones), high=self.max_capacity)
                            })
        #episode 
        self.max_length = 24*60/time_step # 24 hours divided by timestep 
        self.curr_step = None
        self.done = None
        self.reward_history = None
        self.history = None
        self.start_day = None
        self.reset()
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.curr_step = 0
        self.reward_history = []
        self.history = {}
        random_supply = #randomly sample supply from the overall supply capacity
        obs =  {'demand': np.zeros(num_zones), 
                'arrival': np.zeros(num_zones), 
                'supply': random_supply}

        self.prev_supply = random_supply

        self.start_day = self.np_random.integers(0, 30, dtype=np.int64)
        #to-do: initialize a random list based on capacity for each zone 
        return obs

    def step(self, action):
        self.done = False
        
        self.current_step += self.time_step
        
        if self.curr_step == self.max_length:
            self.done = True
        
        demand, arrivals = self.calculate_demand_arrivals(action)
        fulfilled_demand = self.find_fulfilled_demand(demand)
        fulfilled_arrivals = self.find_fulfilled_arrivals(arrivals)
        updated_supply = self.prev_supply - fulfilled_demand + fulfilled_arrivals

        step_award = self.calculate_reward(action)


    def calculate_demand_arrivals(action):
        # calculate number of trips (demands, arrivals) requested per zone as a numpy vector for each timestep
        # for each time step get the index of the trips 
        # for each station start point and endpoint, get zone, and add demand/arrivals to the zone


    def station_zone(station):
        # return zone for each station
    
    def find_fulfilled_demand(demand):
        #find fulfilled demand based on overall demand and user cost model 

    def find_fulfilled_arrivals(arrivals):
        #find fulfileld arrivals based on overall arrivals and user cost model
