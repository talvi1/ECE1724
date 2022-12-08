import gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C

def test():
    env = gym.make("CartPole-v1")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)

        print(action)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

def load_dataset():
    df = pd.read_csv('./data.csv', parse_dates=True, index_col='Time')
    return df

def process_data():
    df = load_dataset()
    prices = df.loc[:, 'Close'].to_numpy()
    frame_bound = (24, df.shape[0])
    window_size = 24
    prices[frame_bound[0] - window_size]  # validate index (TODO: Improve validation)
    prices = prices[frame_bound[0]-window_size:frame_bound[1]]

    diff = np.insert(np.diff(prices), 0, 0)
    signal_features = np.column_stack((prices, diff))
    print(signal_features[(21-10+1):22])
    print(signal_features.shape[1])
    return prices, signal_features

test()