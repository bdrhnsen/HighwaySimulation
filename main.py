import highway_simulation
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from highway_simulation.environments.relative_to_ego_highway_env import HighwayEnv

from highway_simulation.scripts.util.config import Config
#from highway_simulation.environments.visualizer import ObservationVisualizer
#from rl_visualizer import RLVisualizer
env = gym.make(id='highway_env')
config = Config(min_vel=13,max_vel=36,
                min_rewardable_vel=28, max_rewardable_vel=36,
                collision_threshold=1,lane_change_duration=12,
                num_of_vehicles=60, road_length=51000,
                vehicle_width=4.5,
                vehicle_height=2,
                screen_width=1499,
                screen_height=1000,
                lane_width=3.5,
                time_step=0.01,
                num_lanes=3,
                effective_sim_length=7500,
                effective_sim_time=999999,
                aggresive_driver = False,
                ego_drives_with_mobil=False,
                evaluation_mode=False)
#env.set_config(config)
#visualizer = ObservationVisualizer(HighwayEnv.default_config())
#from stable_baselines3.common.env_checker import check_env
#check_env(env)
#model = PPO.load("./models/Aggresive-DriverV5.zip", env=env)
model = PPO.load("./example_model.zip", env=env)
#visualizer = RLVisualizer(env.action_space)
#visualizer = RLVisualizer(env.action_space)
def main():
    total_reward = 0.0
    env.set_config(config)

    while True:
        
        env.seed(6)
        obs,_ = env.reset()
        done = False

        while True:
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, _,_ = env.step(action)
            
            
            total_reward += reward
            print(f"{obs} and action : {action}")
            #visualizer.plot_observation(obs)
            # Obtain the action probabilities from the model's policy
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            dis = model.policy.get_distribution(obs_tensor)
            probs = dis.distribution.probs.cpu().detach().numpy()
            #visualizer.update_visualization(probs, reward=reward, chosen_action=action)

            env.render()
            #if done:
            #    obs = env.reset()
        break   

        

if __name__ == "__main__":
    #check_env(env)
    main()
