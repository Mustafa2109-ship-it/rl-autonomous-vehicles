# -*- coding: utf-8 -*-
"""
Date : 03/31/2026
Mustafa Ahbab

Attempt to change codes within the original script.
Because, the way i see it the og code is very brittle and extremely sensitive to minor changes.


"""

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import os
import matplotlib.pyplot as plt

#seed adjustment beginning
import random
import torch

SEED = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True        #Forces PyTorch to use deterministic (reproducible) algorithms in cuDNN. This line has little effect in this script
torch.backends.cudnn.benchmark = False           #disables the cuDNN search for the best algorithm. Searching for the best may result in different algo in different runs. his line has little effect in this script.
#seed adjustment end

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.engine.asset_loader import AssetLoader

# =====================================================
# 1. Helper to create GIFs (unchanged)
# =====================================================
def make_GIF(frames, name="demo.gif"):
    print("Generate gif...")
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print(os.getcwd())

# =====================================================
# 2. Wrapper for simplified observation & reward
# =====================================================
class SimpleTD3Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    '''
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        info = getattr(self.env, "last_info", {})
        return self._process_obs(info), info
    This line is in the original and replaced by the new line below:
    This ensures every time the environment resets (which happens many times during 100k training steps), 
    it resets with the same seed rather than a random one.
    '''

    #seed adjustment beginning
    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        obs = self.env.reset(**kwargs)
        info = getattr(self.env, "last_info", {})
        return self._process_obs(info), info
    #seed adjustment end



    def step(self, action):
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = info.get("velocity", 0)
        #steering = info.get("steering", 0)
        #acceleration = info.get("acceleration", 0)
        lateral_dist = info.get("lateral_dist", 0)
        collision = info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("crash_building", False)
        offroad = info.get("out_of_road", False)
        
        norm_velocity, steering, acceleration, lat_norm, front_distance_norm, nav_forward, nav_left, nav_right = self._process_obs(info)
        
        #Computing front distance for reward:
        # --- Front distance ---
        agent = self.env.agent
        lidar_obs = self.env.get_single_observation()
        lidar_obs.lidar_observe(agent)
        detected_objs = lidar_obs.detected_objects
    
        min_dist = np.inf
        for obj in detected_objs:
            if obj.id == agent.id:
                continue
            relative_pos = obj.position - agent.position
            forward_proj = np.dot(relative_pos, agent.heading)
            if forward_proj > 0:
                dist = np.linalg.norm(relative_pos)
                if dist < min_dist:
                    min_dist = dist
    
        # Handle empty case
        front_distance = min_dist if min_dist < np.inf else 50.0
        #print("Front_Dist:", front_distance)
        
        #Computing safe distance threshold
        safe_distance = 5.0 + 0.5 * velocity  # in meters   as per: safe_distance = base + time_gap * speed
        #print("safe distance:", safe_distance)
        
        #norm_velocity = velocity/30.0
        #print("Velocity m/s:", norm_velocity)
        #print("Lateral dist: m", lateral_dist)
        # Reward definition
        reward = (
            9.0 * norm_velocity                     # encourage motion strongly
            - 7.0 * abs(lateral_dist)               # keep lane, but not harshly
            - 15.0 * collision
            - 20.0 * offroad
        )
        
        # Encourage throttle use when below speed
        if norm_velocity < 0.3:
            reward += 2.0 * (0.3 - norm_velocity)
        
        # Smooth steering
        if abs(steering) > 0.3:
            reward -= 0.4

        # Penalize harsh braking, reward smooth throttle
        if acceleration < -0.3:
            reward -= 0.5 * abs(acceleration)
        elif acceleration > 0 and norm_velocity < 0.8:
            reward += 0.3 * acceleration

        #new line for initial steering smoothing --- MA
        #reward -= 0.06 * abs(steering)
        
        # Safe distance handling
        safe_distance = 5.0 + 0.5 * velocity
        #print("Safe dist:", safe_distance)
        #print("Safe_dist:", safe_distance)
        #print("Front dist:", front_distance)
        if front_distance < safe_distance:
            reward -= 7.0 * (safe_distance - front_distance) / safe_distance
            #print("If running!")
           #if front_distance < 
        else:
            #print("Else condition running!")
            reward += 2.0 * norm_velocity  # extra bonus for clear road
            #reward -= 2.0 * abs(lateral_dist)
           # print("Else running!")

        

        return self._process_obs(info), reward, terminated, truncated, info

    def _process_obs(self, info):
        velocity_norm = info.get("velocity", 0) / 30.0
        steering = info.get("steering", 0)
        acceleration = info.get("acceleration", 0)
        lateral_dist = info.get("lateral_dist", 0)
        lat_norm = np.clip(lateral_dist / 5.0, -1.0, 1.0)
        
        #Computing front_dist from LIDARStateObservations class inside ScenarioEnv() -> state_obs.py 
        agent = self.env.agent
        lidar_obs = self.env.get_single_observation()    #By default, all obs conctanetaed in LIDARStateObservations class
        #print("Environment obs:", lidar_obs)
        #Get detected vehicles info stored in detected_objects list returned by the lidar_observe function
        lidar_obs.lidar_observe(agent)
        #print("Other_v_info:", lidar_obs.lidar_observe(agent))
        detected_objs = lidar_obs.detected_objects
        
        # Compute closest front vehicle manually
        front_distance = None
        agent_heading = agent.heading
        agent_pos = agent.position
        
        min_dist = np.inf
        for obj in detected_objs:
            if obj.id == agent.id:
                continue
            relative_pos = obj.position - agent_pos
            forward_proj = np.dot(relative_pos, agent_heading)
            if forward_proj > 0:  # only front vehicles
                dist = np.linalg.norm(relative_pos)
                if dist < min_dist:
                    min_dist = dist
                    #front_velocity = obj.velocity
        
        front_distance = min_dist if min_dist < np.inf else 50.0
        front_dist_norm = np.clip(front_distance / 50.0, 0.0, 1.0) 
        nav_forward = float(info.get("navigation_forward", 0.0))
        nav_left = float(info.get("navigation_left", 0.0))
        nav_right = float(info.get("navigation_right", 0.0))
        #print("Front car velocity:", front_velocity)
        #print("Front vehicle distance:", front_distance)
        
        
        return np.array([velocity_norm, steering, acceleration, lat_norm, front_dist_norm, nav_forward, nav_left, nav_right], dtype=np.float32)

# =====================================================
# 3. ScenarioEnv Setup
# =====================================================
nuscenes_data = AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False)
env_config = {
    "use_render": False,
    "data_directory": nuscenes_data,
    "manual_control": False,
    "num_scenarios": 1,
    "start_scenario_index":1,
    "reactive_traffic": True,
    #"no_traffic": True
    "allow_respawn": True,
     "sequential_seed": True  #seed
}

SCENE_ID = env_config["start_scenario_index"]

if False:


    env = ScenarioEnv(env_config)
    env_train = SimpleTD3Wrapper(env)

    class EpisodeDebugCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.throttle_hist = []

        def _on_step(self) -> bool:
            # Store throttle for each step
            action = self.locals.get("action", None)
            if action is not None:
                # Assuming throttle is the second action component (index 1)
                self.throttle_hist.append(action[1])

            # Detect end of episode
            dones = self.locals.get("dones", None)
            rewards = self.locals.get("rewards", None)
            if dones is not None and rewards is not None:
                for done, reward in zip(dones, rewards):
                    if done:
                        episode_reward = sum(self.locals["infos"][0].get("episode_rewards", [reward]))
                        mean_throttle = np.mean(self.throttle_hist) if self.throttle_hist else 0
                        print(f"Episode done: total reward = {episode_reward:.1f}, mean throttle = {mean_throttle:.2f}")
                        self.throttle_hist.clear()
            return True

    # =====================================================
    # 4. TD3 Setup
    # =====================================================
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy",
        env_train,
        seed=SEED,     #seed adjustment
        learning_rate=1e-3,
        buffer_size=200000,
        learning_starts=1000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),       # per-step training (TD3 default)
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )



    # =====================================================
    # 6. Train
    # =====================================================
    model.learn(total_timesteps=100_000, log_interval=10, callback=EpisodeDebugCallback())
    print("Full timesteps :", model.num_timesteps)  #added by me
    print("TRAIN FINISHED, now saving model...")   #added by me
    model.save("MA_td3_driveforward_v50_ogscript_change_v7.zip")




if True:
    # -----------------------------
    # 7. Evaluate / render
    # -----------------------------

    env = ScenarioEnv(env_config)
    env_eval = SimpleTD3Wrapper(env)
    model = TD3.load("MA_td3_v7_seed27.zip",env_eval)
    obs, info = env_eval.reset()

    #env.engine.global_random_seed = SEED   #seed added. trying to fix how the traffic reacts.

    agent = env.agent
    current_lane = agent.lane
    long_start, lateral_start = current_lane.local_coordinates(agent.position)
    print("starting longitudes:", long_start)
    step_time = env.config["physics_world_step_size"]
    print("step time:", step_time)

    frames = []
    total_reward = 0
    count=0
    step_reward=[]
    steering = []
    throttle = []
    velocity = []
    head_raw = [] #MA_acc
    done = False

    MAX_STEPS = 1000
    while (not done) and (count < MAX_STEPS):

    #for i in range(1,200):
        action, _ = model.predict(obs, deterministic=True)
        #print("Action:", action)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        done = terminated or truncated
        head_raw.append(info.get("step_reward_heading", 0))   #MA_acc
        

        #print("Obs:", obs)
        #print("Lateral Dist:", info.get("lateral_dist"))
        #print("Reward:", reward)
        frames.append(env.render(mode="top_down",film_size=(4000, 4000), screen_size=(500, 500)))
        count+=1
        step_reward.append(reward)
        #print("Step no.", count)
        #Computing Traffic density (vehicle density, oedestrian density & traffic obstacles density) 
        
        objects_dict = env.engine.data_manager.get_objects()
        vehicle_keywords = ["Vehicle"]  # catches DefaultVehicle, MVehicle, SVehicle, XLVehicle
        vehicle_count = sum(1 for v in objects_dict.values() if any(key in str(v) for key in vehicle_keywords))
        print("Number of vehicles:", vehicle_count)
        
        
        pedestrian_keywords = ["Pedestrian"]  
        pedestrian_count = sum(1 for v in objects_dict.values() if any(key in str(v) for key in pedestrian_keywords))
        print("Number of pedestrians:", pedestrian_count)
        
        obstacle_keywords = ["Traffic"]  
        obstacle_count = sum(1 for v in objects_dict.values() if any(key in str(v) for key in obstacle_keywords))
        print("Number of Traffic Obstacles:", obstacle_count)
        
        agent = env.agent
        current_lane = agent.lane
        long_now, lateral_now = current_lane.local_coordinates(agent.position)
        print("Distance covered:", long_now)
        #print("Data Manager returned summary:", env.engine.data_manager.get_objects())
        
        #compuitng vehicle inflow rate
        inflow = vehicle_count/(count*step_time)
        print("Step time:", count*step_time)
        print("Inflow Rate:", inflow)
        
        
        #Computing headway
        gap = obs[4]*50.0
        speed = obs[0]*30.0
        headway = gap/speed
        print("Headway:", headway)
        
        print("=======================================================================================")
        
        velocity.append(obs[0]*30.0*3.6)
        steering.append(obs[1]*0.523*(180/np.pi))   #de-normalized: *pi/6 then converted to degree *180/pi
        throttle.append(obs[2]*5.0)   #Metadrive's vehicle dynamics uses acceleration = +5m/s2 for throttle and -5m/s2 for brake. 0 for coasting
        total_reward += reward
        #print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
    print("Min head:", min(head_raw))   #MA_acc
    print("Max head:", max(head_raw))   #MA_acc
    print("Mean head:", np.mean(head_raw))   #MA_acc
    make_GIF(frames, name=f"td3_nuscenes_trainedv50_Scene{SCENE_ID}.gif")
    print("Episode finished! Total reward:", total_reward)
    #print("Scenario length:", env.engine.data_manager.current_scenario_length)


    #Plotting Step Rewards for the episode
    step_count =  list(range(0, count))
    plt.plot(step_count,step_reward)
    plt.xlabel('Episode Steps')
    plt.ylabel('Reward per Step')
    plt.title('Reward Accumulation during the evaluation episode')
    plt.grid(True)
    plt.savefig(f"rewards_td3_nuscenes_trainedv50_Scene{SCENE_ID}.png", dpi=300, bbox_inches='tight')                ###################
    plt.show()



    #Plotting velocity for the episode (km/hr)

    plt.plot(step_count,velocity)
    plt.xlabel('Episode Steps')
    plt.ylabel('Agent velocity (km/h)')
    plt.title('Agent adapted velocity during the evaluation episode')
    plt.grid(True)
    plt.savefig(f"velocity_td3_nuscenes_trainedv50_Scene{SCENE_ID}.png", dpi=300, bbox_inches='tight')                ###################
    plt.show()



    #Plotting steering angles adapted in the episode

    plt.plot(step_count,steering)
    plt.plot(step_count, [0]*len(step_count), color='gray', lw=2, linestyle='--')
    plt.xlabel('Episode Steps')
    plt.ylabel('Steering Angles (degree)')
    plt.title('Agent adapted steering actions during the evaluation episode')
    plt.grid(True)
    plt.savefig(f"steering_td3_nuscenes_trainedv50_Scene{SCENE_ID}.png", dpi=300, bbox_inches='tight')                ###################
    plt.show()

    #Plotting Throttle for the episode

    plt.plot(step_count,throttle)
    plt.plot(step_count, [0]*len(step_count), color='gray', lw=2, linestyle='--')
    plt.xlabel('Episode Steps')
    plt.ylabel('Throttle (m/s^2)')
    plt.title('Agent adapted accelerations during the evaluation episode')
    plt.grid(True)
    plt.savefig(f"throttle_td3_nuscenes_trainedv50_Scene{SCENE_ID}.png", dpi=300, bbox_inches='tight')                ###################
    plt.show()

    """
    #Plotting velocity for the episode (m/s)

    plt.plot(step_count,velocity)
    plt.xlabel('Episode Steps')
    plt.ylabel('Agent velocity (m/s)')
    plt.title('Agent adapted velocity during the evaluation episode')
    plt.grid(True)
    plt.savefig("velocity_td3_nuscenes_trainedv1.png", dpi=300, bbox_inches='tight')                ###################
    plt.show()
    """
    """
    #Plotting velocity for the episode (mph)

    plt.plot(step_count,velocity*2.237)
    plt.xlabel('Episode Steps')
    plt.ylabel('Agent velocity (mph)')
    plt.title('Agent adapted velocity during the evaluation episode')
    plt.grid(True)
    plt.savefig("velocity_td3_nuscenes_trainedv1.png", dpi=300, bbox_inches='tight')                ###################
    plt.show()
    """

    env_eval.close()
