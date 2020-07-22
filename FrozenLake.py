# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0").env

#• makes environment determistic
#from gym.envs.registration import register
#register(
#        
#        id='FrozenLakeNotSlippery-v0',
#        entry_point='gym.envs.toy_text:FronzenLakeEnv',
#        kwargs={'mape_name':'4*4','is_slippery':False},
#        max_episode_steps = 100,
#        reward_threshold = 0.78,        
#)

# Q table
q_table = np.zeros([env.observation_space.n,env.action_space.n])

# Hyperparameter
alpha = 0.8
gamma = 0.9
epsilon = 0.1

#Plotting Matrix
reward_list = []


episode_number = 50000

for i in range(1,episode_number):
    
    # initialize envoriment 
    
    state = env.reset()
    
    reward_count = 0
    
    while True:
        #exploit or expplore olduğun yerde kal veya yeni yerler keşfet
        #%10 keşfet %90 kal 
        
        if random.uniform(0,1) < epsilon:
            action= env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        #neler var
        next_state, reward, done, _ = env.step(action)
        
        #Q learing Fuction
        old_value = q_table[state,action]#old value
        next_max = np.max(q_table[next_state])#nextmax
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        #Q table update
        q_table[state,action] = next_value
        
        #update state
        
        state = next_state 
        #find wrong dropouts
        
#        if reward == -10:
#            dropouts += 1
             
        reward_count += reward  
        
        if done:
            break
       
    if i%10 == 0:       
        reward_list.append(reward_count)
        print("Episode: {}, reward {}".format(i,reward_count))
    
    
plt.plot(reward_list)   
    
    
    
## %%
#        
#fig ,axs = plt.subplots(1,2)
#axs[0].plot(reward_list)        
#axs[0].set_xlabel("episode")
#axs[0].set_ylabel("reward")    
#    
#axs[1].plot(dropout_list)        
#axs[1].set_xlabel("episode")
#axs[1].set_ylabel("dropout")   
#
#axs[0].grid(True) 
#axs[1].grid(True) 
#
#plt.show()    

# %%

#q table 

#env.s = env.encode(1,3,4,4)
#env.render()
#        
#    
#env.s = env.encode(3,2,1,0)
#env.render()    