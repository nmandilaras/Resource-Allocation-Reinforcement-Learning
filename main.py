import gym
import constants
import numpy as np
from quantization import Quantization
from agent import Agent

if __name__ == "__main__":
    env = gym.make(constants.environment)
    high_intervals = env.observation_space.high
    low_intervals = env.observation_space.low

    var_freq = [constants.theta_freq, constants.theta_dot_freq]
    vars_dict = [(low_intervals[2], high_intervals[2], var_freq[0]), (low_intervals[3], high_intervals[3], var_freq[1])]
    discriminator = Quantization(vars_dict)

    agent = Agent(env, var_freq)
    # observation = env.reset()
    # state = discriminator.digitize(observation[2:])
    # print(state)

    for i_episode in range(200):
        observation = env.reset()
        state = discriminator.digitize(observation[2:])
        agent.epsilon /= np.sqrt(i_episode + 1)
        for t in range(200):
            env.render()
            action = agent.choose_action(state)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                break
            new_state = discriminator.digitize(observation[2:])
            agent.update(state, action, new_state, reward)
            state = new_state
        else:
            print("Episode {} finished successful!".format(i_episode))

    env.close()
