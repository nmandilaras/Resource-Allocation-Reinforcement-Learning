import random
import numpy as np
from agents.classic_agent import ClassicAgent


class DoubleQAgent(ClassicAgent):
    """
        Algorithm better suited for stochastic environments
    """

    def __init__(self, env, dimensions, lr=0.1, gamma=0.999, epsilon=1):
        super().__init__(env, dimensions, lr, gamma, epsilon)
        self.q_table_a = np.zeros(dimensions + [env.action_space.n])
        self.q_table_b = np.zeros(dimensions + [env.action_space.n])

    def update(self, cur_state, action, reward, new_state, new_action):
        if random.random() < .5:  # update q_table_a
            upd_table = self.q_table_a
            eval_table = self.q_table_b
        else:   # update q_table_b
            upd_table = self.q_table_b
            eval_table = self.q_table_a

        upd_table[cur_state + (action,)] = (1 - self.lr) * upd_table[cur_state + (action,)] + self.lr * (
                    reward + self.gamma * eval_table[new_state + (np.argmax(upd_table[new_state]),)])

        self.q_table[cur_state + (action,)] = (upd_table[cur_state + (action,)] + eval_table[cur_state + (action,)]) / 2.0

        # we calculated the average of the two Q values for each action as mentioned in the paper
        # so action is taken based on the average