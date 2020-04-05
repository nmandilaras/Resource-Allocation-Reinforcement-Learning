import numpy as np
from agents.classic_agents.td_agent import TDAgent


class QAgent(TDAgent):

    def update(self, cur_state, action, reward, new_state, new_action):
        self.q_table[cur_state + (action,)] = (1 - self.lr) * self.q_table[cur_state + (action,)] + self.lr * (
                    reward + self.gamma * np.max(self.q_table[new_state]))
