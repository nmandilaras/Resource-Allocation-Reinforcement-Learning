import torch
from agents.deep_agent import DeepAgent


class DQNAgent(DeepAgent):

    def __init__(self, num_of_actions, network, criterion, optimizer, mem_size=1000, batch_size=32, gamma=0.999, epsilon=1):
        super().__init__(num_of_actions, network, criterion, optimizer, mem_size, batch_size, gamma, epsilon)

    def compute_loss(self, state_batch, action_batch, reward_batch, non_final_next_state, non_final_mask):

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.policy_net(non_final_next_state).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        return state_action_values, expected_state_action_values.unsqueeze(1)

    def update_target_net(self):
        pass