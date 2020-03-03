import copy
import torch
from agents.deep_agent import DeepAgent


class DDQNAgent(DeepAgent):
    def __init__(self, num_of_actions, network, criterion, optimizer, mem_size=1000, batch_size=32, gamma=0.999, epsilon=1):
        super().__init__(num_of_actions, network, criterion, optimizer, mem_size, batch_size, gamma, epsilon)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # gradient updates never happens in target net

    def compute_loss(self, state_batch, action_batch, reward_batch, non_final_next_state, non_final_mask):

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        # action_batch operates as index, unsqueezed so that each entry corresponds to one row

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        return state_action_values, expected_state_action_values.unsqueeze(1)

    def update_target_net(self):
        # We can also use  other techniques for target updating like Polyak Averaging
        self.target_net.load_state_dict(self.policy_net.state_dict())
