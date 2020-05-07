import torch
from agents.agent import Agent
from torch.distributions import Categorical
import torch.nn.functional as F


class Reinforce(Agent):

    def __init__(self, num_of_actions, network, optimizer, gamma=0.999, gpu=False):
        """

        """
        super().__init__(num_of_actions, gamma)
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        # print(self.device) seems slower with gpu
        self.policy_net = network.to(self.device)
        self.optimizer = optimizer

    def choose_action(self, state, train=True):
        """  """
        state = torch.tensor(state, device=self.device)
        probs = F.softmax(self.policy_net(state), dim=-1)  # each element of probs is the relative probability of sampling the class at that index
        m = Categorical(probs)  # Creates a categorical distribution parameterized by probs
        if train:
            action = m.sample()  # we choose an action based on their probability
        else:
            action = probs.argmax()  # mine addition, it cancels the stochastic nature of the algorithm but cartpole is a deterministic env
        log_prob = m.log_prob(action)
        return action.item(), log_prob, probs.max()  # action is a tensor so we return just the number

    def update(self, log_probs, discounted_rewards):
        """  """
        policy_loss = []
        # print(log_probs)
        # print(discounted_rewards)
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * discounted_reward)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss

    def calculate_rewards(self, rewards):
        """ We start from the end of each episode and we calculate the discounted rewards. Moreover we standardize
         the rewards. By doing this we’re always encouraging and discouraging roughly half of the performed actions"""

        discounted_rewards = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # the standardization doesn't seems to have a great effect, but it is used by all tutorials

        return discounted_rewards

    def train_mode(self):
        self.policy_net.train()

    def eval_mode(self):
        self.policy_net.eval()

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass
