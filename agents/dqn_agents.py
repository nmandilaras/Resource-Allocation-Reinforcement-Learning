import copy
import random
import torch
import math
from utils.constants import *
from agents.agent import Agent


class DQNAgent(Agent):

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.999, eps_decay=0.0005, eps_start=1,
                 eps_end=0.01):
        super().__init__(num_of_actions, gamma, epsilon=1)
        self.device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device) seems slower with gpu
        self.policy_net = network.to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)  # if True else self.policy_net  # for simple Deep Q Learning
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # gradient updates never happens in target net
        self.criterion = criterion
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end

    def choose_action(self, state, train=True):
        if (random.random() < self.epsilon) and train:
            return random.randrange(self.num_of_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # we change max(1) to max(0) since we have only one element in this forward pass
                return self.policy_net(state).max(0)[1].item()

    def update(self, transitions, is_weights=None):
        """
        :param is_weights:
        :param transitions: a list whose elements are transitions
        """

        state, action, next_state, reward, done = list(
            map(lambda tensor: torch.tensor(tensor, device=self.device), zip(*transitions)))
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        predicted_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        # action_batch operates as index, unsqueezed so that each entry corresponds to one row

        # Compute V(s_{t+1}) for all next states.
        next_state_values = self._calc_expected_q(next_state)
        # Compute the expected Q values
        expected_q_values = reward + (1 - done.int()) * self.gamma * next_state_values
        # we want to take into account next states' values only if they are not final states
        pred = predicted_q_values.squeeze(1)
        target = expected_q_values.detach()
        loss = self.criterion(pred, target)
        if is_weights is not None:
            loss = torch.tensor(is_weights, device=self.device) * loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()  # computes gradients
        # for param in self.policy_net.parameters():  # it is used by the DQN paper, to improve stability
        #     # it clips the grads to be between -1 , 1
        #     param.grad.data.clamp_(-1, 1)           # but the paper applies this to the loss
        self.optimizer.step()  # updates weights
        # self.scheduler.step(loss)  # dynamicaly change the lr
        errors = torch.abs(pred - target).data.numpy()

        return loss, errors

    def _calc_expected_q(self, next_state):
        next_state_values = self.target_net(next_state).max(1)[0]

        return next_state_values

    def update_target_net(self):
        # We can also use  other techniques for target updating like Polyak Averaging
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def adjust_exploration(self, decaying_schedule):
        # TODO check VDBE-Softmax
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-decaying_schedule * self.eps_decay)
        # the update function is used by series of Deep RL

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass

    def train_mode(self):
        self.policy_net.train()

    def eval_mode(self):
        self.policy_net.eval()


class DoubleDQNAgent(DQNAgent):
    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.99, eps_decay=0.0005, eps_start=1,
                 eps_end=0.01):
        super().__init__(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)

    def _calc_expected_q(self, next_state):
        """ """
        policy_actions = self.policy_net(next_state).max(1)[1]
        next_state_values = self.target_net(next_state).gather(1, policy_actions.unsqueeze(1))

        return next_state_values.squeeze(1)

