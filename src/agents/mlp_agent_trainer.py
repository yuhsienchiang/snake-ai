import numpy as np
import torch
from torch import nn
from game.game import SnakeGame
from agents.mlp_agent import MLPAgent
from utils.utils import ReplayMemory


class MLPAgentTrainer(object):
    def __init__(
        self,
        game: SnakeGame,
        agent: MLPAgent,
        memory_size: int,
        loss_func_type: str,
        optimizer_type: str,
        model_learn_rate: float,
        discount_rate: float,
        epsilon: float,
    ) -> None:
        self.game = game
        self.agent = agent
        self.memory = ReplayMemory(memory_size=memory_size)
        self.discount_rate = discount_rate
        self.model_learn_rate = model_learn_rate
        self.epsilon = epsilon

        self.optimizer = self.select_optimizer(optimizer_type, model_learn_rate)
        self.loss_func = self.select_loss_func(loss_func_type)

        # set model to training model
        self.agent.main_net.train()
        self.agent.target_net.eval()

    def select_optimizer(self, optimizer_type, learning_rate):
        if optimizer_type == "adam":
            return torch.optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(self.agent.model.parameters(), lr=learning_rate)
        else:
            return None

    def select_loss_func(self, loss_func_type: str):
        if loss_func_type == "mse":
            return nn.MSELoss()
        elif loss_func_type == "huber":
            return nn.SmoothL1Loss()
        else:
            return None

    def train(self):
        # 1. obtain state
        state = self.agent.get_state()

        while True:
            # 2. get action
            # explore and exploit stratagey for obtaining action
            # can implement a better stratagey
            if np.random.rand() <= self.epsilon:  # explore
                action = self.agent.get_action(state=None, random_action=True)
            else:  # exploit
                action = self.agent.get_action(state=state, random_action=False)

            # 3.play action
            reward, game_over, _ = self.game.play_step(action=action)
            # 4. observe new state
            new_state = self.agent.get_state()

            self.memory.save(
                state=state,
                action=action,
                reward=reward,
                next_state=new_state,
                done=game_over,
            )

            # 5. move to next state
            state = new_state

            # 6. train model
            if len(self.memory) > 10:
                self.train_step(memory=self.memory)

            # 7. update target net
            # add condition for undate network
            self.update_target_net(
                target_net=self.agent.target_net, main_net=self.agent.main_net, tau=0.9
            )

    def train_step(self, memory: ReplayMemory):

        # 2. random sample memories
        batch_memory = memory.sample(500)
        
        # 3-1. current state data
        batch_state = batch_memory.state
        batch_action_idx = torch.argmax(batch_memory.action, dim=1, keepdim=True)
        batch_reward = batch_memory.reward
        batch_q_values = self.agent.main_net(batch_state).gather(1, batch_action_idx)

        # 3-2. next state info
        batch_non_done_next_state_mask = torch.tensor([done == 0 for done in batch_memory.done])
        batch_non_done_next_state =  batch_memory.next_state[batch_non_done_next_state_mask]
        with torch.no_grad():
            batch_non_done_next_q_values = self.agent.target_net(batch_non_done_next_state)


        # 4. calculate target q_values
        #
        # [Bellman Equation temperal difference]
        # Q(s, a) = Q(s, a) + lr * [reward + discount * max_a' Q(s', a') - Q(s, a)]
        # Approach 1
        # target : reward + discount * max_a' Q(s', a')
        # predict: Q(s, a)
        #
        # Approach 2
        # target : Q(s, a) + lr * [reward + discount * max_a' Q(s', a') - Q(s, a)]
        # predict: Q(s, a)
        #
        # [Implement]
        # non_done q values = reward
        #     done q values = reward + discount_rate * next_q_values
        batch_target_q_values = batch_reward.clone()
        batch_target_q_values[batch_non_done_next_state_mask] += batch_non_done_next_q_values * self.discount_rate

        # 5. Calculate loss
        loss = self.loss_func(batch_q_values, batch_target_q_values)

        # 6. Backpro
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self, target_net, main_net, tau: float):
        target_net_state = target_net.state_dict()
        main_net_state = main_net.state_dict()

        for key in main_net_state:
            target_net_state[key] = (
                tau * main_net_state[key] + (1 - tau) * target_net_state[key]
            )

        target_net.load_state_dict(target_net_state)

