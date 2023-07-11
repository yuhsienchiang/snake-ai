from itertools import count
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
        batch_size: int = 128,
        memory_size: int = 10000,
        loss_func_type: str = "huber",
        optimizer_type: str = "AdamW",
        model_learn_rate: float = 1e-4,
        discount_rate: float = 0.99,
        epsilon: float = 0.005,
    ) -> None:
        self.game = game
        self.agent = agent
        self.batch_size = batch_size
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
        if optimizer_type == "Adam":
            return torch.optim.Adam(self.agent.main_net.parameters(), lr=learning_rate)
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(self.agent.main_net.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(self.agent.main_net.parameters(), lr=learning_rate)
        else:
            return None

    def select_loss_func(self, loss_func_type: str):
        if loss_func_type == "mse":
            return nn.MSELoss()
        elif loss_func_type == "huber":
            return nn.SmoothL1Loss()
        else:
            return None

    def train(self, episodes_num: int = 50):
        training_steps = 0

        for _ in range(episodes_num):
            # 1. initialise the game env
            self.game.reset()
            # 2. get initial state
            state = self.agent.get_state()
            done = False

            while done is False:
                training_steps += 1

                # 3. get action
                # explore and exploit stratagey for obtaining action
                # can implement a better stratagey
                if np.random.rand() <= self.epsilon:  # explore
                    action = self.agent.get_action(state=None, random_action=True)
                else:  # exploit
                    action = self.agent.get_action(state=state, random_action=False)

                # 4.play action
                reward, done, _ = self.game.play_step(action=action)
                # 5. observe new state
                new_state = self.agent.get_state()

                # 6. save transition info in memory
                self.memory.save(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=new_state,
                    done=done,
                )

                # 7. move to next state
                state = new_state

                # 8. train model every 4 steps
                if training_steps % 4 == 0 or done:
                    self.train_step(memory=self.memory)

                # 9. update target net every 50 steps
                if training_steps % 50 == 0 or done:
                    self.update_target_net(
                        target_net=self.agent.target_net,
                        main_net=self.agent.main_net,
                        tau=0.9,
                    )

    def train_step(self, memory: ReplayMemory):
        if len(memory) < self.batch_size:
            return

        # 1. random sample memories
        batch_memory = memory.sample(self.batch_size)

        # 2-1. current state data
        batch_state = batch_memory.state
        batch_action_idx = torch.argmax(batch_memory.action, dim=1, keepdim=True)
        batch_reward = batch_memory.reward
        batch_q_values = self.agent.main_net(batch_state).gather(1, batch_action_idx)

        # 2-2. next state info
        batch_non_done_next_state_mask = torch.tensor(
            [done == 0 for done in batch_memory.done]
        )
        batch_non_done_next_state = batch_memory.next_state[
            batch_non_done_next_state_mask
        ]
        with torch.no_grad():
            batch_non_done_next_q_values = self.agent.target_net(
                batch_non_done_next_state
            )

        # 3. calculate target q_values
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
        batch_target_q_values[batch_non_done_next_state_mask] += (
            self.discount_rate * batch_non_done_next_q_values
        )

        # 4. Calculate loss
        loss = self.loss_func(batch_q_values, batch_target_q_values)

        # 5. Backpropagation update netword parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self, target_net, main_net, tau: float = 0.9):
        target_net_state = target_net.state_dict()
        main_net_state = main_net.state_dict()

        for key in main_net_state:
            target_net_state[key] = (
                tau * main_net_state[key] + (1 - tau) * target_net_state[key]
            )

        target_net.load_state_dict(target_net_state)
