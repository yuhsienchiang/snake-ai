import numpy as np
import torch
import math
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
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 1000.0,
    ) -> None:
        self.game = game
        self.agent = agent
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size=memory_size)
        self.discount_rate = discount_rate
        self.model_learn_rate = model_learn_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.optimizer = self.select_optimizer(optimizer_type, model_learn_rate)
        self.loss_func = self.select_loss_func(loss_func_type)

        # set model to training model
        self.agent.main_net.train()
        self.agent.target_net.eval()
        self.device = self.agent.main_net.device

        self.score_records = [0]

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

        for idx_episodes in range(episodes_num):
            print(f"episode: {idx_episodes}")
            # 1. initialise the game env
            state = self.game.reset()
            done = False

            while done is False:
                training_steps += 1

                # 3. get action
                # explore and exploit stratagey for obtaining action
                # can implement a better stratagey
                eps_threshold = self.epsilon_end + (
                    self.epsilon_start - self.epsilon_end
                ) * math.exp(-1.0 * training_steps / self.epsilon_decay)
                if np.random.rand() <= eps_threshold:  # explore
                    action = self.agent.get_action(state=None)
                else:  # exploit
                    action = self.agent.get_action(state=state)

                # 4.play action
                next_state, reward, done, score = self.game.play_step(action=action)
                # 5. observe new state

                # 6. save transition info in memory
                self.memory.save(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                # 7. move to next state
                state = next_state

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

                # 10. store episode score and save model if is best record
                if done:
                    self.score_records.append(score)
                    if score > max(self.score_records):
                        self.agent.main_net.save_model()

    def train_step(self, memory: ReplayMemory):
        if len(memory) < self.batch_size:
            return

        # 1. random sample memories
        batch_memory = memory.sample(self.batch_size)

        # 2-1. current state data
        batch_state = batch_memory.state.to(self.device)
        batch_action_idx = torch.argmax(batch_memory.action, dim=1, keepdim=True).to(
            self.device
        )
        batch_reward = batch_memory.reward.to(self.device)
        batch_q_values = self.agent.main_net(batch_state).gather(1, batch_action_idx)

        # 2-2. next state info
        batch_non_done_next_state_mask = torch.tensor(
            [done == 0 for done in batch_memory.done], device=self.device
        )
        batch_non_done_next_state = batch_memory.next_state.to(self.device)[
            batch_non_done_next_state_mask
        ]
        with torch.no_grad():
            batch_non_done_next_q_values = torch.max(
                self.agent.target_net(batch_non_done_next_state), dim=1, keepdim=True
            )[0]

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
        batch_target_q_values = (
            batch_reward.clone().reshape_as(batch_q_values).to(self.device)
        )
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
