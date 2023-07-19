from game.game import SnakeGame
from agents.mlp_agent import MLPAgent
from agents.mlp_agent_trainer import MLPAgentTrainer


if __name__ == "__main__":
    snakeGame = SnakeGame(width=1200, height=900)
    mlpAgent = MLPAgent(snakeGame)

    mlpAgentTrainer = MLPAgentTrainer(game=snakeGame,
                                      agent=mlpAgent,
                                      batch_size=128,
                                      memory_size=10000,
                                      loss_func_type="huber",
                                      optimizer_type="AdamW",
                                      model_learn_rate=0.0001,
                                      discount_rate=0.99,
                                      epsilon_start=0.9,
                                      epsilon_end=0.05,
                                      epsilon_decay=1000.0)
    
    mlpAgentTrainer.train(episodes_num=100)
    print("train done")
