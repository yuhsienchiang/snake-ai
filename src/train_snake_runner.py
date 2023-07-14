from game.game import SnakeGame
from agents.mlp_agent import MLPAgent
from agents.mlp_agent_trainer import MLPAgentTrainer


if __name__ == "__main__":
    snakeGame = SnakeGame(width=1200, height=900)
    mlpAgent = MLPAgent(snakeGame)

    mlpAgentTrainer = MLPAgentTrainer(game=snakeGame,
                                      agent=mlpAgent,
                                      batch_size=16,
                                      memory_size=10000,
                                      loss_func_type="huber",
                                      optimizer_type="AdamW",
                                      model_learn_rate=0.0001,
                                      discount_rate=0.99,
                                      epsilon=0.005)
    
    mlpAgentTrainer.train(episodes_num=50)
    print("train done")
