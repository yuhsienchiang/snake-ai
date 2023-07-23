from game.game import SnakeGame
from agents.mlp_agent import MLPAgent
from agents.mlp_agent_trainer import MLPAgentTrainer


if __name__ == "__main__":
    snakeGame = SnakeGame(width=1200, height=900)
    mlpAgent = MLPAgent(snakeGame)

    mlpAgentTrainer = MLPAgentTrainer(game=snakeGame,
                                      agent=mlpAgent,
                                      memory_size=10000,
                                      loss_func_type="huber",
                                      optimizer_type="AdamW",
                                      model_learn_rate=0.0001,
                                      discount_rate=0.95,
                                      epsilon_start=0.9,
                                      epsilon_end=0.05,
                                      epsilon_decay=5000.0)
    
    mlpAgentTrainer.train(episodes_num=500, batch_size=192)
    snakeGame.end_game_ui(title="Train Done!", score=max(mlpAgentTrainer.score_records))
    print("train done")
    print(f"best score = {max(mlpAgentTrainer.score_records)}")
