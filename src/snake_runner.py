from time import sleep
from game.game import SnakeGame
from agents.human import Human
from utils.motion import Action


if __name__ == "__main__":
    snakeGame = SnakeGame(width=1200, height=900)
    humanAgent = Human(game=snakeGame)

    game_over = False
    score = 0

    while game_over is False:
        action = humanAgent.get_action(state=None)

        if action == Action.QUIT:
            break

        reward, game_over, score = snakeGame.play_step(action=action)

    # draw ending screen
    snakeGame.end_game_ui(title="Game Over!", score=score)
    sleep(1.5)

    snakeGame.pygame.quit()
    print(f"Final Score: {score}")
