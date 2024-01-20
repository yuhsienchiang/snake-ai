# snake-ai

## Description
This project aims to explore the ability of various AI models to play the timeless Snake game.
The primary objective is to employ diverse AI models and assess their performance, drawing comparisons to discern their respective strengths and weaknesses.

## Setup
1. Create and activate virtual env with pyenv
    ```bash
    pyenv virtualenv 3.10.12 snake_ai
    pyenv local snake_ai
    ```
2. install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Play the Game
To train an agent:
```bash
python src/train_snake_runner.py
```
You can also play the game yourself:
```bash
python src/snake_runner.py
```

## Reference
- https://github.com/patrickloeber/snake-ai-pytorch
- https://github.com/linyiLYi/snake-ai
