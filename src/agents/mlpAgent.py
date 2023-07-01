from models.mlp import MLP_QNet


class MLPAgemt(object):
    def __init__(self) -> None:
        self.model = MLP_QNet()

    def get_state(self):
        pass

    def get_action(self, state):
        pass
