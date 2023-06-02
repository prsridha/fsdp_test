class TrainingSpec:
    def __init__(self) -> None:
        self.num_epoch: int = None
        self.param_grid: dict() = None
        self.models: list = None
        self.optimizer = None

    def initialize_worker(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
