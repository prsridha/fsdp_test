from fsdp import FSDPExecutor
from mnist import MNISTSpec


def main():
    m = MNISTSpec()
    datapath = "data"
    hyperparams = {
        "lr": m.param_grid["lr"][1]
    }
    model_path = "models"

    f = FSDPExecutor(m.train, datapath, hyperparams, model_path)
    f.execute_train()


if __name__ == "__main__":
    main()
