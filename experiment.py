import time
from fsdp import FSDPExecutor
from mnist import MNISTSpec


def main():
    m = MNISTSpec()
    datapath = "../data"
    hyperparams = {
        "lr": m.param_grid["lr"][1],
        "batch_size": m.param_grid["batch_size"][0]
    }
    model_path = "../models/model1.pt"

    f = FSDPExecutor(m.train, datapath, hyperparams, model_path)
    f.execute_train()
    f.cleanUp()


if __name__ == "__main__":
    main()
