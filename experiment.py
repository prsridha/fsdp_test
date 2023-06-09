# from fsdp import FSDPExecutor
from ddp import DDPExecutor
from mnist import MNISTSpec


def main():
    m = MNISTSpec()
    datapath = "../data"
    hyperparams = {
        "lr": m.param_grid["lr"][1],
        "batch_size": m.param_grid["batch_size"][0]
    }
    model_path = "../models/model1.pt"

    f = DDPExecutor(m.train, datapath, hyperparams, model_path)
    f.execute_train()


if __name__ == "__main__":
    main()
