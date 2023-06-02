import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from spec import TrainingSpec


class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MNISTSpec(TrainingSpec):
    def __init__(self) -> None:
        self.num_epoch = 2
        self.param_grid = {
            "lr": ["1.0", "0.5"]
        }
        self.model = mnist_model()
        self.optimizer = optim.Adadelta

    def initialize_worker(self):
        pass

    def train(self, parallelize, save_checkpoint, model_filepath, dataloader, hyperparams, device, logger):
        model = self.Net().to(device)
        model = parallelize(model)
        optimizer = optim.Adadelta(model.parameters(), lr=hyperparams["lr"])

        if os.path.isfile(model_filepath):
            checkpoint = torch.load(model_filepath)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        model.train()
        ddp_loss = torch.zeros(2).to(device)

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            loss.backward()
            optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

        logger(ddp_loss, device)

        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, device)
