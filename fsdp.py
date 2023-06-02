import os

import torch
import functools
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.distributed.fsdp import CPUOffload
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from parallelism import Parallelism

# from pytorch_data import GeneralPytorchDataset
# TODO: how do handle GeneralPytorchDataset - ditch Sampler, add "gpu" dir names in etl, so that it is already sharded
# TODO: multiple models - where to pass it - function callbacks
# TODO: stuff like stepLR - that should happen at the end of every epoch - where does this go?


class FSDPExecutor(Parallelism):
    def __init__(self, user_train_func, datapath, hyperparams, model_path) -> None:
        super().__init__()
        self.name = "FSDPExecutor"
        self.datapath = datapath
        self.model_path = model_path
        self.hyperparams = hyperparams
        self.user_train_func = user_train_func
        self.world_size = torch.cuda.device_count()

        # set optimization parameters
        self.wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100
        )
        self.cpu_offload = CPUOffload(offload_params=True)

    def initialize(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanUp(self):
        dist.destroy_process_group()

    def parallelize(self, model):
        fsdp_model = FSDP(model, auto_wrap_policy=FSDPExecutor.wrap_policy, cpu_offload=FSDPExecutor.cpu_offload, device_id=torch.cuda.current_device())
        apply_activation_checkpointing(
            fsdp_model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda l: isinstance(l, FSDP))

        return fsdp_model

    def save_checkpoint(self, states, rank):
        dist.barrier()
        if rank == 0:
            torch.save(states, self.model_path)

    def logger(self, metric, rank):
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        if rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(1, metric[0] / metric[1]))

    def sample(self, model, train_func, data_path):
        pass

    def _train(self, rank):
        self.initialize(rank, self.world_size)
        torch.cuda.set_device(rank)
        # data_loader = GeneralPytorchDataset("train", self.datapath)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('../data', train=True, transform=transform)
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=self.world_size, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams["batch_size"], sampler=sampler, pin_memory=True, shuffle=False)

        self.user_train_func(self.parallelize, self.save_checkpoint, self.model_filepath, dataloader, self.hyperparams, rank)

    def _test(self):
        pass

    def execute_train(self):
        mp.spawn(self._train, nprocs=self.world_size, join=True)

    def execute_test(self):
        mp.spawn(self._test, nprocs=self.world_size, join=True)
