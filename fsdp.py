import os
import warnings

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
    checkpoint_wrapper
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

        warnings.filterwarnings("ignore", "torch.distributed._all_gather_base is a private function")
        warnings.filterwarnings("ignore", "torch.distributed._reduce_scatter_base is a private function")

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanUp(self):
        try:
            dist.destroy_process_group()
        except AssertionError as e:
            print("Couldn't clean up: ", e)

    def parallelize(self, model):
        # fsdp_model = FSDP(model, fsdp_auto_wrap_policy=self.wrap_policy, cpu_offload=self.cpu_offload)
        fsdp_model = FSDP(model)
        return fsdp_model

    def checkpoint(self, states, rank):
        dist.barrier()
        if rank == 0:
            torch.save(states, self.model_path)

    def metrics_logger(self, metrics, rank):
        for k, v in metrics.items():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            if rank == 0:
                print('{}:\nEpoch: {} \tLoss: {:.6f}'.format(k, 1, v[0] / v[1]))

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

        checkpoint_func = functools.partial(self.checkpoint, rank=rank)
        logger_func = functools.partial(self.metrics_logger, rank=rank)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        self.user_train_func(self.parallelize, checkpoint_func, self.model_path, dataloader, self.hyperparams, rank, logger_func)
        end_event.record()

        if rank == 0:
            print(f"CUDA event elapsed time: {start_event.elapsed_time(end_event) / 1000}sec")

    def _test(self):
        pass

    def execute_train(self):
        mp.spawn(self._train, nprocs=self.world_size, join=True)

    def execute_test(self):
        mp.spawn(self._test, nprocs=self.world_size, join=True)
