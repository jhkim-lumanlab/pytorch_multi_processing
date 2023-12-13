# basic usage of PyTorch's Distributed Data Parallel (DDP) for training a simple model across multiple GPUs, with each process running on a separate GPU

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(1000, 10))
        labels = torch.randn(1000, 5).to(rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 
        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    # start time
    start_time = time.time()
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus//2 # set the number of processes you want to run
    run_demo(demo_basic, world_size)
    # end time
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")