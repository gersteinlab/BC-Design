# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer

# # Dummy Dataset
# x = torch.randn(1000, 10)  # 1000 samples, 10 features each
# y = torch.randint(0, 2, (1000, 1), dtype=torch.float32)  # Binary labels
# dataset = TensorDataset(x, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Simple Neural Network Model
# class SimpleModel(pl.LightningModule):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.layer_1 = nn.Linear(10, 64)
#         self.layer_2 = nn.Linear(64, 1)
#         self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, x):
#         x = torch.relu(self.layer_1(x))
#         x = self.layer_2(x)
#         return x

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         return loss

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

# # Initialize Model
# model = SimpleModel()

# # Train on GPU 0 and 1
# trainer = Trainer(
#     max_epochs=5,
#     devices=[0, 1],  # Specify the GPUs you want to use
#     accelerator='gpu'
# )

# # Start training
# trainer.fit(model, dataloader)



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# Dummy Dataset
x = torch.randn(1000, 10)  # 1000 samples, 10 features each
y = torch.randint(0, 2, (1000, 1), dtype=torch.float32)  # Binary labels
dataset = TensorDataset(x, y)

# Simple Neural Network Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer_1 = nn.Linear(10, 64)
        self.layer_2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

# Training Loop
def train(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_P2P_DISABLE'] = '1'

    # Setup distributed training
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Create model and move it to GPU with id rank
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # DataLoader with Distributed Sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    # Training
    for epoch in range(5):  # 5 epochs
        sampler.set_epoch(epoch)  # shuffle dataset differently at each epoch
        for batch in dataloader:
            x, y = batch
            x = x.to(rank)
            y = y.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
