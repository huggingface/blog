# From PyTorch DDP to Accelerate to Trainer, mastery of distributed training with ease

Authors:

- Zachary Mueller (Hugging Face)
- Less Wright (PyTorch)

## General Overview

This tutorial assumes you have a basic understanding of PyTorch and how to train a simple model. It will showcase training on multiple GPUs through a process called Distributed Data Parallelism (DDP) through three different layers:

- Native PyTorch DDP through the `pytorch.distributed` module
- Utilizing 🤗 Accelerate's light wrapper around `pytorch.distributed` that also helps ensure the code can be run on a single GPU and TPUs with zero code changes and miminimal code changes to the original code
- Utilizing 🤗 Transformer's trainer, which is a high level wrapping API to perform a similar result to Accelerate

## What is "Distributed" training and why does it matter?

Take some very basic PyTorch training code below, which sets up and trains a model on MNIST based on the [official MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

We define the training device (`cuda`):

```python
device = "cuda"
```

Build some PyTorch DataLoaders:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)
```

Move the model to the CUDA device:

```python
model = BasicNet().to(device)
```

Build a PyTorch optimizer:

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

Before finally creating a simplistic training and evaluation loop that performs one full iteration over the dataset and calculates the test accuracy:

```python
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
```

Typically from here, one could either throw all of this into a python script or run it on a Jupyter Notebook.

However, how would you then get this to run on say two GPUs if these resources available? Just doing `python myscript.py` will only ever run the script using a single GPU. This is where `torch.distributed` comes into play

## PyTorch Distributed Data Parallelism

As the name implies, `torch.distributed` is meant to work on *distributed* setups. This can include multi-node, where you have a number of machines each with a single GPU, or multi-gpu where a single system has multiple GPUs, or some combination of both.

To convert our above code to work within a distributed setup, a few setup configurations must first be defined, detailed in the [Getting Started with DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

First a `setup` and a `cleanup` function must be declared. This will open up a processing group that all of the compute processes can communicate through

> Note: for this section of the tutorial it should be assumed these are sent in python script files. Later on a launcher using Accelerate will be discussed that removes this necessity

```python
import os
import torch.distributed as dist

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()
```

The last piece of the puzzle is *how do I send my data and model to another GPU?*

This is where the `DistributedDataParallel` module comes into play. It will copy your model into each GPU, and when `loss.backward()` is called the gradients across all these copies of the model will be averaged, backprop is performed, and each of the weights get updated to have the same weights across all devices.

Below is an example of our training setup, refactored as a function, with this capability:

> Note: Here rank is the overall rank of the current GPU compared to all the other GPUs available, meaning they have a rank of `0 -> n-1`

```python
from torch.nn.parallel import DistributedDataParallel as DDP

def train(model, rank, world_size):
    setup(rank, world_size)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    cleanup()
```

The optimizer needs to be declared based on the model *on the specific device* (so `ddp_model` and not `model`) for all of the gradients to properly be calculated.

Lastly, to run the script PyTorch has a convenient `torchrun` command line module that can help. Just pass in the number of nodes it should use as well as the script to run and you are set:

```bash
torchrun --nproc_per_nodes=2 --nnodes=1 example_script.py
```

The above will run the training script on two GPUs that live on a single machine and this is the barebones for performing only distributed training with PyTorch.

Now let's talk about Accelerate, a library aimed to make this process more seameless and also help with a few best practices

## 🤗 Accelerate

[Accelerate](https://huggingface.co/docs/accelerate) is a library designed to allow you to perform what we just did above, without needing to modify your code greatly. On top of this, the data pipeline innate to Accelerate can also improve performance to your code as well.

First, let's wrap all of the above code we just performed into a single function, to help us visualize the difference:

```python
def train_ddp(rank, world_size):
    setup(rank, world_size)
    # Build DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

    # Build model
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Build optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # Train for a single epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
```

Next let's talk about how Accelerate can help. There's a few issues with the above code:

1. This is slightly inefficient, given that `n` dataloaders are made based on each device and pushed.
2. This code will **only** work for multi-GPU, so special care would need to be made for it to be ran on a single node again, or on TPU.

Accelerate helps this through the [`Accelerator`](https://huggingface.co/docs/accelerate/v0.12.0/en/package_reference/accelerator#accelerator) class. Through it, the code remains much the same except for three lines of code when comparing a single node to multinode, as shown below:

```python
def train_ddp_accelerate():
    accelerator = Accelerator(device_placement=True)
    # Build DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

    # Build model
    model = BasicModel()

    # Build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Send everything through `accelerator.prepare`
    train_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, test_loader, model, optimizer
    )

    # Train for a single epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
```

Now any and all custom code needed to launch your code on any distributed setup is simplified to the `Accelerator` object. This code can then still be ran through the `torchrun` CLI, or through Accelerate's own CLI interface, [`accelerate launch`](https://huggingface.co/docs/accelerate/v0.12.0/en/basic_tutorials/launch).

As a result its now trivialized to perform distributed training with Accelerate and keeping as much of the barebones PyTorch code the same as possible.

Earlier it was mentioned that Accelerate also makes the DataLoaders more efficient. This is through custom Samplers that will send parts of the batches automatically to different devices during training allowing for a single copy of the data to be known at one time, rather than four at once into memory.

### Using the `notebook_launcher`

Earlier it was mentioned you can start distributed code directly out of your Jupyter Notebook. This comes from Accelerate's [`notebook_launcher`](https://huggingface.co/docs/accelerate/v0.12.0/en/basic_tutorials/notebook) utility, which allows for starting multi-gpu training based on code inside of a Jupyter Notebook.

To use it is as trivial as importing the launcher:

```python
from accelerate import notebook_launcher
```

And passing the training function we declared earlier, any arguments to be passed, and the number of processes to use (such as 8 on a TPU, or 2 for two GPUs). Both of the above training functions can be ran, but do note that after you start a single launch, the instance needs to be restarted before spawning another

```python
notebook_launcher(train_ddp, args=(), num_processes=2)
```

Or:

```python
notebook_launcher(train_accelerate_ddp, args=(), num_processes=2)
```

## Using 🤗 Trainer

Finally, we arrive at the highest level of API -- the Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

This wraps as much training as possible while still being able to train on distributed systems without the user needing to do anything at all.

First we need to import the Trainer:

```python
from transformers import Trainer
```

Then we define some `TrainingArguments` to control all the usual hyper-parameters. The trainer also works through dictionaries, so a custom collate function needs to be made.

Finally, we subclass the trainer and write our own `compute_loss`.

Afterwards, this code will also work on a distributed setup without any training code needing to be written whatsoever!

```python
from transformers import Trainer, TrainingArguments

model = BasicNet()

training_args = TrainingArguments(
    "basic-trainer",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    remove_unused_columns=False
)

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x":pixel_values, "labels":labels}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["x"])
        target = inputs["labels"]
        loss = F.nll_loss(outputs, target)
        return (loss, outputs) if return_outputs else loss

trainer = MyTrainer(
    model,
    training_args,
    train_dataset=train_dset,
    eval_dataset=test_dset,
    data_collator=collate_fn,
)
```

```python
trainer.train()
```

```python out
    ***** Running training *****
      Num examples = 60000
      Num Epochs = 1
      Instantaneous batch size per device = 64
      Total train batch size (w. parallel, distributed & accumulation) = 64
      Gradient Accumulation steps = 1
      Total optimization steps = 938
```

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 0.875700      | 0.282633        |

Similarly to the above examples with the `notebook_launcher`, this can be done again here by throwing it all into a training function:

```python
def train_trainer_ddp():
    model = BasicNet()

    training_args = TrainingArguments(
        "basic-trainer",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        remove_unused_columns=False
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"x":pixel_values, "labels":labels}

    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(inputs["x"])
            target = inputs["labels"]
            loss = F.nll_loss(outputs, target)
            return (loss, outputs) if return_outputs else loss

    trainer = MyTrainer(
        model,
        training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        data_collator=collate_fn,
    )

    trainer.train()

notebook_launcher(train_trainer_ddp, args=(), num_processes=2)
```

## Resources

To learn more about PyTorch Distributed Data Parallelism, check out the documentation [here](https://pytorch.org/docs/stable/distributed.html)

To learn more about 🤗 Accelerate, check out the documentation [here](https://huggingface.co/docs/accelerate)

To learn more about 🤗 Transformers, check out the documentation [here](https://huggingface.co/docs/transformers)
