---
title: "从 PyTorch DDP 到 Accelerate 到 Trainer，轻松掌握分布式训练"
thumbnail: /blog/assets/111_pytorch_ddp_accelerate_transformers/thumbnail.png
authors:
- user: muellerzr
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 从 PyTorch DDP 到 Accelerate 到 Trainer，轻松掌握分布式训练


## 概述

本教程假定你已经对于 PyToch 训练一个简单模型有一定的基础理解。本教程将展示使用 3 种封装层级不同的方法调用 DDP (DistributedDataParallel) 进程，在多个 GPU 上训练同一个模型：

- 使用 `pytorch.distributed` 模块的原生 PyTorch DDP 模块
- 使用 🤗 Accelerate 对 `pytorch.distributed` 的轻量封装，确保程序可以在不修改代码或者少量修改代码的情况下在单个 GPU 或 TPU 下正常运行
- 使用 🤗 Transformer 的高级 Trainer API ，该 API 抽象封装了所有代码模板并且支持不同设备和分布式场景。

## 什么是分布式训练，为什么它很重要？

下面是一些非常基础的 PyTorch 训练代码，它基于 Pytorch 官方在 MNIST 上创建和训练模型的 [示例](https://github.com/pytorch/examples/blob/main/mnist/main.py)。


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

我们定义训练设备 (`cuda`):

```python
device = "cuda"
```

构建一些基本的 PyTorch DataLoaders:

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

把模型放入 CUDA 设备:

```python
model = BasicNet().to(device)
```

构建 PyTorch optimizer (优化器):

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

最终创建一个简单的训练和评估循环，训练循环会使用全部训练数据集进行训练，评估循环会计算训练后模型在测试数据集上的准确度：

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
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
```

通常从这里开始，就可以将所有的代码放入 Python 脚本或在 Jupyter Notebook 上运行它。

然而，只执行 `python myscript.py` 只会使用单个 GPU 运行脚本。如果有多个 GPU 资源可用，您将如何让这个脚本在两个 GPU 或多台机器上运行，通过 *分布式* 训练提高训练速度？这是 `torch.distributed` 发挥作用的地方。

## PyTorch 分布式数据并行

顾名思义，`torch.distributed` 旨在配置分布式训练。你可以使用它配置多个节点进行训练，例如：多机器下的单个 GPU，或者单台机器下的多个 GPU，或者两者的任意组合。

为了将上述代码转换为分布式训练，必须首先定义一些设置配置，具体细节请参阅 [DDP 使用教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)。

首先必须声明 `setup` 和 `cleanup` 函数。这将创建一个进程组，并且所有计算进程都可以通过这个进程组通信。

> 注意：在本教程的这一部分中，假定这些代码是在 Python 脚本文件中启动。稍后将讨论使用 🤗 Accelerate 的启动器，就不必声明 `setup` 和 `cleanup` 函数了。

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

最后一个疑问是，*我怎样把我的数据和模型发送到另一个 GPU 上？*

这正是 `DistributedDataParallel` 模块发挥作用的地方， 它将您的模型复制到每个 GPU 上 ，并且当 `loss.backward()` 被调用进行反向传播的时候，所有这些模型副本的梯度将被同步地平均/下降 (reduce)。这确保每个设备在执行优化器步骤后具有相同的权重。

下面是我们的训练设置示例，我们使用了 DistributedDataParallel 重构了训练函数：

> 注意：此处的 rank 是当前 GPU 与所有其他可用 GPU 相比的总体 rank，这意味着它们的 rank 为 `0 -> n-1`

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

在上述的代码中需要为 *每个副本设备上* 的模型 (因此在这里是 `ddp_model` 的参数而不是 `model` 的参数) 声明优化器，以便正确计算每个副本设备上的梯度。

最后，要运行脚本，PyTorch 有一个方便的 `torchrun` 命令行模块可以提供帮助。只需传入它应该使用的节点数以及要运行的脚本即可：

```bash
torchrun --nproc_per_nodes=2 --nnodes=1 example_script.py
```

上面的代码可以在在一台机器上的两个 GPU 上运行训练脚本，这是使用 PyTorch 只进行分布式训练的情况 (不可以在单机单卡上运行)。

现在让我们谈谈 🤗 Accelerate，一个旨在使并行化更加无缝并有助于一些最佳实践的库。

## 🤗 Accelerate

[Accelerate](https://huggingface.co/docs/accelerate) 是一个库，旨在无需大幅修改代码的情况下完成并行化。除此之外，🤗 Accelerate 附带的数据 pipeline 还可以提高代码的性能。

首先，让我们将刚刚执行的所有上述代码封装到一个函数中，以帮助我们直观地看到差异：

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

接下来让我们谈谈 🤗 Accelerate 如何便利地实现并行化的。上面的代码有几个问题：

1. 该代码有点低效，因为每个设备都会创建一个 dataloader。
2. 这些代码**只能**运行在多 GPU 下，当想让这个代码运行在单个 GPU 或 TPU 时，还需要额外进行一些修改。

Accelerate 通过 [`Accelerator`](https://huggingface.co/docs/accelerate/v0.12.0/en/package_reference/accelerator#accelerator) 类解决上述问题。通过它，不论是单节点还是多节点，除了三行代码外，其余代码几乎保持不变，如下所示：

```python
def train_ddp_accelerate():
    accelerator = Accelerator()
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
        accelerator.backward(loss)
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

借助 `Accelerator` 对象，您的 PyTorch 训练循环现在已配置为可以在任何分布式情况运行。使用 Accelerator 改造后的代码仍然可以通过 `torchrun` CLI 或通过 🤗 Accelerate 自己的 CLI 界面 [启动](https://huggingface.co/docs/accelerate/v0.12.0/en/basic_tutorials/launch) (启动你的🤗 Accelerate 脚本)。

因此，现在可以尽可能保持 PyTorch 原生代码不变的前提下，使用 🤗 Accelerate 执行分布式训练。

早些时候有人提到 🤗 Accelerate 还可以使 DataLoaders 更高效。这是通过自定义采样器实现的，它可以在训练期间自动将部分批次发送到不同的设备，从而允许每个设备只需要储存数据的一部分，而不是一次将数据复制四份存入内存，具体取决于配置。因此，内存总量中只有原始数据集的一个完整副本。该数据集会拆分后分配到各个训练节点上，从而允许在单个实例上训练更大的数据集，而不会使内存爆炸。

### 使用 `notebook_launcher`

之前提到您可以直接从 Jupyter Notebook 运行分布式代码。这来自 🤗 Accelerate 的 [`notebook_launcher`](https://huggingface.co/docs/accelerate/v0.12.0/en/basic_tutorials/notebook) 模块，它可以在 Jupyter Notebook 内部的代码启动多 GPU 训练。

使用它就像导入 launcher 一样简单：

```python
from accelerate import notebook_launcher
```

接着传递我们之前声明的训练函数、要传递的任何参数以及要使用的进程数（例如 TPU 上的 8 个，或两个 GPU 上的 2 个）。下面两个训练函数都可以运行，但请注意，启动单次启动后，实例需要重新启动才能产生另一个：

```python
notebook_launcher(train_ddp, args=(), num_processes=2)
```

或者：

```python
notebook_launcher(train_accelerate_ddp, args=(), num_processes=2)
```

## 使用 🤗 Trainer

终于我们来到了最高级的 API——Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)。

它涵盖了尽可能多的训练类型，同时仍然能够在分布式系统上进行训练，用户根本不需要做任何事情。

首先我们需要导入 Trainer：

```python
from transformers import Trainer
```

然后我们定义一些 `TrainingArguments` 来控制所有常用的超参数。Trainer 需要的训练数据是字典类型的，因此需要制作自定义整理功能。

最后，我们将训练器子类化并编写我们自己的 `compute_loss`。

之后，这段代码也可以分布式运行，而无需修改任何训练代码！

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

与上面的 `notebook_launcher` 示例类似，也可以将这个过程封装成一个训练函数：

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

## 相关资源

要了解有关 PyTorch 分布式数据并行性的更多信息，请查看 [文档](https://pytorch.org/docs/stable/distributed.html)

要了解有关 🤗 Accelerate 的更多信息，请查看 [🤗 Accelerat 文档](https://huggingface.co/docs/accelerate)

要了解有关 🤗 Transformer 的更多信息，请查看 [🤗 Transformer 文档](https://huggingface.co/docs/transformers)
