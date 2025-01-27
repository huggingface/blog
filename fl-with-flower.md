---
title: "Federated Learning using Hugging Face and Flower" 
thumbnail: /blog/assets/fl-with-flower/thumbnail.png
authors:
- user: charlesbvll
  guest: true
---

# Federated Learning using Hugging Face and Flower


<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/fl-with-flower.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This tutorial will show how to leverage Hugging Face to federate the training of language models over multiple clients using [Flower](https://flower.ai/). More specifically, we will fine-tune a pre-trained Transformer model (distilBERT) for sequence classification over a dataset of IMDB ratings. The end goal is to detect if a movie rating is positive or negative.

A notebook is also available [here](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/fl-with-flower.ipynb) but instead of running on multiple separate clients it utilizes the simulation functionality of Flower (using `flwr['simulation']`) in order to emulate a federated setting inside Google Colab (this also means that instead of calling `start_server` we will call `start_simulation`, and that a few other modifications are needed).

## Dependencies

To follow along this tutorial you will need to install the following packages: `datasets`, `evaluate`, `flwr`, `torch`, and `transformers`. This can be done using `pip`:

```sh
pip install datasets evaluate flwr torch transformers
```

## Standard Hugging Face workflow

### Handling the data

To fetch the IMDB dataset, we will use Hugging Face's `datasets` library. We then need to tokenize the data and create `PyTorch` dataloaders, this is all done in the `load_data` function:

```python
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"

def load_data():
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # We will take a small sample in order to reduce the compute time, this is optional
    train_population = random.sample(range(len(raw_datasets["train"])), 100)
    test_population = random.sample(range(len(raw_datasets["test"])), 100)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(train_population)
    tokenized_datasets["test"] = tokenized_datasets["test"].select(test_population)
    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader
    
trainloader, testloader = load_data()
```

### Training and testing the model

Once we have a way of creating our trainloader and testloader, we can take care of the training and testing. This is very similar to any `PyTorch` training or testing loop:

```python
from evaluate import load as load_metric
from transformers import AdamW


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy
```

### Creating the model itself

To create the model itself, we will just load the pre-trained distillBERT model using Hugging Face’s `AutoModelForSequenceClassification` :

```python
from transformers import AutoModelForSequenceClassification 


net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)
```

## Federating the example

The idea behind Federated Learning is to train a model between multiple clients and a server without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server, which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by using the [Flower](https://github.com/adap/flower) framework. If you want a more complete overview, be sure to check out this guide: [What is Federated Learning?](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html)

### Creating the IMDBClient

To federate our example to multiple clients, we first need to write our Flower client class (inheriting from `flwr.client.NumPyClient`). This is very easy, as our model is a standard `PyTorch` model:

```python
from collections import OrderedDict

import flwr as fl


class IMDBClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}
```

The `get_parameters` function lets the server get the client's parameters. Inversely, the `set_parameters` function allows the server to send its parameters to the client. Finally, the `fit` function trains the model locally for the client, and the `evaluate` function tests the model locally and returns the relevant metrics. 

We can now start client instances using:

```python
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=IMDBClient(),
)
```

### Starting the server

Now that we have a way to instantiate clients, we need to create our server in order to aggregate the results. Using Flower, this can be done very easily by first choosing a strategy (here, we are using `FedAvg`, which will define the global weights as the average of all the clients' weights at each round) and then using the `flwr.server.start_server` function:

```python
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```

The `weighted_average` function is there to provide a way to aggregate the metrics distributed amongst the clients (basically this allows us to display a nice average accuracy and loss for every round).

## Putting everything together

If you want to check out everything put together, you should check out the code example we wrote for the Flower repo: [https://github.com/adap/flower/tree/main/examples/quickstart-huggingface](https://github.com/adap/flower/tree/main/examples/quickstart-huggingface). 

Of course, this is a very basic example, and a lot can be added or modified, it was just to showcase how simply we could federate a Hugging Face workflow using Flower.

Note that in this example we used `PyTorch`, but we could have very well used `TensorFlow`.
