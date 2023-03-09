---
title: "Federated Learning using Hugging Face and Flower" 
thumbnail: /blog/assets/fl-with-flower/thumbnail.gif
authors:
- user: charlesbvll
---

# Federated Learning using Hugging Face and Flower

<!-- {blog_metadata} -->
<!-- {authors} -->

This tutorial will show how to leverage Hugging Face to federate the training of language models over multiple clients using [Flower](https://flower.dev/). More specifically, we will fine-tune a pre-trained Transformer model (distilBERT) for sequence classification over a dataset of IMDB ratings. The end goal is to detect if a movie rating is positive or negative.

## Standard Hugging Face workflow

### Handling the data

To fetch the IMDB dataset, we will use Hugging Face's `datasets` library. We then need to tokenize the data and create `PyTorch` dataloaders, this is all done in the `load_data` function:

```python
def load_data():
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # random 100 samples
    population = random.sample(range(len(raw_datasets["train"])), 100)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(population)
    tokenized_datasets["test"] = tokenized_datasets["test"].select(population)

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
```

### Training and testing the model

Once we have a way of creating our trainloader and testloader, we can take care of the training and testing. This is very similar to any `PyTorch` training or testing loop:

```python
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

To create the model itself, we will just load the pre-trained distillBERT model using the Hugging Face’s `AutoModelForSequenceClassification` :

```python
net = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
```

## Federating the example

The idea behind Federated Learning is to train a model between multiple clients and a server, without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by using the [Flower](https://github.com/adap/flower) framework. If you want a more complete overview, be sure to check out this guide: [What is Federated Learning?](https://flower.dev/docs/tutorial/Flower-0-What-is-FL.html)

### Creating the IMDBClient

To federate our example to multiple clients, we first need to write our Flower client class (inheriting from `flwr.client.NumPyClient`). This is very easy, as our model is a standard `PyTorch` model:

```python
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

The `get_parameters` function allows the server to get the client's parameters, inversely, the `set_parameters` function allows the server to send its parameters to the client. Finally the `fit` function is to train the model locally for the client and the `evaluate` function is to test the model locally and return the relevant metrics. 

We can now start client instances using:

```python
flwr.client.start_numpy_client(server_address="127.0.0.1:8080", 
															 client=IMDBClient())
```

### Starting the server

Once we have a way to instantiate clients, we need to create our server in order to aggregate the results. Using Flower, this can be done very easily, by first choosing a strategy and then using the `flwr.server.start_server` function:

```python
# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```

## Putting everything together

If you want to check out everything put together, you should check out the code example we wrote for the Flower repo: [https://github.com/adap/flower/tree/main/examples/quickstart_huggingface](https://github.com/adap/flower/tree/main/examples/quickstart_huggingface). 

Of course, this is a very basic example, and a lot can be added or modified, it was just to showcase how simply we could federate a Hugging Face workflow using Flower.

Note that in this example we used `PyTorch`, but we could have very well used `TensorFlow`.