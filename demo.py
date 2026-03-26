import trackio as wandb
import random
import time

from datasets import Dataset

dataset = Dasaset.from_dict({
    "input_ids": [[1, 2, 3, 4, 5]],
    "labels": [[0, 1, 0, 1, 0]]
})

runs = 3
epochs = 8

def simulate_multiple_runs():
    for run in range(runs):
        wandb.init(project="fake-training", config={
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 64
        },
        space_id="trackio/documentation")
        for epoch in range(epochs):
            train_loss = random.uniform(0.2, 1.0)
            train_acc = random.uniform(0.6, 0.95)
            val_loss = train_loss - random.uniform(0.01, 0.1)
            val_acc = train_acc + random.uniform(0.01, 0.05)
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            time.sleep(0.2)
    wandb.finish()

simulate_multiple_runs()