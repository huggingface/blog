---
title: "Introducing Trackio: A Lightweight Experiment Tracking Library from Hugging Face"
thumbnail: /blog/assets/trackio/thumbnail.gif
authors:
- user: abidlabs
- user: znation
- user: nouamanetazi
- user: sasha
---

# Introducing Trackio: A Lightweight Experiment Tracking Library from Hugging Face

If you have trained your own machine learning model, you know how important it is to be able to track metrics, parameters, and hyperparameters during training and visualize them afterwards to better understand your training run. 

Most machine learning researchers use specific experiment tracking libraries to do this. However, these libraries can be paid, require complex setup, or lack the flexibility needed for rapid experimentation and sharing.

## Why We Switched to Trackio

At Hugging Face, our science team has been using Trackio for our research projects, and we've found several key advantages over other tracking solutions:

**Easy Sharing and Embedding**: Trackio makes it incredibly simple to share training progress with colleagues or embed plots directly in blog posts and documentation using iframes. This is especially valuable when you want to showcase specific training curves or metrics without requiring others to set up accounts or navigate complex dashboards.

**Standardization and Transparency**: Metrics like GPU energy usage are important to track and share with the community so we can have a better idea of the energy demands and environmental impacts of model training. Using Trackio, which directly gets information from the `nvidia-smi` command, makes it easy to quantify and compare energy usage and to add it to [model cards](https://huggingface.co/docs/hub/model-cards). 

**Data Accessibility**: Unlike some tracking tools that lock your data behind proprietary APIs, Trackio makes it straightforward to extract and analyze the data being recorded. This is crucial for researchers who need to perform custom analysis or integrate training metrics into their research workflows.

**Flexibility for Experimentation**: Trackio's lightweight design allows us to easily experiment with new tracking features during training runs. For instance, we can decide when to move tensors from GPU to CPU when logging tensors while training, which significantly improves training throughput when you need to track model/intermediate states without impacting performance.

That's why the Hugging Face open-source team is excited to introduce: `trackio`, a completely free, open-source Python library...

## Installing

You can install `trackio` using pip:

```bash
pip install trackio
```

Or, if you prefer using `uv`:

```bash
uv pip install trackio
```

## Usage

`trackio` is designed to be a drop-in replacement for experiment tracking libraries like `wandb`. The API is compatible with `wandb.init`, `wandb.log`, and `wandb.finish`, so you can simply import `trackio` as `wandb` in your code. Here is an example:

```python
import trackio as wandb
import random
import time

runs = 3
epochs = 8

def simulate_multiple_runs():
    for run in range(runs):
        wandb.init(project="fake-training", config={
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 64
        })
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
```

After logging your experiments, you can launch the dashboard to visualize your results. Run the following command in your terminal:

```bash
trackio show
```

Or, launch it from Python:

```python
import trackio
trackio.show()
```

You can also specify a project name:

```bash
trackio show --project "my project"
```

Or in Python:

```python
trackio.show(project="my project")
```

To deploy the dashboard to Hugging Face Spaces, pass a `space_id` to `init`:

```python
trackio.init(project="fake-training", space_id="org_name/space_name")
```

If you are hosting your dashboard on Spaces, you can embed it anywhere using an iframe:

```html
<iframe src="https://abidlabs-trackio-1234.hf.space/?project=fake-training&metrics=train_loss,train_accuracy&sidebar=hidden" width=1600 height=500 frameBorder="0">
```

## Design Principles

- API compatible with popular experiment tracking libraries, making migration both to and from Trackio seamless.
- Local-first: logs and dashboards run and persist locally by default, with the option to host on Hugging Face Spaces.
- Lightweight and extensible: the core codebase is under 1,000 lines of Python, making it easy to understand and modify.
- Free and open-source: all features, including hosting on Hugging Face, are free.
- Built on top of 🤗 Datasets and Spaces for robust data handling and visualization.

## Limitations

Trackio is intentionally lightweight and not intended to be a fully-featured experiment tracking solution. It is currently in beta, and the database schema may change, which could require migrating or deleting existing database files (by default at `~/.cache/huggingface/trackio`).

Some features found in other tracking tools, such as artifact management, or complex visualizations, is available yet. Feedback and contributions are welcome as the project evolves.
