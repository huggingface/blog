---
title: 'The Partnership: Amazon SageMaker and Hugging Face'
thumbnail: /blog/assets/17_the_partnership_amazon_sagemaker_and_hugging_face/thumbnail.png
---

<img src="/blog/assets/17_the_partnership_amazon_sagemaker_and_hugging_face/cover.png" alt="hugging-face-and-aws-logo" class="w-full">

> Look at these smiles!

# **The Partnership: Amazon SageMaker and Hugging Face**

Today, we announce a strategic partnership between Hugging Face and [Amazon](https://huggingface.co/amazon) to make it easier for companies to leverage State of the Art Machine Learning models, and ship cutting-edge NLP features faster.

Through this partnership, Hugging Face is leveraging Amazon Web Services as its Preferred Cloud Provider to deliver services to its customers.

As a first step to enable our common customers, Hugging Face and Amazon are introducing new Hugging Face Deep Learning Containers (DLCs) to make it easier than ever to train Hugging Face Transformer models in [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

To learn how to access and use the new Hugging Face DLCs with the Amazon SageMaker Python SDK, check out the guides and resources below.

> _At the 8th of July we extended the Amazon SageMaker experience to inference. If you want to learn how you can [deploy Hugging Face models easily with Amazon SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker) take a look at the [new blog post](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker) or at the [documentation](https://huggingface.co/docs/sagemaker/inference)._

---

## **Features & Benefits 🔥**

## One Command is All you Need

With the new Hugging Face Deep Learning Containers available in Amazon SageMaker, training cutting-edge Transformers-based NLP models has never been simpler. There are variants specially optimized for TensorFlow and PyTorch, for single-GPU, single-node multi-GPU and multi-node clusters.

## Accelerating Machine Learning from Science to Production

In addition to Hugging Face DLCs, we created a first-class Hugging Face extension to the SageMaker Python-sdk to accelerate data science teams, reducing the time required to set up and run experiments from days to minutes.

You can use the Hugging Face DLCs with the Automatic Model Tuning capability of Amazon SageMaker, in order to automatically optimize your training hyperparameters and quickly increase the accuracy of your models.

Thanks to the SageMaker Studio web-based Integrated Development Environment (IDE), you can easily track and compare your experiments and your training artifacts.

## Built-in Performance

With the Hugging Face DLCs, SageMaker customers will benefit from built-in performance optimizations for PyTorch or TensorFlow, to train NLP models faster, and with the flexibility to choose the training infrastructure with the best price/performance ratio for your workload.

The Hugging Face DLCs are fully integrated with the [SageMaker distributed training libraries](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html), to train models faster than was ever possible before, using the latest generation of instances available on Amazon EC2.

---

## **Resources, Documentation & Samples 📄**

Below you can find all the important resources to all published blog posts, videos, documentation, and sample Notebooks/scripts.

## Blog/Video

- [AWS: Embracing natural language processing with Hugging Face](https://aws.amazon.com/de/blogs/opensource/embracing-natural-language-processing-with-hugging-face/)
- [Deploy Hugging Face models easily with Amazon SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker)
- [AWS and Hugging Face collaborate to simplify and accelerate adoption of natural language processing models](https://aws.amazon.com/blogs/machine-learning/aws-and-hugging-face-collaborate-to-simplify-and-accelerate-adoption-of-natural-language-processing-models/)
- [Walkthrough: End-to-End Text Classification](https://youtu.be/ok3hetb42gU)
- [Working with Hugging Face models on Amazon SageMaker](https://youtu.be/leyrCgLAGjMn)
- [Distributed Training: Train BART/T5 for Summarization using 🤗 Transformers and Amazon SageMaker](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq)
- [Deploy a Hugging Face Transformers Model from S3 to Amazon SageMaker](https://youtu.be/pfBGgSGnYLs)
- [Deploy a Hugging Face Transformers Model from the Model Hub to Amazon SageMaker](https://youtu.be/l9QZuazbzWM)

## Documentation

- [Hugging Face documentation for Amazon SageMaker](https://huggingface.co/docs/sagemaker/main)
- [Run training on Amazon SageMaker](https://huggingface.co/docs/sagemaker/train)
- [Deploy models to Amazon SageMaker](https://huggingface.co/docs/sagemaker/inference)
- [Frequently Asked Questions](https://huggingface.co/docs/sagemaker/faq)
- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
- [Python SDK SageMaker documentation for Hugging Face](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
- [Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)
- [SageMaker's Distributed Data Parallel Library](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html)
- [SageMaker's Distributed Model Parallel Library](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html)

## Sample Notebook

- [all Notebooks](https://github.com/huggingface/notebooks/tree/master/sagemaker)
- [Getting Started Pytorch](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb)
- [Getting Started Tensorflow](https://github.com/huggingface/notebooks/blob/master/sagemaker/02_getting_started_tensorflow/sagemaker-notebook.ipynb)
- [Distributed Training Data Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/03_distributed_training_data_parallelism/sagemaker-notebook.ipynb)
- [Distributed Training Model Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)
- [Spot Instances and continue training](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb)
- [SageMaker Metrics](https://github.com/huggingface/notebooks/blob/master/sagemaker/06_sagemaker_metrics/sagemaker-notebook.ipynb)
- [Distributed Training Data Parallelism Tensorflow](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb)
- [Distributed Training Summarization](https://github.com/huggingface/notebooks/blob/master/sagemaker/08_distributed_summarization_bart_t5/sagemaker-notebook.ipynb)
- [Image Classification with Vision Transformer](https://github.com/huggingface/notebooks/blob/master/sagemaker/09_image_classification_vision_transformer/sagemaker-notebook.ipynb)
- [Deploy one of the 10 000+ Hugging Face Transformers to Amazon SageMaker for Inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb)
- [Deploy a Hugging Face Transformer model from S3 to SageMaker for inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/10_deploy_model_from_s3/deploy_transformer_model_from_s3.ipynb)

---

## **Getting started: End-to-End Text Classification 🧭**

In this getting started guide, we will use the new Hugging Face DLCs and Amazon SageMaker extension to train a transformer model on binary text classification using the transformers and datasets libraries.

We will use an Amazon SageMaker Notebook Instance for the example. You can learn [here how to set up a Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html).

**What are we going to do:**

- set up a development environment and install sagemaker
- create the training script `train.py`
- preprocess our data and upload it to [Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)
- create a [HuggingFace Estimator](https://huggingface.co/transformers/sagemaker.html) and train our model

## Set up a development environment and install sagemaker

As mentioned above we are going to use SageMaker Notebook Instances for this. To get started you need to jump into your Jupyer Notebook or JupyterLab and create a new Notebook with the conda_pytorch_p36 kernel.

_**Note:** The use of Jupyter is optional: We could also launch SageMaker Training jobs from anywhere we have an SDK installed, connectivity to the cloud and appropriate permissions, such as a Laptop, another IDE or a task scheduler like Airflow or AWS Step Functions._

After that we can install the required dependencies

```bash
pip install "sagemaker>=2.31.0" "transformers==4.6.1" "datasets[s3]==1.6.2" --upgrade
```

To run training on SageMaker we need to create a sagemaker Session and provide an IAM role with the right permission. This IAM role will be later attached to the TrainingJob enabling it to download data, e.g. from Amazon S3.

```python
import sagemaker

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

role = sagemaker.get_execution_role()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")
```

## Create the training script `train.py`

In a SageMaker `TrainingJob` we are executing a python script with named arguments. In this example, we use PyTorch together with transformers. The script will

- pass the incoming parameters (hyperparameters from HuggingFace Estimator)
- load our dataset
- define our compute metrics function
- set up our `Trainer`
- run training with `trainer.train()`
- evaluate the training and save our model at the end to S3.

```bash
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\\n")

    # Saves the model to s3; default is /opt/ml/model which SageMaker sends to S3
    trainer.save_model(args.model_dir)
```

## Preprocess our data and upload it to s3

We use the `datasets` library to download and preprocess our `imdb` dataset. After preprocessing, the dataset will be uploaded to the current session’s default s3 bucket `sess.default_bucket()` used within our training job. The `imdb` dataset consists of 25000 training and 25000 testing highly polar movie reviews.

```python
import botocore
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets.filesystems import S3FileSystem

# tokenizer used in preprocessing
tokenizer_name = 'distilbert-base-uncased'

# filesystem client for s3
s3 = S3FileSystem()

# dataset used
dataset_name = 'imdb'

# s3 key prefix for the data
s3_prefix = 'datasets/imdb'

# load dataset
dataset = load_dataset(dataset_name)

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# load dataset
train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
test_dataset = test_dataset.shuffle().select(range(10000)) # smaller the size for test dataset to 10k

# tokenize dataset
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

# set format for pytorch
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
train_dataset.save_to_disk(training_input_path,fs=s3)

# save test_dataset to s3
test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'
test_dataset.save_to_disk(test_input_path,fs=s3)
```

## Create a HuggingFace Estimator and train our model

In order to create a SageMaker `Trainingjob` we can use a HuggingFace Estimator. The Estimator handles the end-to-end Amazon SageMaker training. In an Estimator, we define which fine-tuning script should be used as `entry_point`, which `instance_type` should be used, which hyperparameters are passed in. In addition to this, a number of advanced controls are available, such as customizing the output and checkpointing locations, specifying the local storage size or network configuration.

SageMaker takes care of starting and managing all the required Amazon EC2 instances for us with the Hugging Face DLC, it uploads the provided fine-tuning script, for example, our `train.py`, then downloads the data from the S3 bucket, `sess.default_bucket()`, into the container. Once the data is ready, the training job will start automatically by running.

```bash
/opt/conda/bin/python train.py --epochs 1 --model_name distilbert-base-uncased --train_batch_size 32
```

The hyperparameters you define in the HuggingFace Estimator are passed in as named arguments.

```python
from sagemaker.huggingface import HuggingFace

# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'model_name':'distilbert-base-uncased'
                 }

# create the Estimator
huggingface_estimator = HuggingFace(
      entry_point='train.py',
      source_dir='./scripts',
      instance_type='ml.p3.2xlarge',
      instance_count=1,
      role=role,
      transformers_version='4.6',
      pytorch_version='1.7',
      py_version='py36',
      hyperparameters = hyperparameters
)
```

To start our training we call the .fit() method and pass our S3 uri as input.

```python
# starting the train job with our uploaded datasets as input
huggingface_estimator.fit({'train': training_input_path, 'test': test_input_path})
```

---

## **Additional Features 🚀**

In addition to the Deep Learning Container and the SageMaker SDK, we have implemented other additional features.

## Distributed Training: Data-Parallel

You can use [SageMaker Data Parallelism Library](https://aws.amazon.com/blogs/aws/managed-data-parallelism-in-amazon-sagemaker-simplifies-training-on-large-datasets/) out of the box for distributed training. We added the functionality of Data Parallelism directly into the Trainer. If your train.py uses the Trainer API you only need to define the distribution parameter in the HuggingFace Estimator.

- [Example Notebook PyTorch](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)
- [Example Notebook TensorFlow](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb)

```python
# configuration for running training on smdistributed Data Parallel
distribution = {'smdistributed':{'dataparallel':{ 'enabled': True }}}

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3dn.24xlarge',
        instance_count=2,
        role=role,
        transformers_version='4.4.2',
        pytorch_version='1.6.0',
        py_version='py36',
        hyperparameters = hyperparameters
        distribution = distribution
)
```

The "Getting started: End-to-End Text Classification 🧭" example can be used for distributed training without any changes.

## Distributed Training: Model Parallel


You can use [SageMaker Model Parallelism Library](https://aws.amazon.com/blogs/aws/amazon-sagemaker-simplifies-training-deep-learning-models-with-billions-of-parameters/) out of the box for distributed training. We added the functionality of Model Parallelism directly into the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html). If your `train.py` uses the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) API you only need to define the distribution parameter in the HuggingFace Estimator.  
For detailed information about the adjustments take a look [here](https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_general.html?highlight=modelparallel#required-sagemaker-python-sdk-parameters).


- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)


```python
# configuration for running training on smdistributed Model Parallel
mpi_options = {
    "enabled" : True,
    "processes_per_host" : 8
}

smp_options = {
    "enabled":True,
    "parameters": {
        "microbatches": 4,
        "placement_strategy": "spread",
        "pipeline": "interleaved",
        "optimize": "speed",
        "partitions": 4,
        "ddp": True,
    }
}

distribution={
    "smdistributed": {"modelparallel": smp_options},
    "mpi": mpi_options
}

 # create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3dn.24xlarge',
        instance_count=2,
        role=role,
        transformers_version='4.4.2',
        pytorch_version='1.6.0',
        py_version='py36',
        hyperparameters = hyperparameters,
        distribution = distribution
)
```

## Spot instances

With the creation of HuggingFace Framework extension for the SageMaker Python SDK we can also leverage the benefit of [fully-managed EC2 spot instances](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) and save up to 90% of our training cost.

_Note: Unless your training job will complete quickly, we recommend you use [checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) with managed spot training, therefore you need to define the `checkpoint_s3_uri`._

To use spot instances with the `HuggingFace` Estimator we have to set the `use_spot_instances` parameter to `True` and define your `max_wait` and `max_run` time. You can read more about [the managed spot training lifecycle here](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html).

- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb)

```python
# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'model_name':'distilbert-base-uncased',
                 'output_dir':'/opt/ml/checkpoints'
                 }
# create the Estimator

huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
	      checkpoint_s3_uri=f's3://{sess.default_bucket()}/checkpoints'
        use_spot_instances=True,
        max_wait=3600, # This should be equal to or greater than max_run in seconds'
        max_run=1000,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters = hyperparameters
)

# Training seconds: 874
# Billable seconds: 105
# Managed Spot Training savings: 88.0%
```

## Git Repositories

When you create an `HuggingFace` Estimator, you can specify a [training script that is stored in a GitHub](https://sagemaker.readthedocs.io/en/stable/overview.html#use-scripts-stored-in-a-git-repository) repository as the entry point for the estimator, so that you don’t have to download the scripts locally. If Git support is enabled, then `entry_point` and `source_dir` should be relative paths in the Git repo if provided.

As an example to use `git_config` with an [example script from the transformers repository](https://github.com/huggingface/transformers/tree/master/examples/text-classification).

_Be aware that you need to define `output_dir` as a hyperparameter for the script to save your model to S3 after training. Suggestion: define output_dir as `/opt/ml/model` since it is the default `SM_MODEL_DIR` and will be uploaded to S3._

- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/02_getting_started_tensorflow/sagemaker-notebook.ipynb)

```python
# configure git settings
git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'master'}

 # create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='run_glue.py',
        source_dir='./examples/text-classification',
        git_config=git_config,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters=hyperparameters
)
```

## SageMaker Metrics

[SageMaker Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html#define-train-metrics) can automatically parse the logs for metrics and send those metrics to CloudWatch. If you want SageMaker to parse logs you have to specify the metrics that you want SageMaker to send to CloudWatch when you configure the training job. You specify the name of the metrics that you want to send and the regular expressions that SageMaker uses to parse the logs that your algorithm emits to find those metrics.

- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/06_sagemaker_metrics/sagemaker-notebook.ipynb)

```python
# define metrics definitions

metric_definitions = [
{"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
{"Name": "eval_accuracy", "Regex": "eval_accuracy.*=\D*(.*?)$"},
{"Name": "eval_loss", "Regex": "eval_loss.*=\D*(.*?)$"},
]

# create the Estimator

huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        metric_definitions=metric_definitions,
        hyperparameters = hyperparameters
)
```

---

## **FAQ 🎯**

You can find the complete [Frequently Asked Questions](https://huggingface.co/docs/sagemaker/faq) in the [documentation](https://huggingface.co/docs/sagemaker/faq).


_Q: What are Deep Learning Containers?_

A: Deep Learning Containers (DLCs) are Docker images pre-installed with deep learning frameworks and libraries (e.g. transformers, datasets, tokenizers) to make it easy to train models by letting you skip the complicated process of building and optimizing your environments from scratch.

_Q: Do I have to use the SageMaker Python SDK to use the Hugging Face Deep Learning Containers?_

A: You can use the HF DLC without the SageMaker Python SDK and launch SageMaker Training jobs with other SDKs, such as the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-training-job.html) or [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job). The DLCs are also available through Amazon ECR and can be pulled and used in any environment of choice.

_Q: Why should I use the Hugging Face Deep Learning Containers?_

A: The DLCs are fully tested, maintained, optimized deep learning environments that require no installation, configuration, or maintenance.

_Q: Why should I use SageMaker Training to train Hugging Face models?_

A: SageMaker Training provides numerous benefits that will boost your productivity with Hugging Face : (1) first it is cost-effective: the training instances live only for the duration of your job and are paid per second. No risk anymore to leave GPU instances up all night: the training cluster stops right at the end of your job! It also supports EC2 Spot capacity, which enables up to 90% cost reduction. (2) SageMaker also comes with a lot of built-in automation that facilitates teamwork and MLOps: training metadata and logs are automatically persisted to a serverless managed metastore, and I/O with S3 (for datasets, checkpoints and model artifacts) is fully managed. Finally, SageMaker also allows to drastically scale up and out: you can launch multiple training jobs in parallel, but also launch large-scale distributed training jobs

_Q: Once I've trained my model with Amazon SageMaker, can I use it with 🤗/Transformers ?_

A: Yes, you can download your trained model from S3 and directly use it with transformers or upload it to the [Hugging Face Model Hub](https://huggingface.co/models).

_Q: How is my data and code secured by Amazon SageMaker?_

A: Amazon SageMaker provides numerous security mechanisms including [encryption at rest](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-at-rest-nbi.html) and [in transit](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-in-transit.html), [Virtual Private Cloud (VPC) connectivity](https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html) and [Identity and Access Management (IAM)](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html). To learn more about security in the AWS cloud and with Amazon SageMaker, you can visit [Security in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html) and [AWS Cloud Security](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html).

_Q: Is this available in my region?_

A: For a list of the supported regions, please visit the [AWS region table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) for all AWS global infrastructure.

_Q: Do I need to pay for a license from Hugging Face to use the DLCs?_

A: No - the Hugging Face DLCs are open source and licensed under Apache 2.0.

_Q: How can I run inference on my trained models?_

A: You have multiple options to run inference on your trained models. One option is to use Hugging Face [Accelerated Inference-API](https://api-inference.huggingface.co/docs/python/html/index.html) hosted service: start by [uploading the trained models to your Hugging Face account](https://huggingface.co/new) to deploy them publicly, or privately. Another great option is to use [SageMaker Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-main.html) to run your own inference code in Amazon SageMaker. We are working on offering an integrated solution for Amazon SageMaker with Hugging Face Inference DLCs in the future - stay tuned!

_Q: Do you offer premium support or support SLAs for this solution?_

A: AWS Technical Support tiers are available from AWS and cover development and production issues for AWS products and services - please refer to AWS Support for specifics and scope.

If you have questions which the Hugging Face community can help answer and/or benefit from, please [post them in the Hugging Face forum](https://discuss.huggingface.co/c/sagemaker/17).

If you need premium support from the Hugging Face team to accelerate your NLP roadmap, our Expert Acceleration Program offers direct guidance from our open source, science and ML Engineering team - [contact us to learn more](mailto:api-enterprise@huggingface.co).

_Q: What are you planning next through this partnership?_

A: Our common goal is to democratize state of the art Machine Learning. We will continue to innovate to make it easier for researchers, data scientists and ML practitioners to manage, train and run state of the art models. If you have feature requests for integration in AWS with Hugging Face, please [let us know in the Hugging Face community forum](https://discuss.huggingface.co/c/sagemaker/17).

_Q: I use Hugging Face with Azure Machine Learning or Google Cloud Platform, what does this partnership mean for me?_

A: A foundational goal for Hugging Face is to make the latest AI accessible to as many people as possible, whichever framework or development environment they work in. While we are focusing integration efforts with Amazon Web Services as our Preferred Cloud Provider, we will continue to work hard to serve all Hugging Face users and customers, no matter what compute environment they run on.
