---
title: Training a Language Model with Megatron-LM
thumbnail: /blog/assets/100_megatron_training/thumbnail.png
---

<h1>Training a Language Model with Megatron-LM</h1>

<div class="blog-metadata">
    <small>Published August 26, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/megatron-training.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/loubnabnl">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/44069155?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>loubnabnl</code>
            <span class="fullname">Loubna Ben Allal</span>
        </div>
    </a>
</div>

In this tutorial, you will learn how to train a Language Model on NVIDIA GPUs with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), a powerful transformer model framework developed by the Applied Deep Learning Research team at NVIDIA, which gives [2x speedup](https://arxiv.org/pdf/2205.14135.pdf) compared to Hugging Face [Transformers](https://github.com/huggingface/transformers.git). 

Megatron-LM is widely used by researchers to pre-train large language models such as GPT, BERT, and T5. However, it offers less flexibility compared to `transformers` and can seem like a black hole to beginners, which limits its use. We believe that the speedup this framework provides makes it important to learn how to use it, especially when one has limited computing resources.

In this blog, we will try to break down the different steps for training a GPT2 model in this framework, this includes:
* Environment setup
* Data preprocessing
* Training
* Model conversion to 🤗 Transformers

## Why Megatron-LM?

Before getting into the training details, let’s first understand what makes this framework more efficient than others. This section is inspired by this great [blog](https://huggingface.co/blog/bloom-megatron-deepspeed) about BLOOM training in [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed), please refer to it for more details as this blog is intended to give a gentle introduction to Megatron-LM.

### DataLoader

Megatron-LM comes with an efficient DataLoader, where the data is tokenized and shuffled before the training. It is also split into numbered sequences with indexes that are saved in a file, to avoid recomputing them in the training loop. The way the data is loaded makes the learning curve smooth and saves time during the training. 

### Fused CUDA Kernels

In simple words, the idea of fused kernels here is that similar operations that are normally performed separately by Pytorch, are combined into a single hardware operation. So they reduce the number of memory movements done in multiple discrete computations by merging them into one. This gives a significant speed up to the training. Megatron-LM also uses a Fused implementation of AdamW from [Apex](https://github.com/NVIDIA/apex) which is faster than the Pytorch implementation.

While one can customize the DataLoader like Megatron-LM and use Apex’s Fused optimizer with `transformers`, it is hard to build Fused CUDA Kernels.

Now that you are familiar with the framework and what makes it advantageous, let’s get into the training details!

## How to train with Megatron-LM ?

### Setup
The easiest way to setup the environement is to pull an NVIDIA PyTorch Container that comes with all the required installations from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). See [documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) for more details. If you don't want to use this container you will need to install the latest pytorch, cuda, nccl, and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) releases and the `nltk` library.

So after having installed Docker, you can run the container with the following command (`xx.xx` denotes your Docker version), and then clone [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM) inside it:
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:xx.xx-py3
git clone https://github.com/NVIDIA/Megatron-LM
```

You also need to add the vocabulary file and merges table of your tokenizer inside Megatron-LM folder of your container. If you want to copy this data from outside the container you can use the following commands
```bash
sudo docker cp vocab.json CONTAINER_ID:/workspace/Megatron-LM
sudo docker cp merges.txt CONTAINER_ID:/workspace/Megatron-LM
```

### Data preprocessing
In the rest of this tutorial we will be using [CodeParrot](https://huggingface.co/codeparrot/codeparrot-small) model and data as an example.

The training data requires preprocessing. First, you need to convert it into a loose json format, with one json containing a text sample per line. If you're using `datasets`, here is an example on how to do that (always inside Megatron-LM folder):
```python
from datasets import load_dataset

train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')
train_data.to_json("codeparrot_data.json", lines=True)  
```

The data is then tokenized, shuffled and processed into a binary format for training using the following command:
```bash
#if nltk isn't installed
pip install nltk
python tools/preprocess_data.py \
       --input codeparrot_data.json \
       --output-prefix codeparrot \
       --vocab vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --json-keys content \
       --workers 32 \
       --chunk-size 25 \
       --append-eod
```
This outputs two files `codeparrot_content_document.idx` and `codeparrot_content_document.bin` which are used in the training.

### Training
You can configure the model architecture and training parameters as shown below, or put it in a bash script that you will run. This runs on 8 GPUs the 110M parameter CodeParrot pre-training. Note that the data is partitioned by default into a 969:30:1 ratio for training/validation/test sets.
```bash
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/workspace/Megatron-LM/experiments/codeparrot-small
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
DATA_PATH=codeparrot_content_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 2000
--eval-interval 200
--eval-iters 10
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS
```
The training takes almost 12 hours in this setting.

### Convert model to 🤗 Transformers
After training we want to use the model in `transformers` e.g. to evaluate it. You can convert it to `transformers` following this [tutorial](https://huggingface.co/nvidia/megatron-gpt2-345m). For instance, after the training is finished you can copy the weights of the last iteration 150k and convert the `model_optim_rng.pt` file to a `pytorch_model.bin` file that is supported by `transformers`.

```bash
# to execute outside the container:
mkdir -p nvidia/megatron-codeparrot-small
# copy the weights from the container
sudo docker cp CONTAINER_ID:/workspace/Megatron-LM/experiments/codeparrot-small/iter_0150000/mp_rank_00/model_optim_rng.pt nvidia/megatron-codeparrot-small
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/NVIDIA/Megatron-LM.git
export PYTHONPATH=Megatron-LM
python transformers/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py nvidia/megatron-codeparrot-small/model_optim_rng.pt
```
Be careful, you will need to replace the generated vocabulary file and merges table after the conversion, with the original ones if you plan to load the tokenizer from there.

Disclaimer: This framework adds some time overhead because of the extra preprocessing and conversion steps. So it is important that you decide for your case and given your model size which framework is more appropriate. But in general, for large models it's worth giving it a try.

Congratulations 🎉 now you know how to train a GPT2 model in Megatron-LM and make it supported by `transformers`!
