---
title: "TorchNeuron: running HF models on AWS Trainium is easier than ever"
thumbnail: /blog/assets/hf-torchneuron/thumbnail.png
authors:
  - user: regisss
  - user: pagezyhf
  - user: michaelbenayoun
  - user: Jingya
  - user: tengomucho
  - user: dacorvo
  - user: jeffboudier
---

# TorchNeuron: running HF models on AWS Trainium is easier than ever

TorchNeuron provides a new native backend for PyTorch on AWS Trainium. Transformers, Diffusers and Kernels natively support it, making it easier than ever to run models on Trainium and to share them on the HF hub.

## TorchNeuron

[TorchNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html) is a PyTorch plugin that brings a new backend for Trainium enabling support for eager mode, \`torch.compile\` and native distributed APIs like FSDP and TP.
Read the Pytorch [blog](https://pytorch.org/blog/introducing-torchneuron-native-pytorch-backend-for-aws-trainium/) to learn more about TorchNeuron.

Starting Oct. 20, 2026, TorchNeuron will be available as a public beta. Researchers and developers will be able to install it with `pip install torch-neuronx`.

## New backend, native interface in HF libraries

To date, the Hugging Face experience on AWS Trainium and Inferentia — AI chips designed respectively for cost-effective training and inference at scale — has been delivered through [optimum-neuron](https://github.com/huggingface/optimum-neuron). This library relies on the torch-neuronx package, which is based on PyTorch XLA. It was the natural choice at the time, but it meant thinking like a compiler: lazy tensors instead of eager execution, static shapes only, a recompilation for every graph break. Optimum Neuron existed to hide that complexity, and maintained custom code (modeling classes, trainers, etc) to do so.

TorchNeuron changes this. It integrates directly into PyTorch and exposes `neuron` as a native device, with eager execution, PyTorch's own distributed primitives (DTensor, FSDP2) and `torch.compile`.

For the Hugging Face libraries, Trainium becomes a device like any other: Transformers, Diffusers, Accelerate, PEFT and TRL run on it directly, with the same classes and scripts you would use on any other accelerator. From now on, we recommend that Neuron users use these libraries directly instead of going through Optimum Neuron.

### Transformers

#### Inference

Text generation is now natively supported in Transformers for a wide range of models. With TorchNeuron, 89% of the top 100 most-used text-generation model architectures are supported out of the box, compared to 46% with Optimum Neuron v0.4.5. You just need to pass the `device_map=neuron` argument in your code and your model will be moved to the Neuron devices.

The following is an example of text-generation with tensor parallelism, which will enable to fully benefit from your Neuron instance. Note that it uses a static cache for generation as this is strongly recommended with Neuron to prevent frequent recompilations.

```py
import os

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401
from torch.distributed import DeviceMesh
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.distributed import DistributedConfig
from transformers.generation.configuration_utils import CompileConfig

MODEL_ID = "Qwen/Qwen3.8-27B"
MAX_SEQ_LEN = 1024
PREFILL_CHUNK_SIZE = 512

dist.init_process_group(backend="neuron")
world_size = dist.get_world_size()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=world_size),
    device_mesh=DeviceMesh("neuron", list(range(world_size))),
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "The future of AI is"}],
    add_generation_prompt=True,
    padding=True,
    tokenizer_kwargs={"pad_to_multiple_of": PREFILL_CHUNK_SIZE},
    return_tensors="pt",
)

generation_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=False,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    cache_implementation="static",
    max_cache_len=MAX_SEQ_LEN,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
    compile_config=CompileConfig(backend="neuron", fullgraph=False, dynamic=False),
)

with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=generation_config,
    )

if dist.get_rank() == 0:
    prompt_len = inputs["input_ids"].shape[1]
    print(tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True))
```

#### Training

Now that Neuron is a native PyTorch device, training on Trainium uses the same Hugging Face libraries as any other accelerator: Transformers, Accelerate, PEFT and TRL. You no longer need a dedicated trainer or a Neuron-specific fork of the model code. A standard TRL fine-tuning script runs on a Trainium instance with only a few Neuron-specific settings.

Here is a supervised fine-tuning script built on TRL's `SFTTrainer`. If you have fine-tuned a model with TRL on GPUs before, it should look familiar.

```py
import os

import torch
from accelerate import ParallelismConfig
from datasets import load_dataset
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from transformers.distributed import DistributedConfig

def main(script_args, training_args, model_args):
    # Disable automatic dynamic shapes for torch.compile
    torch._dynamo.config.automatic_dynamic_shapes = False

    # Distributed training setup
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    fsdp_size = int(os.environ.get("FSDP_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
    )

    distributed_config = None
    if world_size > 1:
        distributed_config = DistributedConfig(
            tp_size=tp_size,
            fsdp_size=fsdp_size,
        )
        training_args.parallelism_config = ParallelismConfig(tp_size=tp_size, dp_shard_size=fsdp_size)

    # Prepare dataset
    train_dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    # UltraChat also has a string `prompt` column, which makes TRL treat it as a prompt-completion dataset.
    train_dataset = train_dataset.select_columns(["messages"])

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        low_cpu_mem_usage=True,
        distributed_config=distributed_config,
    )

    # Training
    set_seed(training_args.seed)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
```

Parallelism is handled natively by Transformers. You pass a `DistributedConfig` to `from_pretrained` with the tensor-parallel (TP) and FSDP sizes, and the model is sharded as its weights are loaded. The Trainer sees that the model is already distributed and does not wrap it again. You don't need an `--fsdp` flag or an Accelerate FSDP config.

Nothing in the script is tied to Trainium: the same code runs on any hardware. A few things are worth knowing when you run it on Trainium:

* **Static shapes.** The Neuron compiler works best when every step sees the same input shape. The script turns off automatic dynamic shapes in `torch.compile`, and the command packs samples into fixed sequences of 2048 tokens (`--packing true` and `--max_length 2048`). The graph is compiled once and reused at every step.
* **`torch.compile` on.** Passing `--torch_compile true` is all it takes: on a Trainium instance, `torch.compile` detects the device and picks the Neuron backend automatically, with nothing else to configure.
* **Async weight loading off.** The built-in asynchronous weight loading in Transformers is not supported on Neuron, so for now we disable it with `HF_DEACTIVATE_ASYNC_LOAD=1`.

The command below launches training with `torchrun`, which starts one process per NeuronCore. `TP_SIZE` and `FSDP_SIZE` set the parallelism layout, and their product is the number of processes: here 4 x 8 \= 32, matching `--nproc_per_node`. The 32 processes form a 4 x 8 grid. Tensor parallelism splits each layer across the 4 cores of a TP group, and FSDP shards the parameters, gradients and optimizer state across the 8 TP groups.

```sh
TP_SIZE=4 FSDP_SIZE=8 HF_DEACTIVATE_ASYNC_LOAD=1 uv run torchrun --nproc_per_node=32 sft_neuron_example.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --dataset_train_split train_sft \
    --output_dir Qwen3-8B-sft-tp4-fsdp8 \
    --dtype bfloat16 \
    --max_length 2048 \
    --packing true \
    --packing_strategy wrapped \
    --learning_rate 2e-5 \
    --warmup_steps 0.03 \
    --num_train_epochs 1 \
    --logging_steps 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --save_only_model true \
    --torch_compile true
```

We ran it on 8 Trainium2 chips (32 NeuronCores), which is half a `trn2.48xlarge` instance.

Training runs to completion, and `trainer.save_model` writes a standard Transformers checkpoint. Nothing in it is tied to Neuron: you can push it to the Hugging Face Hub as is and load it on any other hardware.

### Diffusers

#### Inference

Many pipelines from the Diffusers library are compatible with TorchNeuron in eager mode. The most widely used ones are also compatible with \`torch.compile\` and tensor parallelism (TP) to make the most of your Neuron devices. The following is an example showing how you can use TP:

```py
import torch, torch.distributed as dist, torch_neuronx # TP
from torch.distributed.device_mesh import DeviceMesh # TP
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, TensorParallelConfig

# Distributed devices initialization
dist.init_process_group(backend="neuron") # TP
dev = torch.neuron.current_device()
mesh = DeviceMesh("neuron", list(range(8))) # TP=8

# Pipeline initialization
model = "Qwen/Qwen-Image-2.1"
transformer = QwenImage21Transformer2DModel.from_pretrained(
    model,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    parallel_config=TensorParallelConfig(mesh=mesh), # TP
)
pipe = QwenImage21Pipeline.from_pretrained(
    model,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.vae.to(dev)
pipe.transformer.compile_repeated_blocks(backend="neuron", fullgraph=True, dynamic=False)

# Image generation
emb, mask, _ = pipe.encode_prompt(prompt="A capybara wearing a wizard hat, oil painting", device="cpu")  # text encoder stays on CPU
image = pipe(
    prompt_embeds=emb.to(dev),
    prompt_embeds_mask=mask.to(dev) if mask is not None else None,
    height=1024,
    width=1024,
    num_inference_steps=40,
    generator=torch.Generator("cpu").manual_seed(0), # same noise on every rank
).images[0]
if dist.get_rank() == 0:
    image.save("out.png")
```

You can run this piece of code with: \``NEURON_LOGICAL_NC_CONFIG=2 torchrun --nproc_per_node=8 run.py`\`.

### Kernels

Make sure to check out the [kernels from the Neuron team](https://huggingface.co/aws-neuron/kernels)  as more and more Neuron kernels will become available on the Hub to speed up your models.
Neuron kernels can be written using the [Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) (NKI), which gives lower control to optimize operations. This way, users can run their own custom operations and share them on the hub.

You can use for example [this kernel](https://huggingface.co/kernels/aws-neuron/vit-encoder-neuron-kernels) to speed up your vision transformers with up to a x2.4 speedup on Trainium!

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-torchneuron/kernel.png)

## Acknowledgments

Special thanks to the AWS team for their work on TorchNeuron and for their reactivity to our feedback to make the HF integration as good as possible.
