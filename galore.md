---
title: "GaLore: Advancing Large Model Training on Consumer-grade Hardware"
authors:
- user: Titus-von-Koeller
- user: jiaweizhao
  guest: true
- user: mdouglas
  guest: true
- user: ybelkada
- user: muellerzr
- user: amyeroberts
- user: smangrul
- user: BenjaminB
---

# GaLore: Advancing Large Model Training on Consumer-grade Hardware

The integration of GaLore into the training of large language models (LLMs) marks a significant advancement in the field of deep learning, particularly in terms of memory efficiency and the democratization of AI research. By allowing for the training of billion-parameter models on consumer-grade hardware, reducing memory footprint in optimizer states, and leveraging advanced projection matrix techniques, GaLore opens new horizons for researchers and practitioners with limited access to high-end computational resources.

## Scaling LLMs with Consumer-Grade Hardware

The capability of GaLore to facilitate the training of models with up to 7 billion parameters, such as those based on the Llama architecture, on consumer GPUs like the NVIDIA RTX 4090, is groundbreaking. This is achieved by significantly reducing the memory requirements traditionally associated with optimizer states and gradients during the training process. The approach leverages the inherent low-rank structure of gradients in deep neural networks, applying a projection that reduces the dimensionality of the data that needs to be stored and manipulated.

## Memory Efficiency in Optimizer States

The optimizer state, especially in adaptive optimization algorithms like Adam, represents a significant portion of the memory footprint during model training. GaLore addresses this by projecting the gradients into a lower-dimensional subspace before they are processed by the optimizer. This not only reduces the memory required to store these states but also maintains the effectiveness of the optimization process.

The memory savings are substantial, with [the authors reporting](https://x.com/AnimaAnandkumar/status/1765613815146893348?s=20) “more than **82.5% reduction in memory for storing optimizer states during training**”, making it feasible to train larger models or use larger batch sizes within the same memory constraints. When combined with 8-bit precision optimizers, these savings can be even more pronounced.

## Subspace Switching and Advanced Projection Techniques

A critical component of GaLore's effectiveness is its dynamic subspace switching mechanism, which allows the model to navigate through different low-rank subspaces throughout the training process. This ensures that the model is not confined to a limited portion of the parameter space, thus preserving the capacity for full-parameter learning. The decision on when and how to switch subspaces is pivotal, with the frequency of these switches being a balance between maintaining a consistent optimization trajectory and adapting to the evolving landscape of the gradient's low-rank structure.

The ability to dynamically adjust these projections in response to changes in the gradient structure is a potent tool in the GaLore arsenal, allowing for more nuanced control over the memory-optimization trade-offs inherent in training large models.

## Combining GaLore with 8-bit Optimizers

The combination of GaLore with 8-bit precision optimizers represents a synergy that maximizes memory efficiency while maintaining the integrity and performance of the training process. 8-bit optimizers reduce the memory footprint by quantizing the optimizer states. When used in conjunction with GaLore's projection mechanism, the result is a highly memory-efficient training regime that does not compromise on model accuracy or convergence speed.

This combination is particularly effective in scenarios where memory is a critical bottleneck, such as training large models on consumer-grade hardware or deploying models in memory-constrained environments. It enables the use of more complex models and larger datasets within the same hardware constraints, pushing the boundaries of what can be achieved with limited resources.

## Implementation Details

Integrating 8-bit optimizers with GaLore for training large language models (LLMs) involves quantizing the gradients, weights, and optimizer states to 8-bit representations. This quantization process significantly reduces the memory footprint, enabling the training of larger models or the use of larger batch sizes within the same memory constraints. The algorithmic details of this integration involve several key steps, some of which would benefit significantly from native CUDA implementation for efficiency gains. GaLore opens new possibilities to integrate these techniques even more tightly with quantization and specialized parameterization of the matrices, which can lead to further reductions in memory usage. We are currently exploring this direction in the bitsandbytes library.

### Algorithmic Overview of 8-bit Optimization with GaLore

**Gradient Projection**: GaLore projects the full-precision gradients into a low-rank subspace using projection matrices. This step reduces the dimensionality of the gradients, which are then quantized to 8-bit format.

**Quantization**: The projected gradients, along with the model weights and optimizer states (such as the moving averages in Adam), are quantized from 32-bit floating-point to 8-bit integer representations. This involves scaling the floating-point values to the 8-bit range and rounding them to the nearest integer.

**Optimizer Update**: The 8-bit quantized gradients are used to update the model weights. This step involves de-quantizing the gradients back to floating-point format, applying the optimizer's update rule (e.g., Adam's moment update and parameter adjustment), and then quantizing the updated optimizer states back to 8-bit for storage.

**De-quantization and Weight Update**: The 8-bit quantized weights undergo de-quantization to a floating-point representation for processing, albeit retaining the 8-bit precision inherent to their quantized form due to the limited range of values. This step is needed because standard operations in frameworks like PyTorch do not support 8-bit integers, and such integer weights cannot accommodate gradients. While this approach does not inherently enhance accuracy, it facilitates the practical application and gradient computation of quantized weights within the constraints of current deep learning libraries. Note that after de-quantization and before applying the weight update, GaLore employs one more projection that projects de-quantized low-rank updates back to the original space.

## Use it with Hugging Face Transformers

To use GaLore optimizers with the Hugging Face transformers library, you first need to update it to a version that supports GaLore optimizers, by either installing the latest update, i.e. `pip install transformers>=4.39.0` or installing transformers from source.

Then install the galore-torch library with `pip install galore-torch`. Below is a full working example of GaLore with transformers, for pretraining Mistral-7B on the imdb dataset:

```python
import torch
import datasets
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM
import trl

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw",
    optim_target_modules=["attn", "mlp"]
)

model_id = "mistralai/Mistral-7B-v0.1"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)

trainer.train()
```

`TrainingArguments`: Simply pass a valid `optim_target_modules` (it supports a single string, regex, or a list of strings or regexes) as well as, for `optim`, a valid GaLore optimizer, such as `galore_adamw`, `galore_adamw_8bit`, `galore_adafactor` – and you’re good to go!

### Layer-wise Updates

Another important point to mention are the _layer-wise_ optimizers (i.e. updating weights one layer at a time). Typically, the optimizer performs a single weight update for all layers after backpropagation. This is done by storing the entire weight gradients in memory. By adopting layer-wise weight updates, we can further reduce the memory footprint during training. Under the hood, this is implemented with PyTorch post-accumulation hooks on the layers the users want to update.

To use this feature, simply append `_layerwise` to the optimizer names, for example `galore_adamw_layerwise`.

## Conclusion

GaLore, with its innovative approach to leveraging the low-rank structure of gradients, represents a significant step forward in the memory-efficient training of LLMs. By enabling the training of billion-parameter models on consumer-grade hardware, reducing the memory footprint of optimizer states through projection techniques, and allowing for dynamic subspace switching, GaLore democratizes access to large-scale model training. The compatibility of GaLore with 8-bit precision optimizers further enhances its utility, offering a pathway to training larger and more complex models without the need for specialized computational resources. This opens up new possibilities for research and application in AI, making it an exciting time for practitioners and researchers alike.

## Resources

Please refer to [the original paper](https://arxiv.org/pdf/2403.03507.pdf). Twitter references: [1](https://twitter.com/AnimaAnandkumar/status/1765613815146893348) [2](https://x.com/_akhaliq/status/1765598376312152538?s=20) [3](https://x.com/tydsh/status/1765628222308491418?s=20). The paper also draws comparisons between GaLore and ReLoRA, which might be of interest to some readers. For readers with questions that remain unanswered, especially after review of the paper, or who would like to constructively discuss the results, please feel free to [join the author’s Slack community](https://galore-social.slack.com/join/shared_invite/zt-2ev152px0-DguuQ5WRTLQjtq2C88HBvQ#/shared-invite/email). For those interested in further releases along these lines, please follow [Jiawei Zhao](https://twitter.com/jiawzhao) and [Titus von Koeller](https://twitter.com/Titus_vK) (for information on the latest `bitsandbytes` releases) as well as [Younes Belkada](https://twitter.com/younesbelkada) for the latest and greatest infos on quantization-related topics within and around the Hugging Face ecosystem.


