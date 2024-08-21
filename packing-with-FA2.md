---
title: "Improving Hugging Face Training Efficiency Through Packing with Flash Attention" 
thumbnail: /blog/assets/packing-with-FA2/thumbnail.gif
authors:
- user: RhuiDih
  guest: true
  org: ibm
- user: ArthurZucker
- user: achikundu
  guest: true
  org: ibm
- user: wynterl
  guest: true
  org: ibm
- user: raghukiran1224
  guest: true
  org: ibm
- user: mayank31398
  guest: true
  org: ibm
---


## TL;DR

Packing instruction tuning examples whilst still availing Flash Attention is now available in Hugging Face, thanks to a [recent PR](https://github.com/huggingface/transformers/pull/31629) and the new [DataCollatorWithFlattening](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithFlattening)
 
Users will find it can provide up to 2x improvement in training throughput while maintaining convergence quality.

## Introduction
It is well known that packing small examples together improves the computational efficiency of training. However, previous implementations of Flash Attention 2 did not consider example boundaries during packing, leading to potential issues in instruction tuning. In instruction tuning, it is important for the masking mechanism to be aware of the example boundaries if the examples are packed together to avoid undesirable cross-example-attention. 
Hugging Face Transformers now address this with a new feature that maintains boundary awareness during packing, alongside the introduction of a new data collator, `DataCollatorWithFlattening`.

By selecting `DataCollatorWithFlattening`, Hugging Face `Trainer` users can now seamlessly concatenate sequences into a single tensor while accounting for sequence boundaries during Flash Attention computations. This is achieved through the `flash_attn_varlen_func`, which calculates each mini-batch's cumulative sequence lengths (`cu_seqlens`).

## Up to 2x throughput increase 

We see significant improvement in training throughput using this feature with the new `DataCollatorWithFlattening`. The figure below shows the throughput measured in tokens/second during training. In this example, the throughput is the per-GPU average over 8 A100-80 GPU over one epoch of a 20K randomly selected sample from two different instruct tuning datasets, FLAN and OrcaMath. 

![throughput](https://github.com/user-attachments/assets/09248359-5aa2-4b36-b896-ba76f98ecbfa)

![throughput](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/thruput.png)


FLAN has short sequences on average but a large variance in sequence length, so that example lengths in each batch may vary widely. This means that padded FLAN batches may require a considerable amount of padding. Training on the FLAN dataset shows a significant benefit from packing with position IDs in terms of increased throughput. We see a 2x throughput increase on the models shown here: llama2-7B, mistral-7B, and granite-8B-code. 

OrcaMath has somewhat longer examples and a much lower variance in example length. As such, the improvement from packing is somewhat lower. Our experiments show a 1.4x increase in throughput when training using this form of packing on the OrcaMath dataset across these three models.

![memory](https://github.com/user-attachments/assets/377caa9c-cef5-4472-9128-85eb158faebf)

![memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/memory.png)


Memory usage also improves through packing with position_ids. The following figure shows the peak memory usage of the same three models training on the same two datasets. Peak memory is reduced by 20% on the FLAN dataset, which benefits considerably from packing. 

That number is a more modest 6% on the OrcaMath dataset with its more homogeneous example lengths.

![ValLoss](https://github.com/user-attachments/assets/3fc30fd6-85a8-4f76-a644-7a0a7f16487d)

![ValLoss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/ValLoss.png)



Packing examples, when it reduces the number of optimization steps, may harm training convergence. The new feature, however, retains the minibatches and, hence, the same number of optimization steps as would be used with padded examples. Thus, there is no impact on train convergence, as we see in the next figure, which shows identical validation loss of the same three models training on the same two datasets, whether the models are trained with padding or packing with `position_ids`.

## How it works 
Consider a batch of data with a batchsize = 4 where the four sequences are as follows:
[10,11,12,13] ; [20,21,22,23,24,25,26,27] ; [30,31,32,33,34] and [40,41,42,43,44,45,46,47,48,49,401].

After concatenating the examples, the padding-free collator returns the input IDs, labels, and the `position_ids` of each example. Hence, the collator provides, for this example,  

input_ids = [[10,11,12,13,20,21,22,23,24,25,26,27,30,31,32,33,34,40,41,42,43,44,45,46,47,48,49,401]]

labels = [[-100,11,12,13,-100,21,22,23,24,25,26,27,-100,31,32,33,34,-100,41,42,43,44,45,46,47,48,49,401]] and

position_ids = [[0,1,2,3,0,1,2,3,4,5,6,7,0,1,2,3,4,0,1,2,3,4,5,6,7,8,9,10]]

The modifications required are lightweight and are limited to providing the `position_ids` to Flash Attention. 

This relies, however, on the model exposing `position_ids`. As of the time of writing, 14 models expose them and are supported by the solution. Specifically, Llama 2 and 3, Mistral, Mixtral, Granite, DBRX, Falcon, Gemma, OLMo, Phi 1, 2, and 3, phi3, Qwen 2 and 2 MoE, StableLM, and StarCoder 2 are all supported by the solution.

## Getting Started
Reaping the benefits of packing with `position_ids` is easy. To use packing with `position_ids`, only two steps are required:

1) Instantiate the model with Flash Attention 2
2) Use the new `DataCollatorWithFlattening`
   
## How To Use It

![image1](https://github.com/user-attachments/assets/43790e8c-c2ca-4bc3-98ce-f06169624b2d)
![image2](https://github.com/user-attachments/assets/6a77f17d-9289-4850-b293-543aa67f7d2e)

![image1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/image1.png)
![image2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/packing-with-FA2/image2.png)

## Conclusions

Packing instruction tuning examples with correct within-example-attention, while still availing of Flash Attention 2, is now available in Hugging Face, thanks to a recent PR Enhancing SFT Training Efficiency Using Packing and FlashAttention2 with Position ID and the new `DataCollatorWithFlattening`. Benefits can be seen in throughput during training and peak memory usage. There is no degradation in training convergence, making it a win-win. Actual throughput and memory improvement depends on the model and the distribution of example lengths in the training data. Training with data that has a wide variation of example lengths will see the greatest benefit, with respect to padding, from using packing with `position_ids`.

For a more detailed analysis, have a look at the paper at https://arxiv.org/abs/2407.09105


