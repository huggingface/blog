# Improving Hugging Face Training Efficiency Through Packing with Flash Attention

## TL;DR
Packing instruction tuning examples whilst still availing Flash Attention is now available in Hugging Face, thanks to a recent PR (https://github.com/huggingface/transformers/pull/31629/commits/8120b3a3c4077ed699b029cbc37256bf3c58b6e9#top)  and the new DataCollatorWithFlattening (https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithFlattening). 
Users will find it can provide up to 2x improvement in training throughput while maintaining convergence quality.

## Introduction
It is well known that packing small examples together improves the computational efficiency of training. Previously, however, packing used with Flash Attention 2 did not take into account example boundaries. In instruction tuning, it is important for the masking mechanism to be aware of the example boundaries if the examples are packed together to avoid undesirable cross-example-attention. This feature is now available in Hugging Face Transformers, along with a new data_collator called DataCollatorWithFlattening.

By selecting DataCollatorWithFlattening, Hugging Face trainer users can now seamlessly concatenate sequences into a single tensor while accounting for sequence boundaries during Flash Attention computation. This is accomplished by using the ‘flash_attn_varlen_func’ by computing the cumulative sequence length ‘cu_seq-Lens’  of each mini-batch.

## Up to 2x throughput increase 

We see significant improvement in training throughput using this feature with the new DataCollatorWithFlattening. The figure below shows the throughput, as measured in tokens/second during training. In this example, the throughput is the per-GPU average over 8 A100-80 GPU, over one epoch of a 20K randomly-selected sample from two different instruct tuning datasets, FLAN and OrcaMath. 

![thruput](https://github.com/user-attachments/assets/09248359-5aa2-4b36-b896-ba76f98ecbfa)
FLAN has short sequences on average but a large variance in sequence length, so that example lengths in each batch may vary widely. This means that padded FLAN batches may require a considerable amount of padding. Training on the FLAN data set hence shows significant benefit from packing with position IDs in terms of increased throughput. We see a 2x throughput increase on the models shown here: llama2-7B, mistral-7B and granite-8B-code. 

OrcaMath, has somewhat longer examples and a much lower variance in example length. As such, the improvement from packing is also somewhat lower. Our experiments show about a 1.4x increase in throughput using this form of packing training on OrcaMath across these three models.

![memory](https://github.com/user-attachments/assets/377caa9c-cef5-4472-9128-85eb158faebf)
Memory usage also improves through the use of packing with position_ids. The next figure shows the peak memory usage of the same three models training on the same two datasets. Again, on the FLAN dataset which benefits considerably from packing, peak memory is reduced by around 20%. 

That number is a more modest 6% on the OrcaMath dataset with its more homogeneous example lengths.

![ValLoss](https://github.com/user-attachments/assets/3fc30fd6-85a8-4f76-a644-7a0a7f16487d)
Packing examples, when it reduces the number of optimisation steps, may have a detrimental impact on training convergence. The new feature, however, retains the minibatches and hence the same number of optimisation steps as would be used with padded examples. Thus, there is no impact on train convergence, as we see in the next figure that shows identical validation loss of the same three models training on the same two datasets, whether the models are trained with padding or packing with position_ids.


## Here’s how it works 
Consider a batch of data with a batchsize = 4 where the four sequences are as follows:
[10,11,12,13] ; [20,21,22,23,24,25,26,27] ; [30,31,32,33,34] and [40,41,42,43,44,45,46,47,48,49,401].

The padding-free collator returns the input IDs, labels, and the position_ids of each example after concatenating the examples together. Hence, the collator provides, for this example, 

input_ids: [[10,11,12,13,20,21,22,23,24,25,26,27,30,31,32,33,34,40,41,42,43,44,45,46,47,48,49,401]]

labels: [[-100,11,12,13,-100,21,22,23,24,25,26,27,-100,31,32,33,34,-100,41,42,43,44,45,46,47,48,49,401]] and

position_ids:[[0,1,2,3,0,1,2,3,4,5,6,7,0,1,2,3,4,0,1,2,3,4,5,6,7,8,9,10]]

The modifications required are lightweight and are limited to providing the position_ids to Flash Attention. 

This relies, however, on the model exposing position_ids. As of the time of writing, 14 models expose position IDs and are supported by the solution. Specifically: llama2/llama3, mistral, mixtral, granite, dbrx, falcon, gemma, olmo, phi/phi2, phi3, qwen2, qwen2_moe, stablelm, starcoder2 are all supported by the solution.

Getting Started
Reaping the benefits of packing with position_ids is easy. To use packing with position_ids, only two steps are required:

1) Instantiate the model with flash attention 2
2) Use the new DataCollatorWithFlattening

## Here’s how to do that:

![image1](https://github.com/user-attachments/assets/43790e8c-c2ca-4bc3-98ce-f06169624b2d)
![image2](https://github.com/user-attachments/assets/6a77f17d-9289-4850-b293-543aa67f7d2e)



## Conclusions
Packing instruction tuning examples with correct within-example-attention, whilst still availing of Flash Attention 2, is now available in Hugging Face, thanks to a recent PR Enhancing SFT Training Efficiency Using Packing and FlashAttention2 with Position ID and the new DataCollatorWithFlattening. Benefits can be seen in throughput during training and peak memory usage. There is no degradation in training convergence, making it a win-win. Actual throughput and memory improvement depend both on the model and the distribution of example lengths in the training data. Training with data that has a wide variation of example lengths will see the greatest benefit, with respect to padding, from using packing with position_ids.


For more detailed analysis, have a look at the paper at  https://arxiv.org/pdf/2407.09105


