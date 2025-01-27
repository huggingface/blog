---
title: "StackLLaMA: A hands-on guide to train LLaMA with RLHF" 
thumbnail: /blog/assets/138_stackllama/thumbnail.png
authors:
- user: edbeeching
- user: kashif
- user: ybelkada
- user: lewtun
- user: lvwerra
- user: nazneen
- user: natolambert
---

# StackLLaMA: A hands-on guide to train LLaMA with RLHF


Models such as [ChatGPT]([https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt)), [GPT-4]([https://openai.com/research/gpt-4](https://openai.com/research/gpt-4)), and [Claude]([https://www.anthropic.com/index/introducing-claude](https://www.anthropic.com/index/introducing-claude)) are powerful language models that have been fine-tuned using a method called Reinforcement Learning from Human Feedback (RLHF) to be better aligned with how we expect them to behave and would like to use them.

In this blog post, we show all the steps involved in training a [LlaMa model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai) to answer questions on [Stack Exchange](https://stackexchange.com) with RLHF through a combination of:

- Supervised Fine-tuning (SFT)
- Reward / preference modeling (RM)
- Reinforcement Learning from Human Feedback (RLHF)

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/instructGPT.png)
*From InstructGPT paper: Ouyang, Long, et al. "Training language models to follow instructions with human feedback." arXiv preprint arXiv:2203.02155 (2022).*

By combining these approaches, we are releasing the StackLLaMA model. This model is available on the [🤗 Hub](https://huggingface.co/trl-lib/llama-se-rl-peft) (see [Meta's LLaMA release](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) for the original LLaMA model) and [the entire training pipeline](https://huggingface.co/docs/trl/index) is available as part of the Hugging Face TRL library. To give you a taste of what the model can do, try out the demo below!

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>

<gradio-app theme_mode="light" src="https://trl-lib-stack-llama.hf.space"></gradio-app>

## The LLaMA model

When doing RLHF, it is important to start with a capable model: the RLHF step is only a fine-tuning step to align the model with how we want to interact with it and how we expect it to respond.  Therefore, we choose to use the recently introduced and performant [LLaMA models](https://arxiv.org/abs/2302.13971). The LLaMA models are the latest large language models developed by Meta AI. They come in sizes ranging from 7B to 65B parameters and were trained on between 1T and 1.4T tokens, making them very capable. We use the 7B model as the base for all the following steps!
To access the model, use the [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) from Meta AI.

## Stack Exchange dataset

Gathering human feedback is a complex and expensive endeavor. In order to bootstrap the process for this example while still building a useful model, we make use of the [StackExchange dataset](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences). The dataset includes questions and their corresponding answers from the StackExchange platform (including StackOverflow for code and many other topics). It is attractive for this use case because the answers come together with the number of upvotes and a label for the accepted answer.

We follow the approach described in [Askell et al. 2021](https://arxiv.org/abs/2112.00861) and assign each answer a score:

`score = log2 (1 + upvotes) rounded to the nearest integer, plus 1 if the questioner accepted the answer (we assign a score of −1 if the number of upvotes is negative).`

For the reward model, we will always need two answers per question to compare, as we’ll see later. Some questions have dozens of answers, leading to many possible pairs. We sample at most ten answer pairs per question to limit the number of data points per question. Finally, we cleaned up formatting by converting HTML to Markdown to make the model’s outputs more readable. You can find the dataset as well as the processing notebook [here](https://huggingface.co/datasets/lvwerra/stack-exchange-paired).


## Efficient training strategies

Even training the smallest LLaMA model requires an enormous amount of memory. Some quick math: in bf16, every parameter uses 2 bytes (in fp32 4 bytes) in addition to 8 bytes used, e.g., in the Adam optimizer (see the [performance docs](https://huggingface.co/docs/transformers/perf_train_gpu_one#optimizer) in Transformers for more info). So a 7B parameter model would use `(2+8)*7B=70GB` just to fit in memory and would likely need more when you compute intermediate values such as attention scores. So you couldn’t train the model even on a single 80GB A100 like that. You can use some tricks, like more efficient optimizers of half-precision training, to squeeze a bit more into memory, but you’ll run out sooner or later.

Another option is to use Parameter-Efficient Fine-Tuning (PEFT) techniques, such as the [`peft`](https://github.com/huggingface/peft) library, which can perform Low-Rank Adaptation (LoRA) on a model loaded in 8-bit. 

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/lora-animated.gif)   
*Low-Rank Adaptation of linear layers: extra parameters (in orange) are added next to the frozen layer (in blue), and the resulting encoded hidden states are added together with the hidden states of the frozen layer.*

Loading the model in 8bit reduces the memory footprint drastically since you only need one byte per parameter for the weights (e.g. 7B LlaMa is 7GB in memory). Instead of training the original weights directly, LoRA adds small adapter layers on top of some specific layers (usually the attention layers); thus, the number of trainable parameters is drastically reduced.

In this scenario, a rule of thumb is to allocate ~1.2-1.4GB per billion parameters (depending on the batch size and sequence length) to fit the entire fine-tuning setup. As detailed in the attached blog post above, this enables fine-tuning larger models (up to 50-60B scale models on a NVIDIA A100 80GB) at low cost. 

These techniques have enabled fine-tuning large models on consumer devices and Google Colab. Notable demos are fine-tuning `facebook/opt-6.7b` (13GB in `float16` ), and `openai/whisper-large` on Google Colab (15GB GPU RAM). To learn more about using `peft`, refer to our [github repo](https://github.com/huggingface/peft) or the [previous blog post](https://huggingface.co/blog/trl-peft)(https://huggingface.co/blog/trl-peft)) on training 20b parameter models on consumer hardware.

Now we can fit very large models into a single GPU, but the training might still be very slow. The simplest strategy in this scenario is data parallelism: we replicate the same training setup into separate GPUs and pass different batches to each GPU. With this, you can parallelize the forward/backward passes of the model and scale with the number of GPUs. 

![chapter10_ddp.png](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/chapter10_ddp.png)

We use either the `transformers.Trainer` or `accelerate`, which both support data parallelism without any code changes, by simply passing arguments when calling the scripts with `torchrun` or `accelerate launch`. The following runs a training script with 8 GPUs on a single machine with `accelerate` and `torchrun`, respectively.

```bash
accelerate launch --multi_gpu --num_machines 1  --num_processes 8 my_accelerate_script.py
torchrun --nnodes 1  --nproc_per_node 8 my_torch_script.py
```

## Supervised fine-tuning

Before we start training reward models and tuning our model with RL, it helps if the model is already good in the domain we are interested in. In our case, we want it to answer questions, while for other use cases, we might want it to follow instructions, in which case instruction tuning is a great idea. The easiest way to achieve this is by continuing to train the language model with the language modeling objective on texts from the domain or task. The [StackExchange dataset](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) is enormous (over 10 million instructions), so we can easily train the language model on a subset of it.

There is nothing special about fine-tuning the model before doing RLHF - it’s just the causal language modeling objective from pretraining that we apply here. To use the data efficiently, we use a technique called packing: instead of having one text per sample in the batch and then padding to either the longest text or the maximal context of the model, we concatenate a lot of texts with a EOS token in between and cut chunks of the context size to fill the batch without any padding.

![chapter10_preprocessing-clm.png](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/chapter10_preprocessing-clm.png)

With this approach the training is much more efficient as each token that is passed through the model is also trained in contrast to padding tokens which are usually masked from the loss. If you don't have much data and are more concerned about occasionally cutting off some tokens that are overflowing the context you can also use a classical data loader.

The packing is handled by the `ConstantLengthDataset` and we can then use the `Trainer` after loading the model with `peft`. First, we load the model in int8, prepare it for training, and then add the LoRA adapters.

```python
# load model in 8bit
model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        device_map={"": Accelerator().local_process_index}
    )
model = prepare_model_for_int8_training(model)

# add LoRA to model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

We train the model for a few thousand steps with the causal language modeling objective and save the model. Since we will tune the model again with different objectives, we merge the adapter weights with the original model weights.

**Disclaimer:** due to LLaMA's license, we release only the adapter weights for this and the model checkpoints in the following sections. You can apply for access to the base model's weights by filling out Meta AI's [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) and then converting them to the 🤗 Transformers format by running this [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). Note that you'll also need to install 🤗 Transformers from source until the `v4.28` is released.

Now that we have fine-tuned the model for the task, we are ready to train a reward model.

## Reward modeling and human preferences

In principle, we could fine-tune the model using RLHF directly with the human annotations. However, this would require us to send some samples to humans for rating after each optimization iteration. This is expensive and slow due to the number of training samples needed for convergence and the inherent latency of human reading and annotator speed.

A trick that works well instead of direct feedback is training a reward model on human annotations collected before the RL loop. The goal of the reward model is to imitate how a human would rate a text. There are several possible strategies to build a reward model: the most straightforward way would be to predict the annotation (e.g. a rating score or a binary value for “good”/”bad”). In practice, what works better is to predict the ranking of two examples, where the reward model is presented with two candidates \\( (y_k, y_j) \\) for a given prompt \\( x \\) and has to predict which one would be rated higher by a human annotator.

This can be translated into the following loss function:


\\( \operatorname{loss}(\theta)=- E_{\left(x, y_j, y_k\right) \sim D}\left[\log \left(\sigma\left(r_\theta\left(x, y_j\right)-r_\theta\left(x, y_k\right)\right)\right)\right] \\)

where \\( r \\) is the model’s score and \\( y_j \\) is the preferred candidate.

With the StackExchange dataset, we can infer which of the two answers was preferred by the users based on the score. With that information and the loss defined above, we can then modify the `transformers.Trainer` by adding a custom loss function. 

```python
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"],  attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
```

We utilize a subset of a 100,000 pair of candidates and evaluate on a held-out set of 50,000. With a modest training batch size of 4, we train the LLaMA model using the LoRA `peft` adapter for a single epoch using the Adam optimizer with BF16 precision. Our LoRA configuration is:

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
```

The training is logged via [Weights & Biases](https://wandb.ai/krasul/huggingface/runs/wmd8rvq6?workspace=user-krasul) and took a few hours on 8-A100 GPUs using the 🤗 research cluster and the model achieves a final **accuracy of 67%**. Although this sounds like a low score, the task is also very hard, even for human annotators.

As detailed in the next section, the resulting adapter can be merged into the frozen model and saved for further downstream use.

## Reinforcement Learning from Human Feedback

With the fine-tuned language model and the reward model at hand, we are now ready to run the RL loop. It follows roughly three steps:

1. Generate responses from prompts
2. Rate the responses with the reward model
3. Run a reinforcement learning policy-optimization step with the ratings

![Untitled](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/trl_loop.png)

The Query and Response prompts are templated as follows before being tokenized and passed to the model:

```bash
Question: <Query>

Answer: <Response>
```

The same template was used for SFT, RM and RLHF stages.

A common issue with training the language model with RL is that the model can learn to exploit the reward model by generating complete gibberish, which causes the reward model to assign high rewards. To balance this, we add a penalty to the reward: we keep a reference of the model that we don’t train and compare the new model’s generation to the reference one by computing the KL-divergence:


\\( \operatorname{R}(x, y)=\operatorname{r}(x, y)- \beta \operatorname{KL}(x, y) \\)

where \\( r \\) is the reward from the reward model and  \\( \operatorname{KL}(x,y) \\) is the KL-divergence between the current  policy and the reference model. 

Once more, we utilize `peft` for memory-efficient training, which offers an extra advantage in the RLHF context. Here, the reference model and policy share the same base, the SFT model, which we load in 8-bit and freeze during training. We exclusively optimize the policy's LoRA weights using PPO while sharing the base model's weights.

```python
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]
		
	# sample from the policy and generate responses
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
	# Log stats to WandB
    ppo_trainer.log_stats(stats, batch, rewards)
```

We train for 20 hours on 3x8 A100-80GB GPUs, using the 🤗 research cluster, but you can also get decent results much quicker (e.g. after ~20h on 8 A100 GPUs). All the training statistics of the training run are available on [Weights & Biases](https://wandb.ai/lvwerra/trl/runs/ie2h4q8p).

![Per batch reward at each step during training. The model’s performance plateaus after around 1000 steps.](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/wandb_reward.png)
*Per batch reward at each step during training. The model’s performance plateaus after around 1000 steps.*

So what can the model do after training? Let's have a look!

![llama prompt](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/llama_prompt.png)

Although we shouldn't trust its advice on LLaMA matters just, yet, the answer looks coherent and even provides a Google link. Let's have a look and some of the training challenges next.

## Challenges, instabilities and workarounds

Training LLMs with RL is not always plain sailing. The model we demo today is the result of many experiments, failed runs and hyper-parameter sweeps. Even then, the model is far from perfect. Here we will share a few of the observations and headaches we encountered on the way to making this example.

### Higher reward means better performance, right?

![Wow this run must be great, look at that sweet, sweet, reward!](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/logs_high_reward.png)
*Wow this run must be great, look at that sweet, sweet, reward!*

In general in RL, you want to achieve the highest reward. In RLHF we use a Reward Model, which is imperfect and given the chance, the PPO algorithm will exploit these imperfections. This can manifest itself as sudden increases in reward, however when we look at the text generations from the policy, they mostly contain repetitions of the string ```, as the reward model found the stack exchange answers containing blocks of code usually rank higher than ones without it. Fortunately this issue was observed fairly rarely and in general the KL penalty should counteract such exploits.

### KL is always a positive value, isn’t it?

As we previously mentioned, a KL penalty term is used in order to push the model’s outputs remain close to that of the base policy. In general, KL divergence measures the distances between two distributions and is always a positive quantity. However, in `trl` we use an estimate of the KL which in expectation is equal to the real KL divergence.


\\( KL_{pen}(x,y) = \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right) \\)

Clearly, when a token is sampled from the policy which has a lower probability than the SFT model, this will lead to a negative KL penalty, but on average it will be positive otherwise you wouldn't be properly sampling from the policy. However, some generation strategies can force some tokens to be generated or some tokens can suppressed. For example when generating in batches finished sequences are padded and when setting a minimum length the EOS token is suppressed. The model can assign very high or low probabilities to those tokens which leads to negative KL. As the PPO algorithm optimizes for reward, it will chase after these negative penalties, leading to instabilities.

![Negative KL](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/logs_neg_kl.png)

One needs to be careful when generating the responses and we suggest to always use a simple sampling strategy first before resorting to more sophisticated generation methods.

### Ongoing issues

There are still a number of issues that we need to better understand and resolve. For example, there are occassionally spikes in the loss, which can lead to further instabilities. 

![Loss spikes](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/logs_loss_spikes.png)

As we identify and resolve these issues, we will upstream the changes `trl`, to ensure the community can benefit.

## Conclusion

In this post, we went through the entire training cycle for RLHF, starting with preparing a dataset with human annotations, adapting the language model to the domain, training a reward model, and finally training a model with RL. 

By using `peft`, anyone can run our example on a single GPU! If training is too slow, you can use data parallelism with no code changes and scale training by adding more GPUs.

For a real use case, this is just the first step! Once you have a trained model, you must evaluate it and compare it against other models to see how good it is. This can be done by ranking generations of different model versions, similar to how we built the reward dataset. 

Once you add the evaluation step, the fun begins: you can start iterating on your dataset and model training setup to see if there are ways to improve the model. You could add other datasets to the mix or apply better filters to the existing one. On the other hand, you could try different model sizes and architecture for the reward model or train for longer.

We are actively improving TRL to make all steps involved in RLHF more accessible and are excited to see the things people build with it! Check out the [issues on GitHub](https://github.com/lvwerra/trl/issues) if you're interested in contributing.


## Citation

```bibtex
@misc {beeching2023stackllama,
    author       = { Edward Beeching and
                     Younes Belkada and
                     Kashif Rasul and
                     Lewis Tunstall and
                     Leandro von Werra and
                     Nazneen Rajani and
                     Nathan Lambert
                   },
    title        = { StackLLaMA: An RL Fine-tuned LLaMA Model for Stack Exchange Question and Answering },
    year         = 2023,
    url          = { https://huggingface.co/blog/stackllama },
    doi          = { 10.57967/hf/0513 },
    publisher    = { Hugging Face Blog }
}
```

## Acknowledgements

We thank Philipp Schmid for sharing his wonderful [demo](https://huggingface.co/spaces/philschmid/igel-playground) of streaming text generation upon which our demo was based. We also thank Omar Sanseviero and Louis Castricato for giving valuable and detailed feedback on the draft of the blog post.
