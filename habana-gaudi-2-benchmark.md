---
title: "Faster Training and Inference: Habana Gaudi 2 vs Nvidia A100 80GB"
thumbnail: /blog/assets/habana-gaudi-2-benchmark/thumbnail.png
---

# Faster Training and Inference: Habana Gaudi 2 vs Nvidia A100 80GB

<div class="blog-metadata">
    <small>Published December 9, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/habana-gaudi-2-benchmark.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/regisss">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1644920200150-620b7c408f5871b8a1a168a7.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>regisss</code>
            <span class="fullname">Régis Pierrard</span>
        </div>
    </a>
</div>

In this article, you will learn how to start using [Habana Gaudi 2](https://habana.ai/training/gaudi2/) to accelerate model training and inference and train bigger models with 🤗 [Optimum Habana](https://huggingface.co/docs/optimum/habana/index). Then, several benchmarks will be presented to assess the performance gap between Gaudi 1, Gaudi 2 and Nvidia A100 80GB.

[Gaudi 2](https://habana.ai/training/gaudi2/) is the second-generation AI hardware accelerator designed by Habana. One accelerator contains 8 devices with 96GB of memory each (against 32GB on Gaudi 1 and 80GB on A100 80GB). The SDK, [SynapseAI](https://developer.habana.ai/), is common to both Gaudi 1 and Gaudi 2.
That means that 🤗 Optimum Habana, which offers a very user-friendly interface between the 🤗 Transformers and 🤗 Diffusers libraries and Gaudi, **works the exact same way on Gaudi 2 as on Gaudi 1!**
So if you already have ready-to-use training or inference workflows for Gaudi 1, we encourage you to try them on Gaudi 2 as they will work without any single change.


## How to Get Access to Gaudi 2?

To start using Gaudi 2, you should follow the following steps:

1. Go to the [Intel Developer Zone](https://www.intel.com/content/www/us/en/my-intel/developer-sign-in.htm) and sign in to your account or register if you do not have one.

2. Go to the [Intel Developer Cloud management console](https://scheduler.cloud.intel.com/#/systems).

3. Select *Habana Gaudi2 Deep Learning Server featuring eight Gaudi2 HL-225H mezzanine cards and latest Intel® Xeon® Processors* and click on *Launch Instance* in the lower right corner as shown below.
<figure class="image table text-center m-0 w-full">
  <img src="assets/habana-gaudi-2-benchmark/launch_instance.png" alt="Cloud Architecture"/>
</figure>

4. You can then request an instance:
<figure class="image table text-center m-0 w-full">
  <img src="assets/habana-gaudi-2-benchmark/request_instance.png" alt="Cloud Architecture"/>
</figure>

5. Once your request is validated, re-do the step 3 and click on *Add OpenSSH Publickey* to add a payment method (credit card or promotion code) and a SSH publick key that you can generate with `ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa`. You may be redirected to step 3 each time you add a payment method or a SSH public key.

6. Re-do step 3 and then click on *Launch Instance*. You will have to accept the proposed general conditions to actually launch the instance.

7. Go to the [Intel Developer Cloud management console](https://scheduler.cloud.intel.com/#/systems) and click on the tab called *View Instances*.

8. You can copy the SSH command to access your Gaudi 2 instance remotely!

> If you terminate the instance and want to use Gaudi 2 again, you will have to re-do the whole process.

You can find more information about this process [here](https://scheduler.cloud.intel.com/public/Intel_Developer_Cloud_Getting_Started.html).


## Benchmarks

Several benchmarks were performed to assess the abilities of Gaudi 1/2 and A100 80GB for both training and inference and for models of various sizes.


### Pre-Training BERT

A few months ago, [Philipp Schmid](https://huggingface.co/philschmid) presented you [how to pre-train BERT on Gaudi with 🤗 Optimum Habana](https://huggingface.co/blog/pretraining-bert). 65k training steps were performed with a batch size of 32 samples per device (so 8*32=256 in total) for a total training time of 8 hours and 53 minutes (you can see the Tensorboard logs of this run [here](https://huggingface.co/philschmid/bert-base-uncased-2022-habana-test-6/tensorboard?scroll=1#scalars)).

We re-ran the same script with the same hyperparameters on Gaudi 2 and we got a total training time of 2 hours and 55 minutes (see the logs [here](https://huggingface.co/regisss/bert-pretraining-gaudi-2-batch-size-32/tensorboard?scroll=1#scalars)). **That makes a x3.04 speedup without changing anything.**

Since Gaudi 2 has roughly 3 times more memory per device as Gaudi 1, it is possible to take advantage of this to have bigger batches. This will give HPUs more work to do and will also enable to try a range of hyperparameter values that was not reachable with Gaudi 1. With a batch size of 64 samples per device (so 512 in total), we got after 20k steps a similar loss convergence to the 65k steps of the previous runs. That makes a total training time of 1 hour and 33 minutes (see the logs [here](https://huggingface.co/regisss/bert-pretraining-gaudi-2-batch-size-64/tensorboard?scroll=1#scalars)). The throughput is x1.16 higher with this configuration, while this new batch size strongly accelerates convergence.
**Overall, with Gaudi 2, the total training time is reduced by a 5.75 factor and the throughput is x3.53 higher compared to Gaudi 1**.

**Gaudi 2 also offers a speedup over A100**: x1.61 for a batch size of 32 and x1.87 for a batch size of 64. We also observed that the throughput of A100 with a batch size of 64 is not better than the one we got with a batch size of 32.

Here is a table that displays the throughputs we got for Gaudi 1, Gaudi 2 and Nvidia A100 80GB GPUs:

<center>

|   | Gaudi 1 (BS=32) | Gaudi 2 (BS=32) | Gaudi 2 (BS=64) | A100 (BS=32) |
|:-:|:---------------:|:---------------:|:---------------:|:------------:|
| Throughput (samples/s) | 520.2 | 1580.2 | 1835.8 | 981.6 |
| Speedup | x1.0 | x3.04 | x3.53 | x1.89 |

</center>

*BS* is the batch size per device. The Gaudi runs were performed in mixed precision (bf16/fp32) and the A100 ones in fp16.


### Generating Images From Text With Stable Diffusion

One of the main new features of 🤗 Optimum Habana 1.3 is [the support for Stable Diffusion](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion). It is now very easy to generate images from text on Gaudi. Unlike with 🤗 Diffusers on GPU, images are generated by batches. Due to model compilation times, the first two batches will be slower than the following iterations. In this benchmark, these first two iterations were discarded to compute the throughputs for both Gaudi 1 and Gaudi 2.

[This script](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion) was run for batch sizes of 4 and 8 samples and returned equal latencies for both. It uses the [`Habana/stable-diffusion`](https://huggingface.co/Habana/stable-diffusion) Gaudi configuration.

The results we got are displayed in the table below.
**Gaudi 2 showcases latencies that are x3.65 faster than Gaudi 1 and x2.21 faster than Nvidia A100.** It can also support bigger batch sizes.

<center>

|   | Gaudi 1 (BS=4/8) | Gaudi 2 (BS=4/8) | A100 (BS=1) |
|:-:|:----------------:|:----------------:|:-----------:|
| Latency (s/img) | 4.34 | 1.19 | 2.63 |
| Speedup | x1.0 | x3.65 | x1.65 |

</center>

*BS* is the batch size.
The Gaudi runs were performed in mixed precision (bf16/fp32) and the A100 ones in fp16 (more information [here](https://huggingface.co/docs/diffusers/optimization/fp16)).


### Fine-tuning T5-3B

With 98 GB of memory per device, Gaudi 2 enables to run much bigger models. For instance, we managed to fine-tune T5-3B (which contains 3 billion parameters) with gradient checkpointing being the only applied memory optimization. This is not possible on Gaudi 1.
[Here](https://huggingface.co/regisss/t5-3b-summarization-gaudi-2/tensorboard?scroll=1#scalars) are the logs of this run where the model was fine-tuned on the CNN Dailymail dataset for text summarization using [this script](https://github.com/huggingface/optimum-habana/tree/main/examples/summarization).

The results we got are presented in the table below. **Gaudi 2 is x2.44 faster than A100 80GB.** We observe that we cannot fit a batch size larger than 1 on Gaudi 2 here. This is due to the memory space taken by the graph where operations are accumulated during the first iteration of the run. To go further, we are looking forward to expanding this benchmark to [DeepSpeed](https://www.deepspeed.ai/) to see if the same trend holds.

<center>

|   | Gaudi 1 | Gaudi 2 (BS=1) | A100 (BS=16) |
|:-:|:-------:|:--------------:|:------------:|
| Throughput (samples/s) | N/A | 19.7 | 8.07 |
| Speedup | / | x2.44 | x1.0 |

</center>

*BS* is the batch size per device. Gaudi 2 and A100 runs were performed in fp32 with gradient checkpointing enabled.


## Conclusion

In this article, you have seen that Habana Gaudi 2 significantly improves Gaudi 1 and is about twice as fast as Nvidia A100 80GB for both training and inference.

You also know now how to setup a Gaudi 2 instance through the Intel Developer Zone. Check out the [examples](https://github.com/huggingface/optimum-habana/tree/main/examples) you can easily run on such a device with 🤗 Optimum Habana.

If you are interested in accelerating your training and/or inference workflows, contact us to learn about our [Hardware Partner Program](https://huggingface.co/hardware) and our [Expert Acceleration Program](https://huggingface.co/support). To learn more about Habana solutions, [read about our partnership and how to contact them](https://huggingface.co/hardware/habana).

---

Thanks for reading! If you have any questions, feel free to contact me, either through [Github](https://github.com/huggingface/optimum-habana) or on the [forum](https://discuss.huggingface.co/c/optimum/59). You can also connect with me on [LinkedIn](https://www.linkedin.com/in/regispierrard/).
