---
title: "Accelerating Vision-Language Models: BridgeTower on Habana Gaudi2"
thumbnail: /blog/assets/bridgetower/thumbnail.png
authors:
- user: regisss
- user: anahita-b
  guest: true
---

# Accelerating Vision-Language Models: BridgeTower on Habana Gaudi2


*Update (29/08/2023): A benchmark on H100 was added to this blog post. Also, all performance numbers have been updated with newer versions of software.*

[Optimum Habana v1.7](https://github.com/huggingface/optimum-habana/tree/main) on Habana Gaudi2 achieves **x2.5 speedups compared to A100 and x1.4 compared to H100** when fine-tuning BridgeTower, a state-of-the-art vision-language model. This performance improvement relies on hardware-accelerated data loading to make the most of your devices.

*These techniques apply to any other workloads constrained by data loading, which is frequently the case for many types of vision models.* This post will take you through the process and benchmark we used to compare BridgeTower fine-tuning on Habana Gaudi2, Nvidia H100 and Nvidia A100 80GB. It also demonstrates how easy it is to take advantage of these features in transformers-based models.


## BridgeTower

In the recent past, [Vision-Language (VL) models](https://huggingface.co/blog/vision_language_pretraining) have gained tremendous importance and shown dominance in a variety of VL tasks. Most common approaches leverage uni-modal encoders to extract representations from their respective modalities. Then those representations are either fused together, or fed into a cross-modal encoder. To efficiently handle some of the performance limitations and restrictions in VL representation learning, [BridgeTower](https://huggingface.co/papers/2206.08657) introduces multiple _bridge layers_ that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations at different semantic levels in the cross-modal encoder.

Pre-trained with only 4M images (see the detail [below](#benchmark)), BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, BridgeTower achieves an accuracy of 78.73% on the VQAv2 test-std set, outperforming the previous state-of-the-art model (METER) by 1.09% using the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets.


## Hardware

[NVIDIA H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/) is the latest and fastest generation of Nvidia GPUs. It includes a dedicated Transformer Engine that enables to perform fp8 mixed-precision runs. One device has 80GB of memory.

[Nvidia A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/) includes the 3rd generation of the [Tensor Core technology](https://www.nvidia.com/en-us/data-center/tensor-cores/). This is still the fastest GPU that you will find at most cloud providers. We use here the 80GB-memory variant which also offers faster memory bandwidth than the 40GB one.

[Habana Gaudi2](https://habana.ai/products/gaudi2/) is the second-generation AI hardware accelerator designed by Habana Labs. A single server contains 8 accelerator devices called HPUs with 96GB of memory each. Check out [our previous blog post](https://huggingface.co/blog/habana-gaudi-2-bloom#habana-gaudi2) for a more in-depth introduction and a guide showing how to access it through the [Intel Developer Cloud](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html). Unlike many AI accelerators in the market, advanced features are very easy to apply to make the most of Gaudi2 with [Optimum Habana](https://huggingface.co/docs/optimum/habana/index), which enables users to port Transformers-compatible scripts to Gaudi with just a 2-line change.


## Benchmark

To benchmark training, we are going to fine-tune a [BridgeTower Large checkpoint](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc) consisting of 866M parameters. This checkpoint was pretrained on English language using masked language modeling, image-text matching and image-text contrastive loss on [Conceptual Captions](https://huggingface.co/datasets/conceptual_captions), [SBU Captions](https://huggingface.co/datasets/sbu_captions), [MSCOCO Captions](https://huggingface.co/datasets/HuggingFaceM4/COCO) and [Visual Genome](https://huggingface.co/datasets/visual_genome).

We will further fine-tune this checkpoint on the [New Yorker Caption Contest dataset](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) which consists of cartoons from The New Yorker and the most voted captions.

Hyperparameters are the same for all accelerators. We used a batch size of 48 samples for each device. You can check hyperparameters out [here](https://huggingface.co/regisss/bridgetower-newyorker-gaudi2-8x#training-hyperparameters) for Gaudi2 and [there](https://huggingface.co/regisss/bridgetower-newyorker-a100-8x#training-hyperparameters) for A100.

**When dealing with datasets involving images, data loading is frequently a bottleneck** because many costly operations are computed on CPU (image decoding, image augmentations) and then full images are sent to the training devices. Ideally, *we would like to send only raw bytes to devices and then perform decoding and various image transformations on device*. But let's see first how to *easily* allocate more resources to data loading for accelerating your runs.


### Making use of `dataloader_num_workers`

When image loading is done on CPU, a quick way to speed it up would be to allocate more subprocesses for data loading. This is very easy to do with Transformers' `TrainingArguments` (or its Optimum Habana counterpart `GaudiTrainingArguments`): you can use the `dataloader_num_workers=N` argument to set the number of subprocesses (`N`) allocated on CPU for data loading.

The default is 0, which means that data is loaded in the main process. This may not be optimal as the main process has many things to manage. We can set it to 1 to have one fully dedicated subprocess for data loading. When several subprocesses are allocated, each one of them will be responsible for preparing a batch. This means that RAM consumption will increase with the number of workers. One recommendation would be to set it to the number of CPU cores, but those cores may not be fully free so you will have to try it out to find the best configuration.

Let's run the three following experiments:
- a mixed-precision (*bfloat16*/*float32*) run distributed across 8 devices where data loading is performed by the same process as everything else (i.e. `dataloader_num_workers=0`)
- a mixed-precision (*bfloat16*/*float32*) run distributed across 8 devices with 1 dedicated subprocess for data loading (i.e. `dataloader_num_workers=1`)
- same run with `dataloader_num_workers=2`

Here are the throughputs we got on Gaudi2, H100 and A100:

| Device     | `dataloader_num_workers=0` | `dataloader_num_workers=1` | `dataloader_num_workers=2` |
|:----------:|:--------------------------:|:--------------------------:|:--------------------------:|
| Gaudi2 HPU | 601.5 samples/s            | 747.4 samples/s            | 768.7 samples/s            |
| H100 GPU   | 336.5 samples/s            | 580.1 samples/s            | 602.1 samples/s            |
| A100 GPU   | 227.5 samples/s            | 339.7 samples/s            | 345.4 samples/s            |

We first see that **Gaudi2 is x1.28 faster than H100** with `dataloader_num_workers=2`, x1.29 faster with `dataloader_num_workers=1` and x1.79 faster with `dataloader_num_workers=0`. Gaudi2 is also much faster than the previous generation since it is **x2.23 faster than A100** with `dataloader_num_workers=2`, x2.20 faster with `dataloader_num_workers=1` and x2.64 faster with `dataloader_num_workers=0`, which is even better than [the speedups we previously reported](https://huggingface.co/blog/habana-gaudi-2-benchmark)!

Second, we see that **allocating more resources for data loading can lead to easy speedups**: x1.28 on Gaudi2, x1.79 on H100 and x1.52 on A100.

We also ran experiments with several dedicated subprocesses for data loading but performance was not better than with `dataloader_num_workers=2` for all accelerators.
Thus, **using `dataloader_num_workers>0` is usually a good first way of accelerating your runs involving images!**

Tensorboard logs can be visualized [here](https://huggingface.co/regisss/bridgetower-newyorker-gaudi2-8x/tensorboard) for Gaudi2 and [there](https://huggingface.co/regisss/bridgetower-newyorker-a100-8x/tensorboard) for A100.


<!-- ### Optimum Habana's fast DDP

Before delving into how to perform hardware-accelerated data loading, let's look at another very easy way of speeding up your distributed runs on Gaudi. The new release of Optimum Habana, version 1.6.0, introduced a new feature that allows users to choose the distribution strategy to use:
- `distribution_strategy="ddp"` to use PyTorch [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP)
- `distribution_strategy="fast_ddp"` to use a lighter and usually faster implementation

Optimum Habana's fast DDP does not split parameter gradients into buckets as [DDP does](https://pytorch.org/docs/stable/notes/ddp.html#internal-design). It also uses [HPU graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html?highlight=hpu%20graphs) to collect gradients in all processes and then update them (after the [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) operation is performed) with minimal host overhead. You can check this implementation [here](https://github.com/huggingface/optimum-habana/blob/main/optimum/habana/distributed/fast_ddp.py).

Simply using `distribution_strategy="fast_ddp"` (and keeping `dataloader_num_workers=1`) on Gaudi2 gives us 705.9 samples/s. **This is x1.10 faster than with DDP and x2.38 faster than A100!**

So adding just two training arguments (`dataloader_num_workers=1` and `distribution_strategy="fast_ddp"`) led to a x1.33 speedup on Gaudi2 and to a x2.38 speedup compared to A100 with `dataloader_num_workers=1`. -->


### Hardware-accelerated data loading with Optimum Habana

For even larger speedups, we are now going to move as many data loading operations as possible from the CPU to the accelerator devices (i.e. HPUs on Gaudi2 or GPUs on A100/H100). This can be done on Gaudi2 using Habana's [media pipeline](https://docs.habana.ai/en/latest/Media_Pipeline/index.html).

Given a dataset, most dataloaders follow the following recipe:

1. Fetch data (e.g. where your JPEG images are stored on disk)
2. The CPU reads encoded images
3. The CPU decodes images
4. The CPU applies image transformations to augment images
5. Finally, images are sent to devices (although this is usually not done by the dataloader itself)

Instead of doing the whole process on CPU and send ready-to-train data to devices, a more efficient workflow would be to send encoded images to devices first and then perform image decoding and augmentations:

1. Same as before
2. Same as before
3. Encoded images are sent to devices
4. Devices decode images
5. Devices apply image transformations to augment images

That way we can benefit from the computing power of our devices to speed image decoding and transformations up.
Note that there are two caveats to be aware of when doing this:
- Device memory consumption will increase, so you may have to reduce your batch size if there is not enough free memory. This may mitigate the speedup brought by this approach.
- If devices are intensively used (100% or close to it) when doing data loading on CPU, don't expect any speedup when doing it on devices as they already have their hands full.

<!-- To achieve this on Gaudi2, Habana's media pipeline enables us to:
- Initialize a media pipeline with all the operators it needs (see [here](https://docs.habana.ai/en/latest/Media_Pipeline/Operators.html#media-operators) the list of all supported operators) and define a graph so that we can specify in which order operations should be performed (e.g. reading data &rarr; decoding &rarr; cropping).
- Create a Torch dataloader with a HPU-tailored iterator. -->

To implement this on Gaudi2, we have got you covered: the [contrastive image-text example](https://github.com/huggingface/optimum-habana/tree/main/examples/contrastive-image-text) in Optimum Habana now provides a ready-to-use media pipeline that you can use with COCO-like datasets that contain text and images! You will just have to add `--mediapipe_dataloader` to your command to use it.

For interested readers, a lower-level overview is given in the documentation of Gaudi [here](https://docs.habana.ai/en/latest/Media_Pipeline/index.html) and the list of all supported operators is available [there](https://docs.habana.ai/en/latest/Media_Pipeline/Operators.html).

We are now going to re-run the previous experiments adding the `mediapipe_dataloader` argument since it is compatible with `dataloader_num_workers`:

| Device     | `dataloader_num_workers=0` | `dataloader_num_workers=2` |  `dataloader_num_workers=2` + `mediapipe_dataloader` |
|:----------:|:--------------------------:|:--------------------------------------------:|:---------------:|
| Gaudi2 HPU | 601.5 samples/s            | 768.7 samples/s                              | 847.7 samples/s |
| H100 GPU   | 336.5 samples/s            | 602.1 samples/s                              | /               |
| A100 GPU   | 227.5 samples/s            | 345.4 samples/s                              | /               |

We got an additional x1.10 speedup compared to the previous run with `dataloader_num_workers=2` only.
This final run is thus x1.41 faster than our base run on Gaudi2 **simply adding 2 ready-to-use training arguments.** It is also **x1.41 faster than H100** and **x2.45 faster than A100** with `dataloader_num_workers=2`!


### Reproducing this benchmark

To reproduce this benchmark, you first need to get access to Gaudi2 through the [Intel Developer Cloud](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html) (see [this guide](https://huggingface.co/blog/habana-gaudi-2-benchmark#how-to-get-access-to-gaudi2) for more information).

Then, you need to install the latest version of Optimum Habana and run `run_bridgetower.py` which you can find [here](https://github.com/huggingface/optimum-habana/blob/main/examples/contrastive-image-text/run_bridgetower.py). Here is how to do it:

```bash
pip install optimum[habana]
git clone https://github.com/huggingface/optimum-habana.git
cd optimum-habana/examples/contrastive-image-text
pip install -r requirements.txt
```

The base command line to run the script is:
```bash
python ../gaudi_spawn.py --use_mpi --world_size 8 run_bridgetower.py \
--output_dir /tmp/bridgetower-test \
--model_name_or_path BridgeTower/bridgetower-large-itm-mlm-itc \
--dataset_name jmhessel/newyorker_caption_contest --dataset_config_name matching \
--image_column image --caption_column image_description \
--remove_unused_columns=False \
--do_train --do_eval --do_predict \
--per_device_train_batch_size="40" --per_device_eval_batch_size="16" \
--num_train_epochs 5 \
--learning_rate="1e-5" \
--push_to_hub --report_to tensorboard --hub_model_id bridgetower\
--overwrite_output_dir \
--use_habana --use_lazy_mode --use_hpu_graphs_for_inference --gaudi_config_name Habana/clip \
--throughput_warmup_steps 3 \
--logging_steps 10
```
which corresponds to the case `--dataloader_num_workers 0`. You can then add `--dataloader_num_workers N` and `--mediapipe_dataloader` to test other configurations.

To push your model and Tensorboard logs to the Hugging Face Hub, you will have to log in to your account beforehand with:
```bash
huggingface-cli login
```

For A100 and H100, you can use the same `run_bridgetower.py` script with a few small changes:
- Replace `GaudiTrainer` and `GaudiTrainingArguments` with `Trainer` and `TrainingArguments` from Transformers
- Remove references to `GaudiConfig`, `gaudi_config` and `HabanaDataloaderTrainer`
- Import `set_seed` directly from Transformers: `from transformers import set_seed`

The results displayed in this benchmark were obtained with a Nvidia H100 Lambda instance and a Nvidia A100 80GB GCP instance both with 8 devices using [Nvidia's Docker images](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html).

Note that `--mediapipe_dataloader` is compatible with Gaudi2 only and will not work with A100/H100.

Regarding fp8 results on H100 using [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html), they are not available because the code crashes and would require modifying the modeling of BridgeTower in Transformers. We will revisit this comparison when fp8 is supported on Gaudi2.


## Conclusion

When dealing with images, we presented two solutions to speed up your training workflows: allocating more resources to the dataloader, and decoding and augmenting images directly on accelerator devices rather than on CPU.
We showed that it leads to dramatic speedups when training a SOTA vision-language model like BridgeTower: **Habana Gaudi2 with Optimum Habana is about x1.4 faster than Nvidia H100 and x2.5 faster than Nvidia A100 80GB with Transformers!**
And this is super easy to use as you just need to provide a few additional training arguments.

To go further, we are looking forward to using HPU graphs for training models even faster and to presenting how to use DeepSpeed ZeRO-3 on Gaudi2 to accelerate the training of your LLMs. Stay tuned!

If you are interested in accelerating your Machine Learning training and inference workflows using the latest AI hardware accelerators and software libraries, check out our [Expert Acceleration Program](https://huggingface.co/support). To learn more about Habana solutions, [read about our partnership and contact them here](https://huggingface.co/hardware/habana). To learn more about Hugging Face efforts to make AI hardware accelerators easy to use, check out our [Hardware Partner Program](https://huggingface.co/hardware).


### Related Topics

- [Faster Training and Inference: Habana Gaudi-2 vs Nvidia A100 80GB](https://huggingface.co/blog/habana-gaudi-2-benchmark)
- [Fast Inference on Large Language Models: BLOOMZ on Habana Gaudi2 Accelerator](https://huggingface.co/blog/habana-gaudi-2-bloom)
