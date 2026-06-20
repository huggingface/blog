---
title: "Beyond LoRA: Can you beat the most popular fine-tuning technique?"
thumbnail: /blog/assets/peft-beyond-lora/thumbnail.png
authors:
- user: BenjaminB
- user: sayakpaul
- user: hubnemo
- user: kashif
---

# Beyond LoRA: Can you beat the most popular fine-tuning technique?

<p align="center">
    <img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/lora-celebration.png" alt="Is LoRA the best PEFT technique?" width="600"/>
</p>

# When you plan to fine-tune a model in a parameter-efficient way, think beyond LoRA

If you want to fine-tune an open model on your own data, you are probably interested in so-called parameter-efficient fine-tuning, in short *PEFT*. This term describes techniques that significantly reduce the memory requirement to fine-tune a model. Although there are dozens of these techniques, almost everyone chooses one called “LoRA”. In this blog post, we explore whether LoRA is really the best choice, what tools are available to make an informed decision, and how you can benefit from extending your horizon beyond LoRA.

# What is PEFT and when do you need it

There are countless open models available, but they often aren't quite good enough for your use case. Prompting may help, but it usually isn't enough. Rather than training a new model from scratch, you should consider fine-tuning an existing one.

Fine-tuning, however, is memory-hungry: you generally need enough memory to fit the whole model several times over. Quantization reduces a model's memory footprint, but quantized models can't be fine-tuned directly. So a set of techniques emerged to cut the memory needed for fine-tuning, called "parameter-efficient fine-tuning", or PEFT.

With PEFT, you can fine-tune a model using only a fraction of that memory and even fine-tune quantized models. It offers other advantages, such as tiny checkpoint sizes, greater resistance to catastrophic forgetting, and the ability to serve multiple fine-tunes from the same base model.

At Hugging Face, we develop the [`PEFT` library](https://github.com/huggingface/peft), which implements many PEFT techniques behind a unified API and integrates well with the ecosystem, for example [`Transformers`](https://huggingface.co/docs/transformers/main/en/peft) and [`Diffusers`](https://huggingface.co/docs/diffusers/main/en/api/loaders/peft). It also supports [multiple quantization methods](https://huggingface.co/docs/peft/developer_guides/quantization), enabling further accessibility in parameter-efficient fine-tuning. `PEFT` provides a good starting point, whether you want to fine-tune on your own data or you're researching a new PEFT method.

# LoRA: The queen of fine-tuning techniques 👑

One parameter-efficient fine-tuning technique that emerged early and proved to be quite effective is called “Low Rank Adaptation”, or short [“LoRA”](https://huggingface.co/papers/2106.09685). It works by adding a handful of parameters on top of the base model, freezing the base model weights, and only training those few parameters.

Among all PEFT techniques, LoRA is by far the most popular. Here are a few estimates:

* Of a sample of 20,834 [model cards on Hugging Face Hub](https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata) that mention exactly one PEFT technique, 20,509 mention LoRA (98.4%).
* We checked which PEFT techniques are popular for image generation on an external site, too. Using a sample of 10,000 checkpoints, we found 7,111 to be LoRAs. The other identified PEFT techniques are LoCon (363) and DoRA (11, arguably a LoRA variant). That means 95.0% of PEFT checkpoints are LoRAs.
* Searching for the code snippet `from peft import <PEFT CONFIG>` on GitHub ([example GH query](https://github.com/search?q=%22from+peft+import+LoraConfig%22&type=code)), 71.3% of results are for LoRA. The runners-up are LoHa (3.7%) and AdaLoRA (3.5%).

Although these estimates are not perfect, the conclusion is nonetheless that LoRA is almost certainly by far the most common PEFT technique.

This could just mean that LoRA works best for everyone, and this fact is reflected in its usage statistics. There is, however, another possibility: LoRA was one of the earlier, popular PEFT techniques. So maybe its usage became self-reinforcing: LoRA has the highest visibility, the highest number of tutorials/examples, and it has the best support in downstream packages. Thus LoRA's popularity feeds on itself.

This all leads to the question: *Are we all leaving performance on the table by shunning better techniques?* After all, there are countless researchers whose papers claim their technique beats LoRA. Isn't that sufficient proof that we should go beyond LoRA in favor of newer techniques?

# Choosing the right PEFT technique based on paper results is problematic

There are dozens of papers that investigate fine-tuning techniques other than LoRA. Just in the `PEFT` library, there are more than 40 distinct PEFT techniques at the time of writing (and numerous more when counting variations of PEFT techniques). For almost all of them, you will find researchers claiming that their technique beats LoRA according to their benchmarks.

The trouble with these claims is that researchers are under pressure to provide results that beat the existing benchmark. Even without ill intent, this can bias the results, e.g. by spending less time tuning the alternative techniques compared to the one proposed by the researchers. [One study](https://arxiv.org/abs/2602.04998) found, for instance, that LoRA can match supposedly better PEFT techniques by tuning the learning rate.

Another complication is that each paper chooses a different set of PEFT techniques to compare to, and a different set of benchmarks to run. And even if the same technique is compared on the same benchmark, the code is often not available or not easy to run yourself, which makes results hard to reproduce.

Overall, it's difficult to figure out the PEFT technique that works best for you by only checking paper results. Therefore, you might be tempted to just go with the default, LoRA.

# How we approach benchmarking in `PEFT`

At Hugging Face, we thought about how we can help users make informed decisions about which PEFT technique to use. With the `PEFT` library, we already provide a package that implements many PEFT techniques and exposes them with the same API. The next step is to provide benchmarks that can shed more light on the discussed issue.

We already had a benchmark that [checks fine-tuning of LLMs on a math dataset](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) for some time. This benchmark takes an LLM and fine-tunes it on chain-of-thought reasoning to produce the result to a mathematical question using a base model that is not instruction fine-tuned. The benchmark thus checks if the model can learn to perform mathematical reasoning and also to adjust the generated output to the expected format.

To extend our findings on another modality, we also added an [image generation benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/image-gen). This one tests whether the model can be fine-tuned to learn a new concept, a [cat plushy](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset), and generate it in new contexts without forgetting existing concepts.

<table align="center">
  <tr>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/metamath-question.png" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/cat-plushy-train-image.jpg" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><em>Left: Sample question and answer from the MetaMathQA dataset. Right: Sample image from the cat plushy dataset.</em></td>
  </tr>
</table>

All PEFT techniques are evaluated according to the exact same conditions: same base model, same dataset, same training and evaluation code, same hardware. As different users have different needs, we track more than just test performance. Besides VRAM usage, we track metrics like forgetting/drift, runtime, and checkpoint size. The results are designed to run on consumer hardware, and adding a new experiment only requires adding a new `PEFT` config and running a script.

Since we compare all PEFT techniques on equal footing and have no horse in the race, we believe that these benchmarks can draw an objective picture of how well different PEFT techniques work. We argue that if you have your own dataset, you can take a similar approach and take advantage of the `PEFT` library to evaluate multiple PEFT techniques.

# Our findings: LoRA works well but is not necessarily the best choice

After finishing the benchmark runs, we found that although LoRA works well, other PEFT methods can beat it on one or multiple axes and should thus be considered. Check the image below that compares the performance of LoRA and five other PEFT techniques.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/benchmark-highlights.png" width="900"/></td>
  </tr>
  <tr>
    <td align="center"><em>Some results from the benchmark. When it comes to test performance and memory usage, LoRA is not necessarily the best choice. Left: MetaMathQA benchmark; right: image generation benchmark. Consult this <a href="https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison">Space</a> for the most up-to-date results.</em></td>
  </tr>
</table>

One way to interpret the results above is to think in terms of tradeoffs, for example: How well does the model perform on the test set vs how much memory is needed to train it? If a PEFT technique cannot be beaten on both of these metrics at the same time by any other technique, it is on the *Pareto Frontier*. In other words: If you want better test accuracy, you need more memory, and if you want more memory efficiency, you have to give up on accuracy.

Let's take a closer look at the results for the LLM Math dataset benchmark. When it comes to test accuracy vs memory, we find that LoRA is indeed on the Pareto frontier. It achieves 53.2% test accuracy and requires 22.6 GB of VRAM at the peak. There are, however, other PEFT techniques on the Pareto Frontier. For instance, [BEFT](https://huggingface.co/docs/peft/main/en/package_reference/beft) achieves 32.9% test accuracy and requires only 20.2 GB of memory at max. On the other end, we have [Lily](https://huggingface.co/docs/peft/main/en/package_reference/lily), which achieves 54.9% test accuracy but requires 25.6 GB of memory. Depending on what's more important to you, you may conclude that LoRA does not present the best tradeoff for you.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/metamath-pareto.png" width="900"/></td>
  </tr>
  <tr>
    <td align="center"><em>Test accuracy vs memory usage tradeoff of fine-tuning <code>meta-llama/Llama-3.2-3B</code> and evaluating it on GSM8K. LoRA does well but so do other PEFT techniques.</em></td>
  </tr>
</table>

It is also worth noting that even though LoRA does well on this task, we're not talking about vanilla LoRA. On one side, we have LoRA with [rank stabilized initialization](https://huggingface.co/papers/2312.03732), which is a technique to scale the LoRA contribution differently from the default initialization and provides very good test accuracy (53.2%). On the other end, we have [LoRA-FA](https://huggingface.co/papers/2308.03303), which uses an optimizer specialized for LoRA that freezes part of the LoRA weights and is thus more memory efficient (20.2 GB). Normal LoRA only achieves an accuracy of 48.1% at 22.5 GB memory and should thus be avoided in favor of the alternatives.

Next let's take a look at the image generation benchmark. In the [Hugging Face Space](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison), choose “image-gen” in the “Select Task” dropdown to show the results. The goal of the task is to learn a new concept, namely a cat plushy, and generalize it to new prompts.

<div align="center">
<table>
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/cat-plushy-lora.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>Cat plushy image created with LoRA fine-tuned on <code>FLUX.2-klein-base-4B</code>.</em></td>
  </tr>
</table>
</div>

For this task, the main metric is “dino similarity”, which measures how much a generated image resembles the picture from a holdout test dataset, with higher values being better. As always, we also want to keep an eye on memory usage. When plotting the Pareto Frontier of these two metrics, we find that LoRA is below that frontier. Let's get concrete numbers: LoRA achieves a similarity score of 0.697 whereas [OFT](https://huggingface.co/docs/peft/package_reference/oft) achieves 0.708; in terms of memory, LoRA requires 9.97 GB, and OFT requires 9.01 GB. Therefore, OFT strictly dominates LoRA on these metrics.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/image-gen-pareto.png" width="900"/></td>
  </tr>
  <tr>
    <td align="center"><em>Test accuracy vs memory usage tradeoff of fine-tuning <code>FLUX.2-klein-base-4B</code> and evaluating it on the test set. Other PEFT techniques like OFT beat LoRA in terms of test score and lower memory usage.</em></td>
  </tr>
</table>

Of course, you should also check the other PEFT methods that are close to the Pareto frontier, as metrics can be subject to small variations due to randomness. Also, you should explore other metrics: is runtime performance important to you or do you care about the size of the checkpoints? Choose the relevant metric from the dropdown and the picture can change considerably. For the image generation benchmark, do inspect the generated sample images to get a vibe of the fine-tuned model's capability.

# Limitations

> [!WARNING]
> Objection: But the benchmarks favor one method over another!

One criticism that could be leveled at the `PEFT` benchmarks is that the choice of hyper-parameters may favor one technique over another. This is true, doing an exhaustive and fair hyper-parameter sweep with this many techniques is difficult. It is, however, very easy for everyone to contribute their own experiments to `PEFT`: If you believe that a specific PEFT technique can be improved by choosing different hyper-parameters, create a PR! We added [instructions on how to do that](https://github.com/huggingface/peft/tree/main/method_comparison#creating-new-experiments). In a similar vein, if you want to contribute a completely new benchmark, reach out to us to discuss your idea.

Another problem with the benchmarks is that they may not fully reflect the capabilities of a specific PEFT technique. We make it possible to compare the techniques along many different dimensions and discover the best ones according to these tradeoffs. But it's impossible to capture all facets this way. For instance, one PEFT technique called [Cartridges](https://huggingface.co/docs/peft/package_reference/cartridges) was developed to compress long prompts, which is not measured in the benchmarks. Other factors can also influence the choice, for instance:

* Depending on the PEFT technique, only certain layer types can be modified.
* Not all PEFT techniques support quantized base models (but we actively expand the support in `PEFT`).
* Some PEFT techniques allow [merging of the adapter](https://huggingface.co/docs/peft/main/en/developer_guides/model_merging) to reduce runtime overhead but others don't.

The benchmarks cannot fully lift the responsibility to do your research, but they can be reasonable pointers.

<table align="center">
  <tr>
    <td align="center"><a href="https://huggingface.co/spaces/peft-internal-testing/PEFT-shop"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/peft-shop.png" width="100%"/></a></td>
  </tr>
  <tr>
    <td align="center"><em>Click on the image to peruse the PEFT shop to find the best PEFT technique for you. It allows you to browse not only by benchmark metrics but also by capabilities, like quantization support.</em></td>
  </tr>
</table>

> Objection: But llama.cpp/vLLM/... only supports LoRA

A limitation of using a PEFT technique other than LoRA is that they don't get the broad support in downstream packages that LoRA sees. For example, if you want to serve the model using vLLM, only LoRA checkpoints can be loaded. Thankfully, `PEFT` now supports [converting other adapters into LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora_conversion). That way, you can convert a non-LoRA checkpoint into LoRA and use it in vLLM or other downstream packages.

To test this, we converted an image adapter using the GraLoRA technique into a LoRA checkpoint. The test scores were virtually identical after conversion (similarity 0.702 → 0.694, 0.260 → 0.269). Below are test images for the prompt “sks cat at the beach”:

<table align="center">
  <tr>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/lora-image-gen.png" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/gralora-image-gen-converted.png" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><em>Left: Image generated by GraLoRA. Right: Image generated by the same GraLoRA checkpoint converted to a LoRA checkpoint. The images quality is comparable.</em></td>
  </tr>
</table>

At the moment, we haven't implemented conversion for all PEFT techniques, but if there is demand, we will expand the support.

# Conclusion and what *you* can do

While working on the `PEFT` package, we noticed that LoRA has a lot of momentum behind it, even though other PEFT techniques are potentially better. Therefore, we set out to add benchmarks to PEFT that could paint a more objective picture of how well different PEFT techniques perform on different metrics.

Given the results we found, we can confidently conclude that LoRA is not a bad choice at all, but there are potentially better choices. Especially when checking the image generation benchmark, LoRA is beaten by other techniques. We discussed that besides metrics, other considerations must be taken into account when choosing the right PEFT technique. However, even then, we are pushing `PEFT` further to achieve feature parity between LoRA and those other techniques.

Our journey is far from finished; we want to extend and improve the existing benchmarks, and we also plan to add more benchmarks in the future. We ensured that it is easy for the community to contribute, so if this is something you would like to do, please open an [issue on the `PEFT` repository](https://github.com/huggingface/peft/issues) and let us know how you would like to contribute.

If you take away only one thing from this article, it is that LoRA should not be the automatic default when choosing a PEFT technique for your use case. Given the unified API provided by `PEFT`, changing from one PEFT technique to another is as easy as switching one config in your code. And even if you stick with LoRA, check out all the variants that are supported in `PEFT`: DoRA, rs-LoRA, LoRA-FA etc. Give these other techniques a try and you might be pleasantly surprised.

Example: Changing from LoRA to OFT using `PEFT`:

```diff
from transformers import AutoModelForCausalLM
-from peft import LoraConfig, get_peft_model
+from peft import OFTConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype="bfloat16")
-config = LoraConfig(target_modules=["q_proj", "v_proj"])
+config = OFTConfig(target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
```
