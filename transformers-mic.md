---
title: "Transformers Model Integration Contracts"
thumbnail: /blog/assets/transformers-mic/mic-logo.png
authors:
- user: tarekziade
---

# Transformers Model Integration Contracts

The [Transformers](https://github.com/huggingface/transformers) framework centralizes model definitions and
supports over 750 different model types.

**88,000+ CPU tests on main, plus 200,000+ GPU tests in a daily run.**
That's the scale of testing behind Transformers. Model tests are only part of it:
we also test pipelines, training, distributed execution, quantization and attention
kernels.

![Transformers testing: 88,000+ CPU tests on main and 200,000+ GPU tests daily. Hardware includes CPUs, NVIDIA GPUs, AMD GPUs, Intel XPUs and Gaudi 3. PRs run selected tests.](/blog/assets/transformers-mic/ci-test-suite.svg)

Each backend runs its supported tests. The integration tests load real
checkpoints and exercise the model's API on real hardware, which makes them
particularly slow and resource-intensive to run.

We can't afford to run all of that on every PR. By the time a scheduled run
finds a regression, it may already be in main.

One way to reduce the gap is to build a smaller model from the real
checkpoint's configuration, preserving its different layer types and
preprocessing, and run "fast integration tests" against it on a cheap CPU runner.

For example, in our [CPU study](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#on-cpu),
the experimental UMT5 fixture used 66 MB of weights versus 1,227 MB for the
source checkpoint.

Using tiny versions of checkpoints also allows us to exercise the architectures
of very large models. Some don't even fit on our standard GPU runners, while
their small fixtures can run on CPU.

Model integrators already build tiny models. We wanted a common recipe, and
checks that go beyond loading the model and running a forward pass.

Tiny models won't replace full integration tests against real checkpoints.
But running them on PRs gives us a chance to catch bugs before they reach main.

## Introducing Model Integration Contracts

[transformers-mic](https://github.com/huggingface/transformers-mic) is a proposed
standard that packages a small model with runnable usage examples and reviewed
expected outputs. A MIC describes an architecture and a usage path. Compatible
checkpoints can share a fixture, while each checkpoint keeps its own expected
results.

We publish these artifacts in [huggingface-mic](https://huggingface.co/huggingface-mic).

Each model published there contains:

- the shrunk model
- a runnable Markdown usage example
- reviewed output baselines

For example, the SmolLM contract generates from "The capital of France is" and
compares its token IDs with a recorded answer. The generation code looks like
this, after loading the model and tokenizer:

```python
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)
new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
print(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
```

The contract also compares training loss, checks that gradients are finite and
verifies that an optimizer step changes the weights. The expected outputs come
from actual runs, and changed values need review.

See the full [usage.md](https://huggingface.co/huggingface-mic/smollm-mic/blob/main/integrations/transformers/usage.md).

The Hub also has a [fleet dataset](https://huggingface.co/datasets/huggingface-mic/transformers-fleet),
which lists the MICs at frozen revisions along with the Transformers commit
used to record their baselines. Each CI run records the fleet revision it uses,
so we can reproduce it later.

On a PR, we select MICs based on the files changed in the patch.

## The UMT5 regression

A UMT5 regression stayed in main for almost two months. Our small CPU fixture
caught it when we replayed those commits.

When Transformers enabled PyTorch's scaled dot-product attention (SDPA) for T5,
UMT5's decoder could attend to future tokens. During training, this lets it
peek at the answer it is supposed to predict. The model still ran, so a test
that only checked for a successful forward pass would miss the problem.

We replayed frozen MICs over five months of Transformers commits. The small
UMT5 fixture flagged the regression from its introduction on [July 30, 2026](https://github.com/huggingface/transformers/commit/560f36cc561fa154d224b7a6dfcba1d57a621d7b)
until its fix on [September 28](https://github.com/huggingface/transformers/commit/d7ba937671edb174fd80d33f9ecbd38f6559dfca).
Label log-probabilities differed from the reviewed baseline by 1.22 and training
loss by about 0.20. The contract used SDPA, the default attention path. Running
only the eager attention variant missed the bug.

We then reverted the fix on a proof branch in Transformers' CI. The check
failed with the same differences, and the selected T5 contract's runner step
took 13.4 seconds on CPU, excluding setup.

The [replay results](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#shadow-backfill)
and the [CI demo](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#transformers-ci-proof)
include the measurements and reproduction details.

## How models are shrunk

The first recipe was simple: make the dimensions smaller and keep fewer
layers. Mixed architectures needed more care. Keeping two layers is no use if
both use sliding attention and the full-attention layers disappear.

The recipe now has four steps:

1. **Shrink the configuration.** Start from a real checkpoint and reduce
   hidden sizes, attention heads and depth. Keep the different layer kinds,
   including mixtures of sliding and full attention or dense and MoE layers.
   Some architectures need extra rules to keep their components compatible.
2. **Keep preprocessing unchanged.** Preserve vocabulary, special tokens and
   input dimensions, and copy tokenizer and processor files byte for byte.
   Loading and preprocessing are part of what we want to test.
3. **Build and validate the weights.** Try seeded random weights and change
   the inputs to check that the model responds. If it doesn't, try training
   it on the contract's small batch of examples. Reload the saved model and
   check that outputs, loss and gradients are finite. Reject failed builds.
4. **Freeze and record.** Save the accepted weights and measure repeatable
   outputs on a known-good Transformers build. Review the baselines for each
   supported platform and precision, then publish a pinned snapshot. CI reuses
   it; it does not rebuild the fixture for every PR.

We also keep the original weights file format. Otherwise, we could skip the
conversion code that runs when a user loads the real checkpoint.

See [the full recipe](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#recipe)
for more detail.

For a visual example, see the [Llama 4 Scout shrink report](https://huggingface-mic-mic-coverage.static.hf.space/shrink/Llama-4-Scout-17B-16E-Instruct-mic.html).
108.6 billion parameters become 44.7 million. The vision stack goes from 34
layers to 2 and the language stack from 48 to 4, keeping the checked layer
kinds and MoE routing style.

Our CLI, `tmic`, creates the contract and builds the small model. For a new
model, an LLM drafts the usage code and manifest. CPU and GPU runs measure the
candidate baselines. Changed expectations require human review.

To start, run this with your source checkpoint in place of `org/model`:

```sh
tmic run org/model
```

The CLI walks through creation, review, measurement and publication. The
[CLI docs](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode)
cover setup and the individual steps.

We used agents to build MICs across model families, fixing the recipe as we
hit edge cases and getting help from model experts. The published models and
remaining gaps are in the
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage).

## Random vs Overfit

With random weights, our tiny Whisper model barely responded to changes in
the audio. It ran, but that made it a poor test of the audio path.

We trained it on a few tones paired with words such as "red" and "green".
In our [weight-initialization study](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#choosing-the-weights),
the random Whisper fixtures caught 18 of 24 injected faults. The overfit
fixtures caught all 24. Teaching it those few answers made audio changes
visible in its outputs; we weren't trying to teach it speech recognition.

That gives us two ways to build the weights:

- Start with random weights and a fixed seed.
- Overfit on a few examples, like "The capital of France is" => "Paris".

Either way, we check input sensitivity, record the outputs and freeze the
weights before CI uses them.

The Whisper result doesn't settle which is better. Those were synthetic
faults, and we haven't found a real regression caught only by the overfit
version. Both versions missed an assisted-decoding bug that the real Whisper
checkpoint exposed.

So we keep random weights when they pass the checks. We overfit when we need
better sensitivity or readable answers, at the cost of training time and
another thing to maintain. The [comparison](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#overfit-versus-random-what-can-we-conclude)
has the results for both.

## What's next

As of November 1, 2026, the [MIC job in Transformers CI](https://transformers-ci.huggingface.cool/d/mic-failures/mic-failures?from=now-4w&to=now&timezone=browser)
reports failures without blocking PRs. Before making it required, we need to
see how often it catches bugs, how often it sends us chasing false alarms, and
how much work baseline reviews take.

We also want to cover more model families and more of the APIs people use.
When a regression slips through, we can add the missing usage path and replay
it on the broken and fixed versions. We'll keep existing tests until their
coverage is accounted for.

We'd also like to link original Hub model pages to their MICs. When a developer
or an agent wants to use a checkpoint, they could start with an example tested
on that checkpoint, its environment and its expected outputs.

Want to help? Pick an uncovered model from the
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage), try
the [tmic workflow](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode),
and contribute a usage path or review a generated contract.
