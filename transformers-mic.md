---
title: "Transformers Model Integration Contracts"
thumbnail: /blog/assets/transformers-mic/thumbnail.png
authors:
- user: tarekziade
---

# Transformers Model Integration Contracts

The [Transformers](https://github.com/huggingface/transformers) framework centralizes model definitions and
supports over 750 different model types.

Transformers has extensive CPU and GPU test suites. Our integration tests check
that supported models still work as expected by exercising the model's API
against real checkpoints on real hardware.

But running full-checkpoint integration tests on every PR would take too long
and require a lot of resources, so we run them selectively.

Nightly runs give us broader coverage once a day, leaving a window in which
regressions can reach main.

One way to reduce the gap is to build a smaller model from the real
checkpoint's configuration, preserving its different layer types and
preprocessing, and run "fast integration tests" against it on a cheap CPU runner.

For example, in our [CPU study](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#on-cpu),
the experimental UMT5 fixture used 66 MB of weights versus 1,227 MB for the
source checkpoint.

Using tiny versions of checkpoints also allows us to exercise the architectures
of very large models. Some don't even fit on our standard GPU runners, while
their small fixtures can run on CPU. How much we save depends on the model and
what the test does.

Building tiny models is something model integrators already do. MIC gives us
a common way to build them and test their outputs as part of model integration.
A model that loads and runs can still return the wrong answer.

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

That gives us the ability to run a selection of MICs on PRs, depending on which
files are changed in the patch.

## A concrete example: catching a UMT5 regression

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

This was a historical replay, not a bug we stopped before it merged. You can
read the [replay results](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#shadow-backfill)
and the [CI demo](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#transformers-ci-proof).

## How models are shrunk

We started with a simple recipe and added rules as we encountered different
architectures. There are four main steps:

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

The process also keeps the original weights file format. Transformers supports
several formats, and we want to exercise the same conversion steps when the
model is loaded.

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

We used agents to build MICs across a growing set of model families. The recipe
evolved as we hit edge cases and incorporated feedback from model experts.
You can check the published models and remaining gaps in the
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage).

## Random vs Overfit

One interesting aspect of building assertions for MICs is how we initialize
the model weights.

There are two strategies:

- Random weights: initialize with a fixed seed, check input sensitivity, then record outputs.
- Overfit weights: train on a few examples, check input sensitivity, then record outputs.

In both cases, we freeze the weights before CI uses them.

Overfitting gives us readable examples, like "The capital of France is" =>
"Paris". It can also make the fixture more sensitive to changes in its inputs.
For example, the tiny Whisper model can largely ignore audio with random
weights. In our [weight-initialization study](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#choosing-the-weights),
random Whisper fixtures caught 18 of 24 injected faults, while overfit fixtures
caught all 24. We trained it on tones paired with words such as "red" and
"green". These are test cases we taught it, not speech transcription.

Those results are for synthetic faults. We haven't found a real regression
caught only by the overfit version, and both versions missed a Whisper
assisted-decoding bug that the real checkpoint exposed.

We keep random weights when they pass the checks, and use overfitting when we
need better input sensitivity or readable answers. Overfitting also adds
training time and another thing to maintain. The
[comparison](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#overfit-versus-random-what-can-we-conclude)
goes into the tradeoffs.

## What's next

As of November 1, 2026, the [MIC job in Transformers CI](https://github.com/huggingface/transformers-ci/blob/main/docs/model-integration-contracts.md#observational-behavior-and-telemetry)
reports failures without blocking PRs. We want to measure the bugs it catches,
false alarms, runtime and the work needed to review baseline changes before
making checks required.

We also want to cover more model families and more of the APIs people use.
When a regression slips through, we can add the missing usage path and replay
it on the broken and fixed versions. Full-checkpoint tests stay: we should
only remove an existing test once its coverage is preserved.

Another idea is to link original Hub model pages to their MICs. Developers and
agents could find a runnable example, the environment where it passed and
outputs to compare against. For that to be useful, the original checkpoint
needs its own passing result; a passing tiny model isn't enough.

Want to help? Pick an uncovered model from the
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage), try
the [tmic workflow](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode),
and contribute a usage path or review a generated contract.
