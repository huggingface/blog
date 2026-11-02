---
title: "Transformers Model Integration Contracts"
thumbnail: /blog/assets/transformers-mic/mic-logo.png
authors:
- user: tarekziade
---

# Transformers Model Integration Contracts

On July 30, 2026, Transformers enabled PyTorch's scaled dot-product attention
(SDPA) for T5. As a side effect, UMT5's decoder could attend to future tokens.
During training, it could peek at the answer it was supposed to predict.
Nothing crashed, and the bug stayed in main for almost two months.

When we later replayed those commits, a small UMT5 model running on CPU flagged
it on every one of them. This post is about that model and the tests built
around it.

## Why bugs reach main

[Transformers](https://github.com/huggingface/transformers) supports over 750
model types, and its test suite is big: **88,000+ CPU tests on main, plus
200,000+ GPU tests in a daily run.** Model tests are only part of it: we also
test pipelines, training, distributed execution, quantization and attention
kernels.

![Transformers testing: 88,000+ CPU tests on main and 200,000+ GPU tests daily. Hardware includes CPUs, NVIDIA GPUs, AMD GPUs, Intel XPUs and Gaudi 3. PRs run selected tests.](/blog/assets/transformers-mic/ci-test-suite.svg)

The integration tests load real checkpoints and exercise the model's API on
real hardware. Running all of that on every PR would mean keeping a lot of GPUs
busy all day long, which isn't practical. By the time a scheduled run finds a
regression, it may already be in main.

Our approach is to shrink the model instead. We build a **fixture**: a small
model made from the real checkpoint's configuration, with fewer and narrower
layers but every kind of layer it has, and its tokenizer and processor files
unchanged. It is published once, then reused. We run "fast integration tests"
against it on a cheap CPU runner. In our
[CPU study](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#on-cpu),
the experimental UMT5 fixture used 66 MB of weights; the source checkpoint
is 1,227 MB.

This also covers very large models: some don't fit on our standard GPU
runners, but their fixtures run on CPU.

Model integrators already build tiny test models (Transformers' `tiny-random-*`
checkpoints). We wanted a common recipe, and
checks that go beyond loading the model and running a forward pass. The UMT5
bug would have passed a forward pass just fine.

Fixtures won't replace full integration tests, but on PRs they give us a
chance to catch bugs before they reach main.

## Introducing Model Integration Contracts

[transformers-mic](https://github.com/huggingface/transformers-mic) is a proposed
standard for packaging these tests. A Model Integration Contract (MIC) has
three parts:

- a **fixture**: the shrunk model described above, published once and pinned
- a **usage path**: a runnable Markdown example of something people do with the
  model, such as generating text or running a training step
- **baselines**: the outputs recorded when running that example, reviewed by a
  human

The contract holds if running the usage path on the fixture still produces the
baselines.

A contract covers an architecture and a usage path, not a single checkpoint:
compatible checkpoints can share a fixture, while each keeps its own expected
results. We publish them in [huggingface-mic](https://huggingface.co/huggingface-mic).

For example, the SmolLM contract generates from prompts such as "The capital
of France is" and compares its token IDs with a recorded answer. After loading
the model and tokenizer, the generation code is, simplified to one prompt:

```python
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)
new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
print(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
```

It also compares training loss, checks that gradients are finite and verifies
that an optimizer step changes the weights. See the full
[usage.md](https://huggingface.co/huggingface-mic/smollm-mic/blob/main/integrations/transformers/usage.md).

A [fleet dataset](https://huggingface.co/datasets/huggingface-mic/transformers-fleet)
on the Hub pins each MIC to a frozen revision, along with the package versions
used to record its baselines. Each CI run records the fleet revision it
used, so we can reproduce it later.

On a PR, we pick MICs based on the files the patch changes.

## The UMT5 regression

We replayed frozen MICs over five months of Transformers commits. The small
UMT5 fixture flagged the regression from its introduction on [July 30, 2026](https://github.com/huggingface/transformers/commit/560f36cc561fa154d224b7a6dfcba1d57a621d7b)
until its fix on [September 28](https://github.com/huggingface/transformers/commit/d7ba937671edb174fd80d33f9ecbd38f6559dfca).
Label log-probabilities differed from the reviewed baseline by 1.22 and training
loss by about 0.20.

One detail mattered: the contract used SDPA, the default attention path.
Running only the eager attention variant missed the bug. A fixture only
helps if it goes through the same code as the real checkpoint.

We then reverted the fix on a proof branch in Transformers' CI. The check
failed with the same differences, and the selected T5 contract's runner step
took 13.4 seconds on CPU, excluding setup.

The [replay results](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#shadow-backfill)
and the [CI demo](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#transformers-ci-proof)
have the measurements and reproduction details.

## How models are shrunk

The first recipe was simple: make the dimensions smaller and keep fewer
layers. That wasn't enough for mixed architectures. Keeping two layers is no
use if both use sliding attention and the full-attention layers disappear.

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

We also keep the original weights file format. Otherwise we would skip the
conversion code that runs when a user loads the real checkpoint.
[The full recipe](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#recipe)
has more detail.

The [Llama 4 Scout shrink report](https://huggingface-mic-mic-coverage.static.hf.space/shrink/Llama-4-Scout-17B-16E-Instruct-mic.html)
shows what this looks like on a big model. 108.6 billion parameters become 44.7
million. The vision stack goes from 34 layers to 2 and the language stack from
48 to 4, keeping the checked layer kinds and MoE routing style.

### Building MICs with agents

Our CLI, `tmic`, creates the contract and builds the fixture. For a new
model, an LLM drafts the usage code and manifest, and CPU and GPU runs measure
the candidate baselines. The LLM drafts; it doesn't approve. Changed
expectations require human review.

To try it on your machine, use your source checkpoint in place of `org/model`:

```sh
pip install "transformers-mic[runtime]" "hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git"
hf auth login
tmic run org/model --local-only --fixtures-only --no-history --bill-to ''
```

It drafts the contract, builds the fixture and records its baselines on your
machine, asking you to review each step, and stops before uploading anything.
For a new model, the draft comes from an LLM through Inference Providers, billed
to your own account (`--bill-to ''`). The
[CLI docs](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode)
cover the individual steps.

We used agents to build MICs across model families. When they hit an edge
case, we fixed the recipe itself rather than the one fixture, with help from
model experts. The published models and remaining gaps are in the
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage).

## Random vs Overfit

With random weights, our Whisper fixture barely responded to changes in
the audio. It ran, but that made it a poor test of the audio path.

So we trained it on a few tones paired with words such as "red" and "green".
We weren't teaching it speech recognition, just making audio changes visible
in its outputs. In our [weight-initialization study](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#choosing-the-weights),
the random Whisper fixtures caught 18 of 24 injected faults. The overfit
fixtures caught all 24.

That gives us two ways to build the weights:

- Start with random weights and a fixed seed.
- Overfit on a few examples, like "The capital of France is" => "Paris".

The Whisper result doesn't settle which is better. Those were synthetic
faults, and we haven't found a real regression caught only by the overfit
version. Both versions missed an assisted-decoding bug that the real Whisper
checkpoint exposed.

So we keep random weights when they pass the checks, and overfit when we need
better sensitivity or readable answers. Training takes seconds, but it is one
more step that can fail to converge, and one more thing to maintain. The [comparison](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#overfit-versus-random-what-can-we-conclude)
has the results for both.

## Open questions

As of November 1, 2026, the [MIC job in Transformers CI](https://transformers-ci.huggingface.cool/d/mic-failures/mic-failures?from=now-4w&to=now&timezone=browser)
reports failures without blocking PRs. Whether it should block depends on
things we haven't measured yet:

- How many real bugs does it catch on PRs, not just in replays?
- How often does it send us chasing false alarms?
- How much work do baseline reviews take?

We also don't know how many bugs, like the Whisper assisted-decoding one, only
show up with real weights, or whether overfitting catches real bugs that random weights miss.

Coverage is still partial, across model families and the APIs people use.
When a regression slips through, we can add the missing usage path and replay
it on the broken and fixed versions. We'll keep existing tests until their
coverage is accounted for.

Finally, we'd like to link original Hub model pages to their MICs, so a
developer or an agent using a checkpoint could start from an example tested on
that checkpoint, its environment and its expected outputs.

Want to help? Pick an uncovered model from the
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage), try
the [tmic workflow](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode),
and contribute a usage path or review a generated contract.
