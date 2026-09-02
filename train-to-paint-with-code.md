---
title: "Training a coding model to paint watercolours with TRL and OpenEnv"
thumbnail: /blog/assets/train-to-paint-with-code/thumbnail.png
authors:
- user: sergiopaniego
---

# Training a coding model to paint watercolours with TRL and OpenEnv

![Training a coding model to paint watercolours](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/thumbnail.png)

On 23 August, [Surya
Narreddi](https://x.com/kickingkeys/status/2091570990048276897) posted a beautiful video
of watercolours painted by a language model. The model writes JavaScript through
[p5.brush](https://github.com/acamposuribe/p5.brush), a library that "adds natural
drawing tools to p5.js". The video went viral fast, over 1.5M views at the time of
writing.

The video came with [a blog
post](https://surya.website/rling-qwen-to-paint-with-code) explaining the training
behind an earlier and narrower stage of the project, close-up flowers rather than the
full compositions in the video, sadly without open artifacts yet. His site says a full
technical report is coming, so ensure you follow him. The original idea is his, coming from the art and design side, where [his skills are way beyond mine](https://x.com/kickingkeys/status/2094901433149612118). My attempt is on the engineering side, reproducing the recipe in the open with every piece published.

> **Note:** for the context behind the project, told by Surya himself, watch [this
> video of his thesis](https://vimeo.com/1190839818).

In this article I try and reproduce his idea with [TRL](https://huggingface.co/docs/trl) and
[OpenEnv](https://github.com/huggingface/OpenEnv). The reference pool dataset, the RL
environment, the training scripts and the trained models, all open.

The whole pipeline runs on Hugging Face, end to end:
* training on [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
* the RL environment and the scorer model as [Spaces](https://huggingface.co/docs/hub/spaces)
* the pairwise judge through [Inference Providers](https://huggingface.co/docs/inference-providers)
* and every artifact on the Hub

Once the two Spaces are up, the recipe is one command. Duplicate the
[environment](https://huggingface.co/spaces/HuggingEnvs/watercolour-env) and the
[scorer model](https://huggingface.co/spaces/HuggingEnvs/watercolour-hpsv3), set two
environment variables for the reward mix, and launch:

```bash
hf jobs uv run train/watercolour_grpo.py --flavor h200 --timeout 48h --secrets HF_TOKEN -- \
  --env-url https://<you>-watercolour-env.hf.space \
  --model Qwen/Qwen3.5-35B-A3B --lora --all-linear --bf16 --gradient-checkpointing \
  --subject 'a peach hibiscus' --references 4 \
  --top-p 0.95 --top-k 20 \
  --lr 5e-5 --lr-scheduler constant_with_warmup --warmup-steps 5 \
  --scale-rewards none \
  --steps 200 --n-episodes 240 --num-generations 8 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --max-completion-length 8192 \
  --run-tag my-run --out <you>/watercolour-grpo --push-to-hub
```

The rest of this article is the story of getting there, and [every piece is in the
repo](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour).

I have followed the original blog step by step, and only changed something when strictly needed. Every idea of my own went into a list instead of into the experiment, and that list became "What I would try next" at the end, next to the full list of published artifacts. If you have already read his post, the framing and
the reward design will be familiar. The new material is the open implementation, the
hand-rated pool, and three reward mixes trained and compared, and it starts at [The RL
environment you need to build](#the-rl-environment-you-need-to-build).

<figure class="image text-center">
  <video controls autoplay muted loop playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-runs-evolution.mp4" title="Three runs evolving in parallel"></video>
  <figcaption>Three runs, one per reward mix, evolving in parallel. The median painting per step as training advances. Which run is which is what this article explains.</figcaption>
</figure>

## Why people loved it

The paintings look loose, imperfect, handmade, at a moment when image models produce
perfect pictures. My guess is that this contrast is a big part of why the video went
viral. It reminded me of the early days of generative AI art, when the point was
to explore the medium:
[DeepDream](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/)
(2015) was a debugging tool that people turned into art, works like [Edmond de
Belamy](https://en.wikipedia.org/wiki/Edmond_de_Belamy) (2018) came from artists probing
what a GAN could do, and artists like [Mario
Klingemann](https://quasimondo.com) spent those years making [dreamy portraits with neural
networks](https://artsandculture.google.com/asset/memories-of-passerby-i-mario-klingemann/aAHG7iV3aXme8g).

This project feels closer to those early days. In his thesis, Surya describes the path
that led here. He started with prompting, where more detail in the prompt buys more
control over the image, and realised that training the model itself goes further. The
other half of the idea is the medium. The model writes a program of about 150
lines of JavaScript that paints the image. That model output is code. You can read it, edit it and run it
again, and the decision behind each brushstroke is visible. And the style comes from a
restriction where the model is only allowed *ten of the library's methods*. More on that below.

In that same period, [Anna Ridler](https://annaridler.com/works/myriad-tulips)
photographed thousands of tulips, hand-labelled every one, exhibited the dataset itself
as the artwork, and later trained a model on it. I found her work through the references
AI agents brought back while building this project and loved it because this
project does something very similar by curating a set of images by hand, and then training against them.

## RL over taste

Most of the recent RL work on language models uses rewards you can verify. For example, math problems with a known answer, code that passes tests, or graders that are right or wrong and cheap to run. This project is closer to the older exception, RLHF, where the model learns a reward model from human preferences.

Here the reward is aesthetic preference. There is no *correct* answer. The real question
of the project is whether you can do RL over taste.

The reward, as his blog defines it and as the RL environment I built implements it:

| term | weight | what it measures |
| --- | --- | --- |
| `gate` | 0.05 | the sketch compiles, paints something, does not cheat |
| `length` | 0.05 | a soft push towards longer code snippets |
| pairwise judge | 0.60 | style, compared against references drawn from a pool |
| [HPSv3](https://huggingface.co/MizzenAI/HPSv3) | 0.30 | aesthetic preference on the render |

[HPSv3](https://huggingface.co/MizzenAI/HPSv3) is an open 7B preference model. Give it an image and a text description, and it returns
a score for how much a person would prefer that image. It was trained on a large set of
human choices between pairs of images, so its score is an average of many people's taste. The pairwise judge is
[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), a
general vision model called through HF Inference Providers. The pairwise judge sees the candidate
painting
next to four references randomly selected from the pool, guided by a written description of what to weigh (bleeds, translucent washes, soft edges),
each comparison in both presentation orders, and its score is the share of comparisons
the candidate wins. Its only standard is the pool, so its score is my taste, as encoded
in those ratings.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/reward-diagram.png" alt="The reward pipeline, piece by piece">
  <figcaption>Two of the four terms in the reward function are models. Both are proxies for someone's taste.</figcaption>
</figure>

Those are the weights Narreddi converged on after a first rubric with nine signals
plateaued. The pool defines taste here. That moves the work from tuning hyperparameters
to building the set that decides what is beautiful.

I trained three runs with this reward. They differ only in how the weight splits between
the two model judges:

| run | pairwise judge | HPSv3 | role |
| --- | --- | --- | --- |
| `judge-led` | 0.60 | 0.30 | the original mix, stopped at step 110 |
| `hps-led` | 0.30 | 0.60 | the middle point, stopped at step 110 |
| `hps-only` | 0.00 | 0.90 | the validation run, stopped at step 60 |

I started with `hps-only` to validate that the pipeline could learn at all. Once the reward was going up and the metrics were healthy, there was no reason to run it longer, so I launched the two longer runs instead. The question that the longer runs ask is how much of HPSv3's power can you hand to the pairwise judge? The more weight the judge carries, the more
the reward means *my* taste instead of everyone's, and the harder it should be to climb. Incidentally, if you push it far enough or your style is too far from the average, the model could stop entirely.

Fortunately, it did not stop and both runs with the pairwise judge on learned too. The hand-rated pool
can steer the policy, at least as far as the metrics and the final paintings show. The
numbers are below.

> **Disclaimer.** If we use a frontier model, it can already generate the JavaScript code that paints watercolour from a prompt. That's the starting point. The work here is about teaching a smaller model to do it combined with a person's own artistic preferences.

## The RL environment you need to build

The environment wraps everything that sits between the model and the reward, including the JavaScript library that the model uses for painting, the system prompt that restricts it, the headless Chromium that renders each sketch, and the gate that rejects cheats.

The library does more of the work than it seems.
[p5.brush](https://github.com/acamposuribe/p5.brush), by
[@acamposuribe](https://x.com/acamposuribe), simulates a medium rather than drawing
shapes: pigment bleeds past the edges of a fill, paper has texture, strokes have mass,
flow fields drag brushwork around. When the model calls `brush.fillBleed(0.25)` it is deciding how far the ink
runs.

> **Note.** The author of p5.brush had been trying to teach a machine to paint long
> before any of this. In 2022 he made a generative art series that hides a diary about
> teaching p5.js to draw like a child: *"It is barely able to use the crayons [...] It
> cannot follow simple commands. I'm done for today, very infuriating."* The series was
> meant to have three pieces, and he made two. When Surya's video went viral, [he
> quoted it](https://x.com/acamposuribe/status/2091668313449316651), sharing that diary
> and saying this work is the third piece arriving on its own.

p5.brush exposes 47 methods. The prompt allows 10: `scaleBrushes`, `noStroke`, `fill`, `noFill`, `fillBleed`, `fillTexture`, `beginShape`, `vertex`, `endShape` and `circle`. What the other thirty-seven add, lines, hatching, custom brushes, would break the watercolour look. With these ten the model can only paint filled shapes, and the library adds the bleed to every one of them.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/sketch-and-render.png" alt="A generated sketch and the painting it produces">
  <figcaption>Part of one rollout's <code>draw()</code> and what it renders to. The comments are the model's own. Reward 0.864, 129 lines, step 22. The full source of every painting is in the rollouts dataset.</figcaption>
</figure>


His blog post saved me a lot of time I could have wasted iterating on the prompt. 
A long API reference makes the model
invent methods that do not exist, and his 200 GEPA iterations converged on a strict
allowlist with no documentation. I saw the same failures and wrote the allowlist by
hand. My only addition to that recipe is one sentence: paint
each petal two or three times, a big pass first and a smaller, more opaque one inside
it. This small change made my outputs a lot more colorful.

> **Note.** If it's the first time you hear about
> [GEPA](https://huggingface.co/papers/2507.19457), it is an automatic prompt optimizer.
> A language model reflects in plain words on where the current prompt failed and
> proposes a better one, and the loop repeats.

The gate is the final piece. The sketch has to compile, use the library instead of direct p5 calls, put real pigment on the canvas, and not try to trick the scorer, for example by writing text on the canvas.


## The pool is the reward function

The pool consists of [178 paintings](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool) divided into
two tiers based on my personal preferences, `love` and `okay`. All of them are actually generated by a model. Four open-weight models,
called through Inference Providers, wrote p5.brush sketches, each one working from a
real, openly licensed photo of a hibiscus from iNaturalist. A vision model gave written
feedback on every sketch, over three refinement iterations. Every final render was then rated one at a time, by me, and 178 made the cut.

| generator | number of paintings |
| --- | --- |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 64 |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 57 |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 35 |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 22 |

Here I chose four different families of models to test their different styles. These four were the open models that produced a valid sketch every time in a quick reliability check, and two other candidates were dropped for failing it. If you want to produce your own pool, you might choose others.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/love-and-okay.png" alt="A love reference beside an okay one">
  <figcaption>The two tiers, as they actually look. Disagreeing with the rating is reasonable, somebody's judgement is now the reward function.</figcaption>
</figure>

The tiers do real work in the reward. When the pairwise judge draws four references,
half come from `love` and half from `okay`, so the policy always faces some rivals it
can sometimes beat, and a win pays the same against either tier. This is one of my few
deliberate changes: the original compares against its top tier only, and I kept the
easier tier in the draw so a weak early policy still gets signal.

No human-made painting is in there, which is a real limitation. p5.brush is a niche
library, and the human work that exists in it with accessible code is a handful of pieces, nowhere near what a training corpus
would need, as his blog also notes.

The interesting idea here, as I've already discussed previously, is that the model will learn to imitate what the pool contains. If we point the environment at a different dataset, the reward would automatically change without touching a single line of code. For the dataset I generated and openly share, I also include the source sketch.

If we look closer at the judges, the two of them answer different questions. HPSv3 decides whether it is a flower, and
the pairwise judge decides whether it is well painted in the style I chose.

## Just one more yolo run

Before anything worked, there was a long stretch of flat reward curves. If you've tried to reproduce a research paper/blog without open artifacts, you probably can relate. Every run
tested what I thought was a reasonable theory about what was wrong. As always, starting from something easier that works and then building on top of that was the answer.
A simple control task, with no browser and no judges, was the first attempt that learned, and the reason was that my learning rate was just too low.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/four-curves.png" alt="Three reward experiments flat against the run that worked">
  <figcaption>Three experiments on the reward, three flat lines. I swapped the pool, removed renderer noise and turned the pairwise judge off, and none of it moved the curve. The run that moves changed the trainer configuration, so the difference is the trainer, not the reward mix.</figcaption>
</figure>

Another change that cost me time to find was adjusting correctly LoRA. The usual `target_modules` list assumes a dense model, and [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) is a mixture of experts that names most of its projections differently, so the adapter was training ten layers out of forty. I solved this by changing it to `all-linear`, which reaches every linear layer. The routed experts in this architecture are fused tensors that even `all-linear` leaves frozen, but everything else gets an adapter, and that was enough to learn.

The fix was four changes in TRL's [`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer):

| setting | from | to | why |
| --- | --- | --- | --- |
| learning rate | 2e-5 | **5e-5** | the ceiling *LoRA Without Regret* uses for GRPO |
| scheduler | `linear` | **`constant_with_warmup`** | linear decay had spent most of the learning rate by mid-run, so the reward never took off |
| `scale_rewards` | `group` | **`none`** | one gate rejection was shrinking every other advantage in the group |
| `target_modules` | hand list | **`all-linear`** | reach every linear layer |

These four changes unlocked the first successful run (`hps-only`), with the reward clearly improving.

With that configuration, all three runs learn. Both judge runs were launched for 200 steps and stopped at 110, with the reward still climbing slowly. A step takes fifteen to eighteen minutes, and the comparison between mixes was already stable, so I stopped both to save compute. Mean group reward over the first and final third of each run:

| run | steps | first third | final third | Δ |
| --- | --- | --- | --- | --- |
| `hps-only` | 60 | 0.58 | 0.71 | +0.13 |
| `judge-led` | 110 | 0.45 | 0.72 | +0.27 |
| `hps-led` | 110 | 0.57 | 0.82 | +0.24 |

![The three runs, one curve each](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-mixes.png)

The three curves line up with how much weight my taste carries. The more the judge
weighs, the lower the start and the noisier the climb. `judge-led` spent its first thirty steps nearly flat
before it moved. It is the same move that solved the debugging. Shrink the problem until
something learns, then add the hard parts back one at a time.

The pairwise judge term itself climbed in the two runs that used it. The model wins more
comparisons against the pool as training advances, which is the claim `hps-only` could
not make. No group in any run collapsed to identical rewards, the GRPO failure mode that
kills the gradient. For the curious, the per-metric curves (HPSv3, paint coverage,
entropy) are [in the
repository as CSV](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour/results).

The full launch command, the hardware and the two environment variables that turn this
into the other two runs are in [the
recipe](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour).

## What it actually learned

**In every run, the first thing the model learned was to stop producing bad
paintings**, the near-blank canvases and shapeless washes that score under 0.3 in total
reward. In
`hps-only`, three quarters of the rise in the group mean comes from bad paintings
becoming rare. In the judge runs the collapse is even steeper: rollouts under 0.3 fall
from 99 to 16 across `judge-led`'s thirds, and from 37 to 4 in `hps-led`.

![Median against best, per step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/median-vs-best.png)

This is why the obvious visual, the best painting of each step, shows almost no difference in
`hps-only`. It moves **+0.034** across the run while the median moves **+0.155**. The
learning is visible in the middle of the distribution.

**What the pairwise judge changes is the top.** In `hps-only`, paintings got more
reliable without getting better. The quality of the good ones added just +0.03 to the
group mean, and once HPSv3 saw petals around a centre and a stem, it stopped asking for
more pigment. With the judge on, the other half of the story appears. *Better* here means closer to
the pool, so closer to what I rated as "good" or something I liked more. That added +0.12 in `judge-led` and +0.16
in `hps-led`, the best of each step rose too, and
paint coverage doubled in both runs (0.11 to 0.23, and 0.13 to 0.30) where `hps-only`
barely moved it. With a reference left to beat, a good painting can still get better,
and the model starts being rewarded for using more pigment.

One more finding. The model ignores an explicit instruction, and it is right to. The
system prompt asks for fifteen to thirty filled shapes. If we look at the real mean, it is between 7 and 9, and
`n_shapes` barely correlates with reward in any run (+0.000, −0.14, +0.07). The policy
is not rewarded for obeying that sentence, so it does not obey it.

There is also a ceiling on the `hps-only` route. If every rollout matched its good
ones, that run's mean would sit at 0.771. Whether more steps would break it is
an open question.

The paintings also show something that the tables miss. Within each run, they all look
similar. As training advances, the rewards inside each group get closer together, and the
median paintings in the opening video look like takes of the same flower. That is GRPO
doing what it is designed to do with a pool built from one subject. The pool decides
what counts as variety, the same way it decides what counts as quality. If the reward
only pays for matching one flower, the model learns to paint that one flower. More diverse output would need a
more diverse pool, and building one is more curation work. Surya's newer compositions are an example of this. [Alex Yango's animal
paintings](https://x.com/alexyango/status/2091696296931574217) are the same recipe with
different choices in the pool. This is the biggest difference between an aesthetic
reward and a maths grader. Behind the number there is a very human job, deciding what
belongs in the reward set. [Jason Liu's essay on
taste](https://x.com/jxnlco/status/2073819508729684462) says the general version in one
line. AI shifted the bottleneck from making to noticing.

Surya closes his blog with some of his favourites. Instead of picking mine, below is a
wall with the 178 paintings the reward scored highest across the two judge runs, the
same number the reference pool holds, in no particular order. Open it and pick your own.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/pick-your-favourites.png" alt="178 paintings from the two judge runs, unlabelled">
  <figcaption>The reward's 178 favourites from the two judge runs, shuffled. Now pick yours.</figcaption>
</figure>

To choose, you looked at many and kept a few, and that is exactly the job that built the
reward of this project. Every painting of every run, with its sketch and its reward, is
in the rollouts datasets if you want them all.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-styles.png" alt="The median painting of the last step of each run">
  <figcaption>The last step's median painting of each run, in the same order as the opening video. Same base model, same pool, three reward mixes, three styles.</figcaption>
</figure>

## Infra is hard

This project is mostly infra. A run needs a trainer, two Spaces, an inference router and
a websocket to stay healthy for hours straight, and every piece that fails quietly turns
into a wrong number somewhere else. Half the work is checking that the number you read
matches what actually happened.

**Failures of the infrastructure were entering the reward as zeros.** A render that
timed out or a scorer that did not answer scored the same as a bad painting, 0.0 inside
the group. Across all my runs that was about 1.5% of rollouts, and in the worst run it
reached 5.2%. That trains the model on noise, so those paths now
return `None` and the rollout is excluded from the group.

**I also found a bug in OpenEnv, and sent the fix upstream.** The client keeps one
persistent websocket, and a socket closed by the far end stayed cached, so every later
call failed even though the environment was healthy. It cost me two half-finished runs
to find it. The fix is [submitted
upstream](https://github.com/huggingface/OpenEnv/pull/1103), and the runs launched with
it have been running clean since.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/step-11-vs-step-12.png" alt="Steps 11 and 12 of the judge-led run, best four of each">
  <figcaption>Two consecutive steps of the <code>judge-led</code> run, best four of each. Whether step 12's paintings look half a point worse is for you to decide.</figcaption>
</figure>

**The reward of a step depends on which references it drew.** The pairwise judge
samples four references per step, so every step faces a different set of rivals, and
some draws are simply harder. GRPO itself is mostly safe, because advantages are
computed inside the group and a hard draw moves the whole group together. The curve I
was reading was not safe, and some of what looked like a bad step was just a hard draw.
The image above is one example. Step 12 scored half a point below step 11 mostly because
it drew the hardest references of the run, while the paintings themselves look close.

## What it costs

Rounded numbers, and only for the runs that finished.

| piece | what it needs |
| --- | --- |
| trainer | 1 H200. **18 hours** for 60 steps, about **34** for 110 |
| HPSv3 | an `a100-large` Space, up for the whole run |
| the environment | a `cpu-upgrade` Space, which renders comfortably in time |
| the pairwise judge | Inference Providers quota for `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| the pool, one-off | openly licensed photos from iNaturalist, Inference Providers quota for the four generators, and rating is your own hours |

A step is eight rollouts and takes fifteen to eighteen minutes, of which **70 to 80% is
rendering**. A single render takes 69 to 96 seconds against a 90 second deadline. Part of that is
expected, the Space has no GPU, so Chromium renders the WEBGL canvas in software and
p5.brush's bleeds and textures are heavy pixel work. Even so, I expected it to be
faster, and I have not found the full cause.

A scorer can cost more than the training that uses it: HPSv3 has to be up for the whole
run, so pause the Space, or set its sleep timer, when the run ends.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/infra-diagram.png" alt="The infrastructure: what is billed during a run, and what outlives it">
  <figcaption>Four paid services have to stay healthy at once. Only the Hub and trackio outlive the run.</figcaption>
</figure>

Everything runs on [HF
Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), with the environment as
a [Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker) and metrics in
[trackio](https://huggingface.co/docs/trackio).

## What I would try next

The rule of this project was to reproduce the recipe with every resource open, not to
improve it, so a list of untried ideas piled up along the way. These are the ones I
would actually try, in order of how much evidence there is.

**Multi-step, and letting the model see what it paints.** This is the first thing I
would try. The original blog trains single turn, so I trained single turn, and in this
setup the model paints with its eyes closed. No image ever goes in, and the only
feedback it gets is one number. The evidence that a feedback loop works is the pool
itself. The reference paintings came from models iterating three rounds under a vision
critic, and the later rounds are better. The material that defines the reward was made
with a loop the policy never gets.

**Smaller models.** There is evidence that 35B is more than needed. In my side
experiments a 4B already wrote valid sketches that passed the gate. If a 4B can learn
this, the cost of the experiment drops by an order of magnitude.

Other ideas on the list are SFT on the pool sources before starting RL, rewarding pigment
explicitly, moving the judge's reference mix from easy to hard as the run advances until
only `love` remains, widening the ten-method allowlist for more visual range (my
attempts on this crashed more sketches and broke the watercolour look), and checking how consistent the pairwise judge really is by
scoring the same image twice.

And the method is not specific to flowers. [Alex Yango painted animals with the same
mechanism](https://x.com/alexyango/status/2091696296931574217), and [Brendan Hogan
trained canvas animations](https://x.com/brendanh0gan/status/2092650655789855222)
against a pool of hand-rated clips. I had also played with something similar before using [Simon
Willison's pelican
benchmark](https://huggingface.co/blog/sergiopaniego/pelican-env-openenv), where code is
rendered to an image and scored.

And underneath all of it sits the question this project cannot close. **178 paintings
made by models define what this trained model considers beautiful.** The pool is the bottleneck,
and it is the part of the pipeline with no principled answer.

## Everything is published

| artifact | where |
| --- | --- |
| the recipe, and how to reproduce it | [`02-watercolour/`](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour) |
| the reference pool, with every source sketch | [`watercolour-reference-pool`](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool) |
| the environment, ready to duplicate | [`watercolour-env`](https://huggingface.co/spaces/HuggingEnvs/watercolour-env) |
| the HPSv3 scorer, ready to duplicate | [`watercolour-hpsv3`](https://huggingface.co/spaces/HuggingEnvs/watercolour-hpsv3) |
| the hps-only adapter and rollouts | [`watercolour-grpo-hps-only`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-only) · [`watercolour-rollouts-hps-only`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-only) |
| the judge-led adapter and rollouts | [`watercolour-grpo-judge-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-judge-led) · [`watercolour-rollouts-judge-led`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-judge-led) |
| the hps-led adapter and rollouts | [`watercolour-grpo-hps-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-led) · [`watercolour-rollouts-hps-led`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-led) |
| the training curves | live: [`judge-led`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-judge-led) · [`hps-led`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-hps-led) · [`hps-only`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-hps-only), and the CSV files in `results/` |
| all of it | [Paint with Code](https://huggingface.co/collections/HuggingEnvs/paint-with-code-6a955b79d63f67f1631d9be6) |

The per-rollout numbers in this article can be recomputed from the published datasets.
Nothing here depends on any Space staying switched on.

The method and original idea are Surya Narreddi's. The library is Alejandro Campos Uribe's.

