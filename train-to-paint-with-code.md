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
technical report is coming, so ensure you follow him. The idea is his, and he comes at it from
the art and design side, where [his skills are way beyond
mine](https://x.com/kickingkeys/status/2094901433149612118). My attempt comes from the engineering side: reproduce the recipe,
see the code, publish every piece.

> **Note:** for the context behind the project, told by Surya himself, watch [this
> video of his thesis](https://vimeo.com/1190839818).

In this article I try and reproduce his idea with [TRL](https://huggingface.co/docs/trl) and
[OpenEnv](https://github.com/huggingface/OpenEnv). The reference pool dataset, the RL
environment, the training scripts and the trained models, all open.

The whole pipeline runs on Hugging Face, end to end:
* training on [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
* the environment and the scorer as [Spaces](https://huggingface.co/docs/hub/spaces)
* the pairwise judge through [Inference Providers](https://huggingface.co/docs/inference-providers)
* and every artefact on the Hub

I have followed the original blog step by step, and only changed something when strictly needed. Every idea of my own went into a list instead of into the experiment, and that list became "What I would try next" at the end, next to the full list of published artefacts. The rest of the article documents the process, the
problems it hit and what fixed them.

<video controls autoplay muted loop playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-runs-evolution.mp4"
title="Three runs evolving in parallel"></video>

*Three runs, one per reward mix, evolving in parallel: the median painting per step as
training advances. Which run is which is what this article explains.*

## Why people loved it

The paintings look loose, imperfect, handmade, at a moment when image models produce
perfect pictures. My guess is that this contrast is a big part of why the video went
viral. It reminded me of the early days of generative AI art, when the point was
exploring the medium:
[DeepDream](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/)
(2015) was a debugging tool that people turned into art, works like [Edmond de
Belamy](https://en.wikipedia.org/wiki/Edmond_de_Belamy) (2018) came from artists probing
what a GAN could do, and artists like [Mario
Klingemann](https://quasimondo.com) spent those years making [portraits with neural
networks](https://artsandculture.google.com/asset/memories-of-passerby-i-mario-klingemann/aAHG7iV3aXme8g).

This project feels closer to those early days. In his thesis, Surya describes the path
that led here: he started with prompting, where more detail in the prompt buys more
control over the image, and realised that training the model itself goes further. The
other half of the idea is the medium. The model writes a program of about 150
lines of JavaScript that paints the image. The output is code. You can read it, edit it and run it
again, and the decision behind each brushstroke is visible. And the style comes from a
restriction where the model is only allowed *ten of the library's methods*. More on that below.

In that same period, [Anna Ridler](https://annaridler.com/works/myriad-tulips)
photographed thousands of tulips, hand-labelled every one, exhibited the dataset itself
as the artwork, and later trained a model on it. I found her work through the references
AI agents brought back while building this project, which felt fitting, because this
project does something very similar, to curate a set of images by hand, then train against
it. She described that labour as *"repetitive, time-consuming, often unauthored, but
necessary"*, and that is also the most accurate description of the hardest part of what
follows.

## RL over taste

Most of the recent RL on language models uses rewards you can verify: maths with a known
answer, code that passes tests, graders that are right or wrong and cheap to run. This
project is closer to the older exception, RLHF, which learns a reward model from human
preferences.

Here the reward is aesthetic preference. There is no *correct* answer. The real question
of this project is whether you can do RL over taste.

The reward, as his blog defines it and as my environment implements it:

| term | weight | what it measures |
| --- | --- | --- |
| `gate` | 0.05 | the sketch compiles, paints something, does not cheat |
| `length` | 0.05 | a soft push towards longer sketches |
| pairwise judge | 0.60 | style, compared against references drawn from a pool |
| [HPSv3](https://huggingface.co/MizzenAI/HPSv3) | 0.30 | aesthetic preference on the render |

[HPSv3](https://huggingface.co/MizzenAI/HPSv3) is an open 7B preference model. Give it an image and a text description, and it returns
a score for how much a person would prefer that image. It was trained on a large set of
human choices between pairs of images, so its score is everyone's taste, averaged. The pairwise judge is
[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), a
general vision model called through HF Inference Providers. The pairwise judge sees the candidate
painting
next to four references drawn from the pool, with written criteria about the painting,
each comparison in both presentation orders, and its score is the share of comparisons
the candidate wins. Its only standard is the pool, so its score is my taste, as encoded
in those ratings.

![The reward pipeline, piece by piece](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/reward-diagram.png)

*Two of the four terms are models. Both are proxies for someone's taste.*

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

I started with `hps-only` to check that the pipeline could learn at all. Once it did, there was no reason to run it longer, and I launched the two longer runs instead. The question that the longer runs ask is how much of HPSv3's power can you hand to the pairwise judge? The more weight the judge carries, the more
the reward means *my* taste instead of everyone's, and the harder it should be to climb.
Push it far enough and, if the style is too complex for the model, learning could stop
entirely.

Fortunately, it did not stop: both runs with the pairwise judge on learned too. The hand-rated pool
can steer the policy. The numbers are below.

> **Note.** A frontier model already paints this from a prompt today. That is the
> starting point. The work here is teaching a smaller model to do it, and that puts the
> effort in the dataset and in what you optimise against.

## The RL environment you need to build

The environment wraps everything that sits between the model and the reward: the library
the model paints with, the prompt that restricts it, the headless Chromium that renders
each sketch, and a gate that rejects cheats.

The library does more of the work than it seems.
[p5.brush](https://github.com/acamposuribe/p5.brush), by
[@acamposuribe](https://x.com/acamposuribe), simulates a medium rather than drawing
shapes: pigment bleeds past the edges of a fill, paper has texture, strokes have mass,
flow fields drag brushwork around. When the model calls `brush.fillBleed(0.25)` it is deciding how far the ink
runs. The library knows watercolour, and the model learns to drive it.

> **Note.** The author of p5.brush had been trying to teach a machine to paint long
> before any of this. In 2022 he made a generative art series that hides a diary about
> teaching p5.js to draw like a child: *"It is barely able to use the crayons [...] It
> cannot follow simple commands. I'm done for today, very infuriating."* The series was
> meant to have three pieces, and he made two. When Surya's video went viral, [he
> quoted it](https://x.com/acamposuribe/status/2091668313449316651), sharing that diary
> and saying this work is the third piece arriving on its own.

The library exposes 47 methods while the prompt allows only 10. This is done in order to restrict the model. The thirty-seven methods left out would break the watercolour look. The ten methods used are `scaleBrushes`, `noStroke`, `fill`, `noFill`, `fillBleed`, `fillTexture`, `beginShape`, `vertex`, `endShape`, and `circle`.  These can only paint filled shapes, and the library adds the watercolour bleed to every one of them. The model paints watercolours because it cannot do anything else.

![A generated sketch and the painting it produces](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/sketch-and-render.png)
*Part of one rollout's `draw()` and what it renders to. The comments are the model's
own. Reward 0.864, 129 lines, step 22. The full source of every painting is in the
rollouts dataset.*


His blog post saved me the most time on the prompt. A long API reference makes the model
invent methods that do not exist, and his 200 GEPA iterations converged on a strict
allowlist with no documentation. I saw the same failures and wrote the allowlist by
hand. My only addition, and my stand-in for his GEPA, is one sentence of craft: paint
each petal two or three times, a big pass first and a smaller, more opaque one inside
it.

The gate closes the circle: the sketch has to compile, use the library instead of bare
p5 calls, put real pigment on the canvas, and not write text on it to trick an image
scorer.

## The pool is the reward function

[178
paintings](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool) in
two tiers, `love` and `okay`, every one of them model output. Four open-weight models,
called through Inference Providers, wrote p5.brush sketches, each one working from a
real, openly licensed photo of a hibiscus from iNaturalist. A vision model gave written
feedback on every sketch, over three refinement rounds. Every render was then rated by
hand, one at a time, and 178 made the cut.

| generator | paintings |
| --- | --- |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 64 |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 57 |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 35 |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 22 |

Here we choose four different families of model to test of different styles. These four were the open models that produced a valid sketch every time in a quick reliability check, and two other candidates were dropped for failing it.

![A love reference beside an okay one](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/love-and-okay.png)
*The two tiers, as they actually look. Disagreeing with the rating is reasonable:
somebody's judgement is now the reward function.*

The tiers do real work in the reward. When the pairwise judge draws four references,
half come from `love` and half from `okay`, so the policy always faces some rivals it
can sometimes beat, and a win pays the same against either tier. This is one of my few
deliberate changes: the original compares against its top tier only, and I kept the
easier tier in the draw so a weak early policy still gets signal.

No human-made painting is in there, which is a real limitation. p5.brush is a niche
library, and the human work that exists in it is a handful of pieces, nowhere near a
training corpus, as his blog also notes.

**What the pool contains is what the model learns to imitate, and that is in no
hyperparameter.** Point the environment at a different dataset and you have changed what
the agent is rewarded for without touching a line of code. Mine ships with the source
sketch and the photo licence for every painting.

The two judges answer different questions: HPSv3 decides whether it is a flower, and
the pairwise judge decides whether it is well painted in the style I chose. A small
test, changing HPSv3's text prompt, confirmed the split.

## Just one more yolo run

Before anything worked, there was a long stretch of flat reward curves. Every run
tested a reasonable theory about what was wrong. A run takes hours, so I always launched
the next one before I had fully read the last. I changed the reward three ways, and the
curve stayed flat every time. A simple control task, with no browser and no judges, gave
the answer: the model could learn, my learning rate was just too low.

![Three reward experiments flat against the run that worked](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/four-curves.png)
*Three experiments on the reward, three flat lines. The run that moves changed the
trainer configuration. One of the flat controls also ran with the pairwise judge off,
so the difference is the trainer, not the reward mix.*

The change that cost the most to find was the LoRA one. The usual `target_modules` list
assumes a dense model, and
[`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) is a mixture of
experts that names most of its projections differently, so the adapter was training ten
layers out of forty. You could hunt down the right names for each architecture;
`all-linear` just reaches everything.

The fix was four changes, all in [TRL's
`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer), none in the environment:

| setting | from | to | why |
| --- | --- | --- | --- |
| learning rate | 2e-5 | **5e-5** | the ceiling *LoRA Without Regret* uses for GRPO |
| scheduler | `linear` | **`constant_with_warmup`** | linear decay had spent most of the learning rate by mid-run, so the reward never took off |
| `scale_rewards` | `group` | **`none`** | one gate rejection was shrinking every other advantage in the group |
| `target_modules` | hand list | **`all-linear`** | reach the whole model |

The four went in together, so this run does not rank them. The strict claim is only that
this configuration learns and the previous one does not.

With that configuration, all three runs learn. Both judge runs were launched for 200
steps and stopped at 110, still climbing: by then the three curves already told the
story this article needed. Mean group reward across the thirds of each run:

| run | first third | second | third | slope t |
| --- | --- | --- | --- | --- |
| `hps-only` | 0.58 | 0.64 | 0.71 | +6.4 |
| `judge-led` | 0.45 | 0.65 | 0.72 | +10.5 |
| `hps-led` | 0.57 | 0.74 | 0.82 | +15.6 |

![The three runs, one curve each](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-mixes.png)

Read the curves along the taste axis: the more weight my taste carries, the lower the
start and the noisier the climb. `judge-led` spent its first thirty steps nearly flat
before it moved. It is the same move that solved the debugging: shrink the problem until
something learns, then add the hard parts back one at a time.

The pairwise judge term itself climbed in the two runs that used it: the model wins more
comparisons against the pool as training advances, which is the claim `hps-only` could
not make. No group in any run collapsed to identical rewards, the GRPO failure mode that
kills the gradient. The slope t is a least-squares fit of per-step mean reward against
step index, and the per-metric curves (HPSv3, paint coverage, entropy) are in the
repository as CSV.

The full launch command, the hardware and the two environment variables that turn this
into the other two runs are in [the
recipe](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour).

## What it actually learned

**In every run, the first thing the model learned was to stop producing bad
paintings**, the near-blank canvases and shapeless washes that score under 0.3. In
`hps-only`, three quarters of the rise in the group mean comes from bad paintings
becoming rare. In the judge runs the collapse is even steeper: rollouts under 0.3 fall
from 99 to 16 across `judge-led`'s thirds, and from 37 to 4 in `hps-led`.

![Median against best, per step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/median-vs-best.png)

This is why the obvious visual, the best painting of each step, shows almost nothing in
`hps-only`: it moves **+0.034** across the run while the median moves **+0.155**. The
learning is visible in the middle of the distribution.

**What the pairwise judge changes is the top.** In `hps-only`, paintings got more
reliable without getting better: the quality of the good ones added just +0.03 to the
group mean, and once HPSv3 saw petals around a centre and a stem, it stopped paying for
more pigment. With the judge on, the other half of the story appears: *better* here means closer to
the pool, so closer to what I rated as good. That added +0.12 in `judge-led` and +0.16
in `hps-led`, the best of each step rose too, and
paint coverage doubled in both runs (0.11 to 0.23, and 0.13 to 0.30) where `hps-only`
barely moved it. With a reference left to beat, a good painting can still get better,
and pigment goes from unpaid to paid.

One more finding. The model ignores an explicit instruction, and it is right to. The
system prompt asks for fifteen to thirty filled shapes; the real mean is 7 to 9, and
`n_shapes` barely correlates with reward in any run (+0.000, −0.14, +0.07). The policy
is not paid for obeying that sentence, so it does not obey it, under any reward mix.

There is also a ceiling on the `hps-only` route. If every rollout rose to the level of
its good ones, that run's mean would sit at 0.771. Whether more steps would break it is
an open question.

And the paintings add one thing the tables only hint at: they all look alike, within
each run. The spread of rewards inside each group narrows steadily as a run advances,
and the median paintings in the opening video read as takes of the same flower.
That is GRPO working as designed against a pool built from one subject: the pool defines
variety exactly as it defines quality, and a policy paid to match one flower learns one
flower. More diverse output would need a more diverse pool, which is a curation decision
with a curation price. [Alex Yango's animal
paintings](https://x.com/alexyango/status/2091696296931574217) are the same mechanism
pointed at a different set of choices. This is where an aesthetic reward differs most
from a maths grader: the number is downstream of a very human job, deciding what belongs
in the set. [Jason Liu's essay on
taste](https://x.com/jxnlco/status/2073819508729684462) puts the general version in one
line: AI shifted the bottleneck from making to noticing.

Surya closes his video with some of his favourites. I am not going to pick mine. Below
is a wall with the 178 paintings the reward scored highest across the two judge runs,
as many as the reference pool holds, in no particular order. Open it and choose your own. What you just did to
choose, look at many and keep a few, is the exact job that built the reward of this
project. Every painting of every run, with its sketch and its reward, is in the
rollouts datasets if you want the full haystack.

![178 paintings from the two judge runs, unlabelled](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/pick-your-favourites.png)
*The reward's 178 favourites from the two judge runs, shuffled. Now pick yours.*

![The median painting of the last step of each run](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-styles.png)
*The last step's median painting of each run, in the same order as the opening video.
Same base model, same pool, three reward mixes, three styles.*

## Infra is hard

This project is mostly infra. A run needs a trainer, two Spaces, an inference router and
a websocket to stay healthy for hours straight, and every piece that fails quietly turns
into a wrong number somewhere else. Half the work is making the number you read be the
number that happened.

**Failures of the infrastructure were entering the reward as zeros.** A render that
timed out or a scorer that did not answer scored the same as a bad painting: 0.0 inside
the group, about 1.5% of my rollouts. That trains the model on noise, so those paths now
return `None` and the rollout is excluded from the group.

**I also found a bug in OpenEnv, and sent the fix upstream.** The client keeps one
persistent websocket, and a socket closed by the far end stayed cached, so every later
call failed even though the environment was healthy. It cost me two half-finished runs
to understand. The fix is [submitted
upstream](https://github.com/huggingface/OpenEnv/pull/1103), and the runs launched with
it have been running clean since.

![Steps 11 and 12 of the judge-led run, best four of each](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/step-11-vs-step-12.png)
*Two consecutive steps of the `judge-led` run, best four of each. Whether step 12's
paintings look half a point worse is for you to decide.*

**And the reward of a step depends on which references it drew.** The pairwise judge
samples four references per step, so every step faces a different set of rivals, and
some draws are simply harder. GRPO itself is mostly safe, because advantages are
computed inside the group and a hard draw moves the whole group together. The curve I
was reading was not safe: some of what looked like a bad step was just a hard draw. The
image above is one example: step 12 scored half a point below step 11 mostly because it
drew the hardest references of the run, while the paintings themselves look close.

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
rendering**. A single render takes 69 to 96 seconds against a 90 second deadline, and I
have not found out why.

A scorer can cost more than the training that uses it: HPSv3 has to be up for the whole
run, so pause the Space, or set its sleep timer, when the run ends.

![The infrastructure: what is billed during a run, and what outlives it](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/infra-diagram.png)

*Four paid services have to stay healthy at once. Only the Hub and trackio outlive the
run.*

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
setup the model paints with its eyes closed: no image ever goes in, and the only
feedback it gets is one number. The evidence that a feedback loop works is the pool
itself: the reference paintings came from models iterating three rounds under a vision
critic, and the later rounds are better. The material that defines the reward was made
with a loop the policy never gets.

**Smaller models.** There is evidence that 35B is more than needed: in my side
experiments a 4B already wrote valid sketches that passed the gate. If a 4B can learn
this, the cost of the experiment drops by an order of magnitude.

Also on the list: SFT on the pool sources before starting RL, rewarding pigment
explicitly, moving the judge's reference mix from easy to hard as the run advances until
only `love` remains, widening the ten-method allowlist for more visual range (my first
attempts made things worse), and checking how consistent the pairwise judge really is by
scoring the same image twice.

And the method is not specific to flowers. [Alex Yango painted animals with the same
mechanism](https://x.com/alexyango/status/2091696296931574217), and [Brendan Hogan
trained canvas animations](https://x.com/brendanh0gan/status/2092650655789855222)
against a pool of hand-rated clips. I had also played this loop before, on [Simon
Willison's pelican
benchmark](https://huggingface.co/blog/sergiopaniego/pelican-env-openenv), code rendered
to an image and scored. To point this environment at a subject of your own, two things
are needed: a short description of the subject for the prompt and the pairwise judge,
and a reference pool for it. The description takes an afternoon. The pool is most of the
work.

And underneath all of it sits the question this project cannot close: **178 paintings
made by models define what this model considers beautiful.** The pool is the bottleneck,
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

The method is Surya Narreddi's. The library is Alejandro Campos Uribe's.

---
