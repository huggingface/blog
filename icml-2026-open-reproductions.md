---
title: "What We Learned by Reproducing 2,200 papers from ICML"
thumbnail: /blog/assets/icml-2026-open-reproductions/thumbnail.png
authors:
  - user: abidlabs
---

# What We Learned by Reproducing 2,200 papers from ICML


Back in July, we ran a [hackathon](https://huggingface.co/ICML-2026-agent-repro) where more than 1,200 community members brought their own coding agents and tried to reproduce the papers published at ICML 2026, claim by claim. In 19 days, participants published [6,816 Trackio logbooks](https://icml-2026-agent-repro-challenge.static.hf.space/gallery.html) reproducing 2,226 papers, about a third of the conference 🤯

In this post, we're sharing what we learned from running this hackathon, and what it suggests about _the role humans will play_ when agents are doing the research experiments.

## More papers than anyone can review

Questions about how reproducible AI research really is are older than the current AI wave. But these questions are exacerbated by scale. ICML 2026 received **23,918 submissions and accepted 6,352 papers**, roughly double the previous year, continuing an exponential trend that is at least partly driven by AI agents making it faster to run experiments and write them up.

Reviewing capacity has not doubled along with it. Reviewers at most conferences are volunteers who may not have the time or expertise to fully review a paper. Here is a review of one accepted ICML 2026 spotlight paper, in the reviewer's own words:

![An OpenReview reviewer admitting the proofs were not checked carefully](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/icml-2026-open-reproductions/reviewer-quote.png)

> "My low confidence score is because I did not check all the proofs carefully."

Note that **this paper got strong scores and a spotlight**. Keep it in mind, because we will come back to this exact paper later in the post, and to what happened when we finally did check the proofs carefully.

What has changed, though, is that the same technology driving the flood of submissions can also help us keep up with it. Coding agents like Claude Code, Codex, Cursor, and Pi can now read a paper, write the code, launch the experiments, and report back on what they found. Checking a paper carefully used to cost a reviewer a weekend; an agent can attempt it in an afternoon, in parallel, thousands of times over.

So the question we wanted to ask was: **if we actually re-examined a major conference at scale, and tried to reproduce every paper, what would we find?**

## The hackathon (July 15 - August 2nd)

Rather than audit papers ourselves, we opened it up to the whole community, with all the diversity of agent frameworks, compute budgets, and scientific taste that brings. From July 15 to August 2, 2026, the [ICML 2026 Open Reproductions challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge) worked like this:

1. **Pick a paper.** We indexed all 6,341 accepted ICML 2026 papers with their abstracts and extracted the core scientific claims of each one, so an agent could start from a concrete, checkable target rather than a 40-page PDF. Multiple people reproducing the same paper was encouraged.
2. **Bring your own agent.** Participants used Claude Code, Codex, Cursor, OpenResearch's `orx`, and everything in between. We provided a streamlined interface so an agent could pull the paper, its claims, and the challenge instructions with a single command.
3. **Reproduce, then publish everything.** Every run produced a [Trackio](https://huggingface.co/docs/trackio) logbook: a static Hugging Face Space containing the write-up, the code that ran, the artifacts it produced, and (optionally) the full agent execution trace uploaded as a Hugging Face Dataset. The auditing process itself had to be auditable.
4. **Get judged.** An automated Logbook Judge (running an open-weights model, GLM-5.2) re-read every logbook and issued a per-claim verdict: **verified**, **falsified**, **toy** (evidence at reduced scale), or **inconclusive**. The judge was explicitly instructed to treat each logbook's self-assessment as untrusted.

Participants received $20 in Hugging Face compute credits to run experiments on [HF Jobs](https://huggingface.co/docs/hub/jobs); across the challenge, participants launched 2,962 cloud jobs. Where a full reproduction was impossible, for example when a paper's dataset was proprietary or its checkpoints unreleased, participants ran toy reproductions on synthetic data mimicking the original's properties.

Here is what a finished reproduction looks like:

![Anatomy of a reproduction logbook: logbook pages, agent traces, and artifacts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/icml-2026-open-reproductions/anatomy.png)

By the numbers, this hackathon was probably the largest attempted reproduction of a scientific conference:

- **1,221** community members joined the [organization](https://huggingface.co/ICML-2026-agent-repro)
- **6,816** reproduction logbooks published
- **2,226** papers attempted, **34% of the entire conference**, many by several independent teams
- **35,908** claims judged, with all verdicts frozen in a public dataset at challenge close
- **2,962** HF Jobs launched; **274** full agent-trace datasets published on Hugging Face

## What we found

Aggregating the claim-level verdicts per paper:

**51% of examined papers (1,103) had at least one claim independently verified.** Of those, 266 papers were fully reproduced, with every extracted claim verified, and 632 more were partially reproduced with nothing falsified. In total, 3,978 individual claims were confirmed with real experiments.

**23% of examined papers (496) had at least one claim falsified or contested.** That includes 49 papers where all claims were falsified and nothing could be verified, and, maybe most interestingly, 242 papers where independent reproduction teams reached *opposite* verdicts on the same claims. Reproducibility is not binary; it is adversarial.

The remainder sat in the middle: 502 papers with toy-scale evidence only, and 280 where nothing could be established either way (missing artifacts were the most common cause).

### Reproductions done well

Some papers came through the gauntlet looking great, and the community's best logbooks are worth reading in their own right:

- **["Flat Minima and Generalization: Insights from Stochastic Convex Optimization"](https://huggingface.co/spaces/visv-Bro/repro-flat-minima-and-generalization-insights-from-stochastic-convex-optimization)** was reproduced by 20 independent teams, 12 of which verified every claim. The one linked included and published the full agent trace.
- **["A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness"](https://huggingface.co/spaces/gchauhan/repro-a-coin-flip-for-safety-llm-judges-fail-to-reliably-measure-adversarial-robustness)** had 14 of 17 logbooks verify every claim. A paper about unreliable LLM judges holding up under scrutiny by LLM agents :)

### Falsifications, and what happened when we checked them

35 participants formally claimed they had falsified something. We adversarially re-verified every claimed falsification: re-reading the paper, re-reading the logbook, and re-deriving the math or re-implementing the experiment from the paper's own text.

A few of the confirmed falsifications, linking to the logbook that found it:

**The paging paper from the introduction.** The reviewer who did not check the proofs carefully? The paper, "Towards Optimal Robustness in Learning-Augmented Paging," claims its algorithm achieves robustness \\(H_k + O(1)\\). [One participant's logbook](https://huggingface.co/spaces/Auenchanters/repro-towards-optimal-robustness-in-learning-augmented-paging) measured the additive term growing like \\(0.38 \ln k\\) and located the exact step of the proof that breaks. Our own re-implementation extended the sweep to k = 1,024 and confirmed the growth at roughly nine sigma. The true robustness is \\(H_k + \Theta(\log k)\\). 



**A theorem that falls after step 224.** "Attention's forward pass and Frank-Wolfe" proves that token particles collapse to the origin whenever the origin starts inside their convex hull. Three independent teams found counterexamples, with violations first appearing at t = 224, ~3,800, and 6,416 steps, which neatly explains why everyone else "verified" the claim: finite-horizon checks stop too early. [The cleanest counterexample](https://huggingface.co/spaces/SabaPivot/repro-attention-frank-wolfe) is stated in exact rational arithmetic, so there is no floating-point ambiguity to hide behind. The authors confirmed the same day and are working on a fix.

**Theory written for one loss, results produced by another.** In "Self-Distillation Enables Continual Learning," the paper's central equation and its entire theory section analyze reverse KL divergence, but the released code's default, which per the authors produced all the paper's results, computes forward KL. [The logbook that caught it](https://huggingface.co/spaces/codemaivanngu/repro-self-distillation-enables-continual-learning) also failed to reproduce the paper's headline +4pp result under the authors' own code and data. The authors have already uploaded a clarified version to arXiv.

**An evaluation diluted by padding.** In "Do Transformers Need Three Projections?", [a participant discovered](https://huggingface.co/spaces/stresearch-dev/63430) that ~66% of evaluated label positions were EOS padding tokens that train to near-zero loss, deflating perplexity roughly threefold. The abstract's "3.1% quality cost for 50% cache reduction" becomes roughly 9.4% once corrected.

**🚨 False falsifications.** Sometimes we found flaws in an attempted reproduction. One logboook claimed dramatically, "the paper's method is 2x slower than the baseline"; this turned out to be an arithmetic bug in the *reproduction*: per-trajectory time compared against per-batch-of-50 time. Correctly normalized, the participant's own data confirms the paper's claimed 8x speedup.

### Talking to authors

We have begun writing to the authors of every confirmed finding, with a simple framing: here is what we found, here is all the evidence, do you agree or is our analysis wrong? The early responses have been very positive:

![An author response confirming the finding and promising an arXiv correction](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/icml-2026-open-reproductions/author-response.png)

So far authors have confirmed findings on multiple papers, two arXiv corrections are in flight, and in one case an author had quietly fixed the error in a new arXiv version a month before the challenge found it, which we count as independent convergence 🤗


### The role of humans

The most interesting question that this hackathon raises is: do humans still have a role in reviewing papers? We think so, for several reasons:

**Pure agent execution hits real limits.** Agents got stuck in local loops, misread scale-dependent behavior (several "verified" verdicts on the paging paper came from checks that stopped before the log-k growth became visible), and occasionally built an entire falsification on top of a units mismatch. The challenge's most reliable results came from workflows where a human was steering: re-pointing the agent, questioning an assumption, or deciding that an experiment's premise was wrong before burning a week of compute on it.

**Some evaluation is irreducibly human, for now.** Our [human-in-the-loop winner](https://huggingface.co/spaces/KwabsHug/repro-robuq-pushing-dits-to-w1-58a2-via-robust-activation-quantization) is the clearest example. The paper claimed stable image generation under extreme quantization. Numerical metrics said "no collapse"; whether the images were actually *usable* was a perceptual question. The agent built a purpose-built review UI, and the human personally judged all 128 image pairs, with the annotations committed to the repo and the agent validating their consistency afterward. The published agent trace captures the whole exchange, down to the participant asking how the review tool works and coming back with "I have gone over the pairs and put the csv in the repo, please check."

So what are our roles as human reviewers? We think it is to **manage intelligence effectively.** Much like a professor or principal investigator (PI) sets up an environment where grad students can do good work, with compute, harnesses, data access, and targeted feedback at the right moments, the participants who got the most out of their agents were the ones who built the right environment and _asked the right questions_, then let the agents do the running.

## Thank you

To the 1,221 people who joined, the winners, the authors who responded with grace, and our organizers at Hugging Face and alphaXiv: thank you. Every logbook, verdict, trace, and artifact from the challenge is public, starting from the [challenge Space](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge). We think this is the largest open, claim-by-claim audit of a machine learning conference to date, and we would love for it not to hold that record for long.

Stay tuned for future reproduction events. 🤗
