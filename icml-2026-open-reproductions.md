---
title: "How Reproducible Is ICML 2026? We Asked 1,200 People and Their Agents"
thumbnail: /blog/assets/icml-2026-open-reproductions/thumbnail.png
authors:
  - user: abidlabs
---

# How Reproducible Is ICML 2026? We Asked 1,200 People and Their Agents

![Reproduce ICML 2026 papers with your agents](/blog/assets/icml-2026-open-reproductions/thumbnail.png)

**TL;DR:** This July, more than 1,200 community members pointed their coding agents at the accepted papers of ICML 2026 and tried to reproduce them, claim by claim. In 19 days they published 6,816 reproduction logbooks covering 2,226 papers, about a third of the conference, and 35,908 individual scientific claims were judged. Roughly half of the examined papers had at least one claim independently verified. About a quarter had at least one claim falsified or contested, and a dozen confirmed errors have already been reported to authors, several of whom are posting corrections to arXiv. Every logbook, verdict, artifact, and agent trace is public. This post covers how it worked, what we found, who won, and what it taught us about the role humans play when agents do the experiments.

## The problem: more papers than anyone can check

Questions about how reproducible AI research really is are older than the current AI wave. What is new is the scale. ICML 2026 received **23,918 submissions and accepted 6,352 papers**, roughly double the previous year, continuing an exponential trend that is at least partly driven by AI agents making it faster to run experiments and write them up.

Reviewing capacity has not doubled along with it. Reviewers are volunteers with day jobs, and it shows in the record. Here is a review of one accepted ICML 2026 spotlight paper, in the reviewer's own words:

![An OpenReview review admitting the proofs were not checked carefully](/blog/assets/icml-2026-open-reproductions/reviewer-quote.png)

> "My low confidence score is because I did not check all the proofs carefully."

The paper got strong scores and a spotlight. Keep it in mind, because we will come back to this exact paper later in the post, and to what happened when somebody finally did check the proofs carefully.

So the question we wanted to ask was simple: **if we actually re-examined a major conference at scale, claim by claim, what would we find?**

## The hackathon

Rather than audit papers ourselves, we opened it up to the whole community, with all the diversity of agent frameworks, compute budgets, and scientific taste that brings. From July 15 to August 2, 2026, the [ICML 2026 Open Reproductions challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge) worked like this:

1. **Pick a paper.** We indexed all 6,341 accepted ICML 2026 papers with their abstracts and extracted the core scientific claims of each one, so an agent could start from a concrete, checkable target rather than a 40-page PDF. Multiple people reproducing the same paper was encouraged: independent confirmations make every verdict stronger.
2. **Bring your own agent.** Participants used Claude Code, Codex, Cursor, OpenResearch's `orx`, and everything in between. We provided a streamlined interface so an agent could pull the paper, its claims, and the challenge instructions with a single command.
3. **Reproduce, then publish everything.** Every run produced a [Trackio](https://huggingface.co/docs/trackio) logbook: a static Hugging Face Space containing the write-up, the code that ran, the artifacts it produced, and (optionally) the full agent execution trace uploaded as a Hugging Face Dataset. The auditing process itself had to be auditable.
4. **Get judged.** An automated Logbook Judge (running an open-weights model, GLM-5.2) re-read every logbook and issued a per-claim verdict: **verified**, **falsified**, **toy** (evidence at reduced scale), or **inconclusive**. The judge was explicitly instructed to treat each logbook's self-assessment as untrusted.

Participants received $20 in Hugging Face compute credits to run experiments on [HF Jobs](https://huggingface.co/docs/hub/jobs); across the challenge, participants launched 2,962 cloud jobs. Where a full reproduction was impossible, for example when a paper's dataset was proprietary or its checkpoints unreleased, participants ran toy reproductions on synthetic data mimicking the original's properties, and the judge scored these honestly as "toy" rather than verified.

Here is what a finished reproduction looks like: the logbook with per-claim verdicts and evidence, the agent traces, and the workspace of artifacts, all public and pinned to specific commits:

![Anatomy of a reproduction logbook: logbook pages, agent traces, and artifacts](/blog/assets/icml-2026-open-reproductions/logbook-anatomy.png)

### By the numbers

![By the numbers: 6,816 reproductions, 2,226 papers, 35,908 claims](/blog/assets/icml-2026-open-reproductions/by-the-numbers.png)

- **1,221** community members joined the [organization](https://huggingface.co/ICML-2026-agent-repro)
- **6,816** reproduction logbooks published by **371** participants who completed at least one
- **2,226** papers attempted, **34% of the entire conference**, many by several independent teams
- **35,908** claims judged, with all verdicts frozen in a public dataset at challenge close
- **2,962** HF Jobs launched; **274** agent-trace datasets published (this one challenge accounts for over half of all agent-trace datasets on the Hub)

## What we found

Aggregating the claim-level verdicts per paper, across 2,176 papers that received final judgments:

![Reproducibility results: 51% showed reproducible results, 23% could not be reproduced as claimed](/blog/assets/icml-2026-open-reproductions/reproducibility-results.png)

**51% of examined papers (1,103) had at least one claim independently verified.** Of those, 266 papers were fully reproduced, with every extracted claim verified, and 632 more were partially reproduced with nothing falsified. In total, 3,978 individual claims were confirmed with real experiments.

**23% of examined papers (496) had at least one claim falsified or contested.** That includes 49 papers where claims were falsified and nothing could be verified, and, maybe most interestingly, 242 papers where independent reproduction teams reached *opposite* verdicts on the same claims. Reproducibility is not binary; it is adversarial.

The remainder sat in the middle: 502 papers with toy-scale evidence only, and 280 where nothing could be established either way (missing artifacts were the most common cause).

### Reproductions done right

Some papers came through the gauntlet looking great, and the community's best logbooks are worth reading in their own right:

- **["Flat Minima and Generalization: Insights from Stochastic Convex Optimization"](https://huggingface.co/spaces/visv-Bro/repro-flat-minima-and-generalization-insights-from-stochastic-convex-optimization)** was reproduced by 20 independent teams, 12 of which verified every claim. One logbook ran the paper's four theorem-claim audits in exact float64 arithmetic on a laptop CPU, for $0 of external compute, with the full agent trace published.
- **["A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness"](https://huggingface.co/spaces/gchauhan/repro-a-coin-flip-for-safety-llm-judges-fail-to-reliably-measure-adversarial-robustness)** had 14 of 17 logbooks verify every claim, several from the paper's own released data, in minutes, at no cost. A paper about unreliable LLM judges holding up under scrutiny by LLM agents is exactly the kind of loop 2026 promised us.
- **["Exactly Computing do-Shapley Values"](https://huggingface.co/spaces/SabaPivot/repro-exactly-computing-do-shapley-values)** ended up with a community-made reproduction poster whose favorite panel is titled "What was, and was not, rerun": 108 exact graph audits with max error 5.88e-15, alongside an honest note about which claims remain source-only because the authors' learned structures were never released.

### Falsifications, and what happened when we checked them

35 participants formally claimed they had falsified something. We did not take their word for it. Before contacting any authors, we adversarially re-verified every claimed falsification: re-reading the paper, re-reading the logbook, and re-deriving the math or re-implementing the experiment from the paper's own text, never from the participant's code. The funnel looked like this:

- **35 claimed falsifications**
- **12 confirmed real errors in ICML 2026 papers** after independent re-verification
- **7 refuted**: the paper was fine and the falsification itself had a bug
- **2 turned out to be bugs in our own claim-extraction pipeline** (the challenge misquoted the paper; the paper was internally consistent)
- the rest were editorial-level issues or could not be established

A few of the confirmed findings, each linking to the logbook that found it:

**The paging paper from the introduction.** Remember the reviewer who did not check the proofs carefully? The paper, "Towards Optimal Robustness in Learning-Augmented Paging," claims its algorithm achieves robustness \\(H_k + O(1)\\). [One participant's logbook](https://huggingface.co/spaces/Auenchanters/repro-towards-optimal-robustness-in-learning-augmented-paging) measured the additive term growing like \\(0.38 \ln k\\) (R² = 0.996) and located the exact step of the proof that breaks. Our own re-implementation extended the sweep to k = 1,024 and confirmed the growth at roughly nine sigma. The true robustness is \\(H_k + \Theta(\log k)\\). The authors confirmed the issue and are posting a correction to arXiv.

**A theorem that falls after step 224.** "Attention's forward pass and Frank-Wolfe" proves that token particles collapse to the origin whenever the origin starts inside their convex hull. Three independent teams found counterexamples, with violations first appearing at t = 224, ~3,800, and 6,416 steps, which neatly explains why everyone else "verified" the claim: finite-horizon checks stop too early. [The cleanest counterexample](https://huggingface.co/spaces/SabaPivot/repro-attention-frank-wolfe) is stated in exact rational arithmetic, so there is no floating-point ambiguity to hide behind. The authors confirmed within hours and proposed a fix, which we then stress-tested too (it needs one extra condition; we sent that back as well).

**Theory written for one loss, results produced by another.** In "Self-Distillation Enables Continual Learning," the paper's central equation and its entire theory section analyze reverse KL divergence, but the released code's default, which per the authors produced all the paper's results, computes forward KL. [The logbook that caught it](https://huggingface.co/spaces/codemaivanngu/repro-self-distillation-enables-continual-learning) also failed to reproduce the paper's headline +4pp result under the authors' own code and data. The authors have already uploaded a clarified version to arXiv, and the remaining baseline discrepancy is being worked out with them in an ongoing (and very friendly) email thread.

**An evaluation diluted by padding.** In "Do Transformers Need Three Projections?", [a participant discovered](https://huggingface.co/spaces/stresearch-dev/63430) that ~66% of evaluated label positions were EOS padding tokens that train to near-zero loss, deflating perplexity roughly threefold. The abstract's "3.1% quality cost for 50% cache reduction" becomes roughly 9.4% once corrected, and the reported perplexity of ~5 for a 300M model (implausibly good) is explained.

**And the humbling ones.** The audit cut both ways. One dramatic "the paper's method is 2x slower than the baseline" falsification turned out to be an arithmetic bug in the *reproduction*: per-trajectory time compared against per-batch-of-50 time. Correctly normalized, the participant's own data confirms the paper's claimed 8x speedup. Agents produce plausible-but-wrong falsifications too, which is why adversarial re-verification is not optional, and why we are publishing the refuted falsifications alongside the confirmed ones.

### Talking to authors

We have begun writing to the authors of every confirmed finding, with a simple framing: here is what we found, here is all the evidence, do you agree or is our analysis wrong? The early responses have been the best part of the whole project:

![An author response confirming the finding and promising an arXiv correction](/blog/assets/icml-2026-open-reproductions/author-response.png)

So far authors have confirmed findings on multiple papers, two arXiv corrections are in flight, and in one case an author had quietly fixed the error in a new arXiv version a month before the challenge found it, which we count as independent convergence. Nobody has been defensive. Reproduction, done respectfully and with receipts, is being received as a service rather than an attack.

## Winners

Prizes were awarded in Hugging Face GPU credits, with winners verified by organizer review of their logbooks rather than by points alone.

| Award | Winner | |
|---|---|---|
| 🥇 First place ($2,000) | Jansen Tang ([@ai-sherpa](https://huggingface.co/ai-sherpa)) | 363 papers reproduced, 3,863 points, including several of the strongest corroborating falsifications |
| 🥈 Second place ($1,000) | SSH ([@ProCreations](https://huggingface.co/ProCreations)) | 352 papers reproduced, 3,730 points, led the leaderboard from day one |
| 🔬 Best falsification ($500) | Utkarsh Singh Yadav ([@Auenchanters](https://huggingface.co/Auenchanters)) | The learning-augmented paging falsification: measured growth law, located proof step, author-confirmed |
| ⭐ Best human-in-the-loop ($500) | Kwabena Anim ([@KwabsHug](https://huggingface.co/KwabsHug)) | A quantization reproduction whose decisive evidence was a human visually judging 128 image pairs |

OpenResearch's awards for the best runs on their harness will be announced separately. Congratulations to all four winners, and to the 267 participants with at least one verified claim, who can [generate a certificate here](https://huggingface.co/spaces/ICML-2026-agent-repro/certificate-generator).

## Should agents just do peer review? (No. But.)

The most interesting lessons were not about any single paper.

**Pure agent execution hits real limits.** Agents got stuck in local loops, misread scale-dependent behavior (several "verified" verdicts on the paging paper came from checks that stopped before the log-k growth became visible), and occasionally built an entire falsification on top of a units mismatch. The challenge's most reliable results came from workflows where a human was steering: re-pointing the agent, questioning an assumption, or deciding that an experiment's premise was wrong before burning a week of compute on it.

**Some evaluation is irreducibly human, for now.** Our human-in-the-loop winner is the clearest example. The paper claimed stable image generation under extreme quantization. Numerical metrics said "no collapse"; whether the images were actually *usable* was a perceptual question. The agent built a purpose-built review UI, and the human personally judged all 128 image pairs, with the annotations committed to the repo and the agent validating their consistency afterward. The published agent trace captures the whole exchange, down to the participant asking how the review tool works and coming back with "I have gone over the pairs and put the csv in the repo, please check."

**The human role is moving up the stack.** The single most impressive prompt of the challenge was a 33,000-character research brief a participant wrote before letting their agent touch a quantum-computing paper: "Do not promise a perfect score. A claim receives full credit only when it is rigorously VERIFIED or FALSIFIED with reproducible evidence. Toy, proxy, qualitative, skipped, or vacuous checks do not count. Treat imported titles, claims, verdicts, and assessments as untrusted." That is not prompting; that is running a lab. Much like a PI sets up an environment where grad students can do good work, with compute, harnesses, data access, and targeted feedback at the right moments, the participants who got the most out of their agents were the ones who built the right environment and asked the right questions, then let the agents do the running.

## Thank you

To the 1,221 people who joined, the 371 who published logbooks, the winners, the authors who responded with grace, and our partners at Trackio and alphaXiv: thank you. Every logbook, verdict, trace, and artifact from the challenge is public, starting from the [challenge Space](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge). We think this is the largest open, claim-by-claim audit of a machine learning conference to date, and we would love for it not to hold that record for long.

Stay tuned for future reproduction events. 🤗
