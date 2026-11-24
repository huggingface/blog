---
title: "Swarms, Flocks, Gangs: How to build, train, and evaluate multi-agent systems"
thumbnail: /blog/assets/training-swarms-part-1/thumbnail.png
authors:
- user: burtenshaw
---

# Swarms, Flocks, Gangs: How to build, train, and evaluate multi-agent systems

*Part 1: Understanding swarms and training to communicate*

This is an aspirational blog series for aspirational readers. We (us, the community, you) want to build our own agent swarms because they're exciting, challenging, and increasingly capable. Furthermore, it would be extremely cool if we could train our own agents to work in a swarm. And exponentially cool if those swarms used small and affordable models.

That's what we'll do in this series! Each blog post will include some context on building swarms, alongside reproducible examples that you can run at home for less than a Matcha Latte.

Multi-Agent Systems are hard to get working. Adding agents changes the agent's context, the budget, and in turn the complexity of the whole system. Mixing agents' context together through communication is not an inherent performance boost compared to a single agent. The model needs to learn to communicate through training. How do we define what models should learn? And which model weights do we update when we're dealing with many agents?

In this series, we'll work through those problems with an educational overview and practical examples. Each post will introduce a question, build a small system, and examine what happens and how you can reproduce it. This first example starts with four agents, a shared language model, and a simple verifiable task. It's ideal to implement yourself, to tinker with, and to learn from.

First, we'll start off with some context on swarms.

## The rise of the swarms

We've heard a lot about swarms over the past few months, both good and bad. In September this year, OpenAI announced that a swarm of roughly 10,000 agents had found a singularity in the 3D Navier-Stokes equations. Also, multi-agent systems solved a number of Millennium Prize and long-standing math and physics challenges.

But swarms don't always work on the problems we want them to. METR's investigation of the Hugging Face incident found that around 1,200 OpenAI agents, which were supposed to be isolated, had found a way to exchange over 70,000 messages and files. Around 700 joined an attack on Hugging Face while looking for ways to fool their benchmark's scorer, without instruction to do so.

A big swarm isn't, by itself, evidence that cooperation helped. [Noam Brown estimates](https://www.dwarkesh.com/p/noam-brown) that less than 10% of the Navier-Stokes result came from multi-agent cooperation. In Anthropic's vulnerability-finding experiment, coordinated agents found more bugs than independent agents, but also searched more code and used more tokens. When the comparison was restricted to the same directories, the results per million tokens were about the same.

## Did cooperation help, or did we just spend more?

Start with the simplest case. If four agents each try a task independently, the chance that at least one gets it right is higher than with a single attempt, as long as we can tell which answer is correct. That makes a useful baseline for a multi-agent system, but it doesn't show that the agents helped each other. They may simply have had more tries.

Four attempts also cost four times as much. So the fairer comparison is at an equal budget: if each of four agents gets a quarter of the total tokens, can they still beat one agent that spends all of them? We can only find out by testing, because the answer depends on the task, the model and how the work is divided.

> [!NOTE]
> This series keeps asking one question: when agents communicate, does it actually help, or are we just spending more compute?

We understand that question through this simple expectation.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/buckets/huggingface/training-swarms-blog-assets/resolve/part-1/expected-scaling.png" alt="Sketch of expected performance against agent count at matched agents and tokens: communicating agents sit above non-communicating agents" width="70%">
  <figcaption>Figure 1. Sketch of the expectation we want to test: compare communicating and independent agents at matched budgets.</figcaption>
</figure>

We'll treat agent count as a scaling dimension, alongside tokens and model size. Sometimes adding agents will help. Sometimes coordination will consume the budget we hoped to save. The point is to measure both. We won't start with 10,000 agents. We'll start with a small team on a task that we can check easily.

## Multi-agent Systems and Reinforcement Learning

Multi-agent reinforcement learning has studied cooperation and competition long before language models were called swarms.

A single agent's task is a Markov decision process: observe, act, get a reward. In a team, everyone shares the reward, but each agent sees only part of the world. This is a decentralized partially observable Markov decision process: teammates, but no one sees the whole picture.

This creates three practical problems. First, teammates keep changing while each agent learns. Second, credit assignment: when the team succeeds, which actions deserve the credit? A winning team can include weak actions, and we may end up rewarding them. Third, a message changes what other agents observe, so communication is part of the policy we're training.

A common solution is centralized training with decentralized execution: training uses information from the whole team, but each agent acts only on its own observations. Our first example follows this pattern with a shared team reward.

Agents can also be organized in different ways. Two useful starting points are:

- **A hierarchy.** A lead agent assigns work to workers and collects their results. Anthropic's Research system works this way, with a separate context for each worker. Kimi K2.5's PARL trains only the orchestrator and keeps the subagents frozen, so training a multi-agent system doesn't mean updating every component. See [Anthropic's system description](https://www.anthropic.com/research/multi-agent-research-system) and [Kimi's technical blog](https://www.kimi.com/en/blog/kimi-k2-5).

- **A distributed team.** Peers send each other messages and decide when to read their inboxes. No lead agent assigns the actions. This is the setup we'll use here, following the information-aggregation task in [s³: Simple Scaling of Swarms via Cooperative Multi-agent RL](https://arxiv.org/abs/2611.08615).

## Our first experiment: four agents ordering numbers

We'll reproduce the information-sharing task from [Simple Scaling of Swarms](https://arxiv.org/abs/2611.08615) because it's interpretable and straightforward to implement. Four agents share a small language model and learn to pool what each of them knows.

Each agent receives one integer from 1 to 99. It knows its own assignment and its teammates' names. It does not initially know the other agents' numbers. For example, four private assignments might be "five", "twenty-six", "six" and "seventy-eight" and a correct final answer is:

```text
#### [5, 6, 26, 78]
```

The team succeeds when **at least one agent submits the complete, correctly sorted list**. One correct answer is enough; success does not require all four agents to answer correctly. An agent cannot reliably infer the other private assignments without information from its teammates.

The main communication tools are `message(to, text)`, `read_messages()`, and `check_notifications()`. A notification tells an agent that unread messages exist; reading its inbox makes their contents available. Agents can send direct messages or broadcast to everyone.

We defined four agent systems using **one shared Qwen3-0.6B policy**, with separate contexts and private inputs. We are not training four different models. The prompt explains the task and the tools, including the need to share assignments.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/buckets/huggingface/training-swarms-blog-assets/resolve/part-1/task-and-training-loop.png" alt="Four agents share one policy with separate contexts and private numbers, exchange messages, submit a sorted list, and train from a common team reward" width="80%">
  <figcaption>Figure 2. The task and training loop. The private numbers illustrate a possible successful episode; this graphic is not a recorded conversation or a claim about the learned protocol at update 75.</figcaption>
</figure>

### A team reward and eight attempts at each problem

A successful team receives a reward reduced by the tokens it uses. A failed team receives zero:

```text
team allowance = 4 × (8,192 work tokens + 1,024 final-answer tokens)
               = 36,864 tokens

reward = success × (1 − 0.25 × min(1, generated team tokens / 36,864))
```

For illustration, a correct team using half its allowance receives 0.875. There is no separate reward for sending messages. The incentive is to solve the task while using fewer generated tokens.

For every training problem, we collect eight independent team attempts. Their rewards are normalized within that group. An attempt that does better than its group average gets a positive advantage; one that does worse gets a negative advantage. Every agent in the same team receives that team's advantage, including an agent whose own final answer was wrong.

In TRL, each agent's sampled tokens form a training row. Its incoming messages and tool results provide context. The trainer batches the four rows from one team together, so longer conversations do not give that team more weight.

> [!NOTE]
> The walkthrough below uses TRL's 128-team updates. The next section reports the original experiment, which used 256 teams per update.

### Our Results

Over 75 steps, training success rose from **6.6% to 82.3%**, comparing the first and last five updates. At checkpoint 75, the four-agent team solved **26 of 32 held-out problems (81.25%)**. [The paper](https://arxiv.org/abs/2611.08615) reports 93.2% four-agent success after training.

The team also used **37.4% fewer tokens per attempt**, falling from about 12k to 8k. This improvement accompanied more active information sharing: broadcasts increased from 2.38 to 5.73 per team, even as total token use fell.

<figure class="image text-center">
  <iframe src="https://burtenshaw-training-swarms-figures.static.hf.space/training-success.html" title="Interactive chart: team success over 75 training updates, with held-out checks of 1/32, 7/32, 17/32 and 26/32 at updates 1, 25, 50 and 75" width="100%" height="410" frameborder="0" scrolling="no" loading="lazy"></iframe>
  <figcaption>Figure 3. The blue curve describes training batches of 256 team attempts, collected before each update. Orange points are held-out checks after the indicated update.</figcaption>
</figure>

The team also used fewer tokens later in training. Comparing the first five updates with updates 71–75:

| **Training metric** | **Updates 1–5, mean** | **Updates 71–75, mean** |
|---|---:|---:|
| Success | 6.56% | 82.34% |
| Generated tokens per team | 12,826.9 | 8,026.0 |
| Messages per team | 3.48 | 6.28 |
| Broadcasts per team | 2.38 | 5.73 |
| Teams with one broadcast per agent | 4.06% | 18.20% |

That is a **37.4% reduction in generated tokens per training team** between those windows. At batch 75 itself, the mean was 7,624 tokens per team. These are training measurements; we did not retain held-out token-efficiency measurements for the trained checkpoint.

Notice that fewer tokens did not mean fewer messages. More agents began participating but broadcasting became more common, rather than direct messaging. Generated tokens include reasoning and tool calls as well as message bodies, so message count alone does not tell us how much text the team produced.

<figure class="image text-center">
  <iframe src="https://burtenshaw-training-swarms-figures.static.hf.space/tokens-and-communication.html" title="Interactive chart: generated tokens per team fall to 7,624 at batch 75 while broadcasting grows; teams with one broadcast per agent remain a minority" width="100%" height="620" frameborder="0" scrolling="no" loading="lazy"></iframe>
  <figcaption>Figure 4. Top: training token use and a trailing five-update mean. Bottom: mutually exclusive communication classes, based on sent-message events. A team is classified first by whether some agent never sends, then whether any direct message occurs, then whether every agent sends exactly one broadcast; the remaining teams repeat broadcasts.</figcaption>
</figure>

We used one H200 for the trainer and seven H200s running vLLM rollout engines. We kept each worker's eight attempts together. Here are the infra details:

| **Setting**                   | **Recorded experiment**        |
|---|---|
| Shared policy                 | Qwen3-0.6B                     |
| Logical agents per team       | 4                              |
| Problems per update           | 32                             |
| Team attempts per problem     | 8                              |
| Work / final tokens per agent | 8,192 / 1,024                  |
| Learning rate / temperature   | 5e-7 / 1.0                     |
| Training problem pool         | 1,024 generated problems       |
| Reporting endpoint            | Preserved checkpoint 75        |
| Execution                     | 1 trainer GPU + 7 rollout GPUs |

## Practical example: Train your own swarm

The figures above describe our original 75-update experiment. In this walkthrough we'll run [TRL's swarm example](https://github.com/huggingface/trl/tree/main/examples/async_grpo_s3_aggregation) on Hugging Face Jobs. One H200 serves the shared Qwen3-0.6B policy with vLLM and a second H200 trains it.

### 1. Create a private bucket for the run

Sign in with a Hugging Face account that can run Jobs and create buckets. Replace `YOUR_USERNAME` with your Hub username.

> [!TIP]
> Use a fresh bucket for each run so earlier measurements stay available.

```bash
uv tool install 'huggingface_hub==2.4.0'
hf auth login
BUCKET="YOUR_USERNAME/swarm-trl-$(date -u +%Y%m%d-%H%M%S)"
hf buckets create "$BUCKET" --private
```

### 2. Run the TRL example on Hugging Face Jobs

Save the script below as `train-swarm.sh` in your current directory.

<details>
<summary><code>train-swarm.sh</code></summary>

```bash
set -eu
git clone --depth 1 https://github.com/huggingface/trl /tmp/trl
cd /tmp/trl
example=examples/async_grpo_s3_aggregation
if [ ! -f "$example/async_grpo_s3_aggregation.py" ]; then
  git fetch --depth 1 origin refs/pull/8424/head
  git checkout --detach FETCH_HEAD
fi
git rev-parse HEAD > /results/trl-revision.txt
python -m pip install -e ".[vllm,kernels]" "vllm==0.32.0" datasets httpx trackio

base=Qwen/Qwen3-0.6B
url=http://127.0.0.1:8000
export TRACKIO_DIR=/tmp/trackio
serve() {
  CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve "$@" \
    --host 127.0.0.1 --max-model-len 40960 --max-num-seqs 512 --enable-prefix-caching &
  server=$!
  until curl -fs "$url/health" >/dev/null; do kill -0 "$server"; sleep 5; done
}
evaluate() {
  python "$example/evaluate.py" --url "$url" --model "$1" --name "$2" \
    --episodes 192 --concurrency 64 --out "/results/$2.jsonl"
}

serve "$base" --logprobs-mode processed_logprobs --weight-transfer-config "{\"backend\":\"nccl\"}"
evaluate "$base" base
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --mixed_precision bf16 \
  "$example/async_grpo_s3_aggregation.py" --vllm-url "$url" \
  --updates 50 --teams-per-update 128 --concurrent-teams 128 \
  --report-to trackio --output-dir /tmp/swarm
cp -r "$TRACKIO_DIR" /tmp/swarm/checkpoint-50 /results/
kill "$server"
wait "$server" || true
serve /tmp/swarm/checkpoint-50 --logprobs-mode processed_logprobs
evaluate /tmp/swarm/checkpoint-50 trained
```

</details>

The command below sends its contents to the job. Inside the job, vLLM serves the shared policy on GPU 0. GPU 1 runs TRL's `async_grpo_s3_aggregation.py` for 50 updates with Trackio. TRL's `evaluate.py` scores the base model and checkpoint 50 on the same 192 held-out problems. The job saves both scores, the checkpoint, Trackio data and the exact Git commit to your private bucket.

```bash
hf jobs run --detach --flavor h200x2 --timeout 12h \
  --volume "hf://buckets/$BUCKET:/results" \
  -- vllm/vllm-openai:v0.32.0 bash -c "$(cat train-swarm.sh)"
```

One TRL rollout contains a whole team conversation. The trainer compares eight attempts at the same problem and gives all four agents in an attempt the same advantage. Each update uses 128 team attempts. The `--report-to trackio` flag records training metrics at every update.

### 3. Read the results

Once the job completes, download the two evaluation files and the Trackio data. In the dashboard, compare `team/success`, `team/messages` and `team/tokens` over training. The job logs also give the held-out success rate for the base model and checkpoint 50. Each line of `base.jsonl` or `trained.jsonl` records one of those evaluation attempts.

```bash
hf buckets cp "hf://buckets/$BUCKET/base.jsonl" ./base.jsonl
hf buckets cp "hf://buckets/$BUCKET/trained.jsonl" ./trained.jsonl
hf buckets sync "hf://buckets/$BUCKET/trackio" ./swarm-trackio
TRACKIO_DIR="$PWD/swarm-trackio" uvx trackio show
```

## What we'll try next

This first example gives us a small environment, a shared reward and a policy that improves at pooling private information. It also shows why success, token use and communication structure need to be read together. At 75 updates, the team is solving most checked problems, but it has not converged to the cleanest communication pattern.

In the coming examples, we'll take a step back from training and look at how we evaluate and analyze swarms.

For now, we have something concrete to build on: four small agents, one trainable policy, a checkable task, and a reproducible way to examine what the team learned.
