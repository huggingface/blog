---
title: "DeepMath: A Lightweight Math Reasoning Agent for LLMs"
thumbnail: /blog/assets/intel-deepmath/banner.png
authors:
- user: danf
  guest: true
  org: Intel
---

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-deepmath/deepmath-figure.jpg" style="width:600" alt="An LLM is using a calculator to answer questions." />

# DeepMath: A Lightweight Math Reasoning Agent for LLMs

*By Intel AI — Daniel Fleischer, Moshe Berchansky, Moshe Wasserblat*


Large language models (LLMs) have made impressive strides in reasoning tasks, yet mathematical problem-solving remains a challenge. Traditional "chain-of-thought" reasoning often produces verbose explanations and error-prone arithmetic. **DeepMath** tackles this by combining a small Python executor with a fine-tuned LLM, enabling concise, computation-driven reasoning.

## The Big Idea

DeepMath is built on **Qwen3-4B Thinking** and fine-tuned with **GRPO (Group Relative Policy Optimization)**. Instead of verbose text, the model emits **tiny Python snippets** for intermediate steps, runs them in a secure sandbox, and folds the results back into its reasoning, reducing errors and output length.

✅ No file I/O, no network calls, strict timeouts.

✅ Safe, deterministic, and auditable.

We evaluate DeepMath on four math datasets: **MATH500, AIME, HMMT, and HLE,** and show that:

- The math agent alone improves accuracy and reduces verbosity.

- GRPO training alone biases outputs toward brevity and correctness.

- Combining the agent with GRPO yields the largest gains.

👉 Code and evaluation scripts: <https://github.com/IntelLabs/DeepMath>
👉 Model: <https://huggingface.co/Intel/deepmath-v1>

## Why DeepMath?

LLMs often struggle with numeric precision and produce unnecessarily long reasoning chains. Two opportunities stand out:

1. **Offload deterministic computation** to a safe executor.

2. **Train models to prefer concise, computation-oriented traces** over verbose text.

DeepMath implements both. The model learns to generate short Python snippets, which are executed in a sandbox and reintegrated into the context. GRPO fine-tuning encourages this behavior by rewarding correctness and encouraging shoter outputs.

## How It Works

- Base model: [Qwen3-4B Thinking](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).
- Executor constraints: sandboxed environment, allow-list of imported modules, per-snippet timeout.
- Inference: based on [SmolAgents](https://github.com/huggingface/smolagents/), a math agent was created. vLLM is used as the inference engine.
- Training: based on the GRPO trainer in [TRL](https://github.com/huggingface/trl), we modified TRL's vLLM client and server to generate GRPO completions using our DeepMath agent.

<figure>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-deepmath/trl-grpo-vllm-deepmath.png" style="width:400" alt="Changes to vLLM client and server in TRL library." />
<figcaption><p>Figure 1: The vLLM client and server were modified to use the DeepMath agent in generating the candidates, while using the vLLM backend.</p></figcaption>
</figure>

- **Agent Interface:** During inference, the model can output normal tokens or special agent calls containing Python snippets.

- **Execution:** Snippets run in a sandboxed environment with strict safety constraints (no file I/O, no network, timeouts).

- **Design Goals:**

  - **Concision:** Replace multi-line textual calculations with short, focused snippets.

  - **Determinism & Safety:** Enforce strict execution limits.

  - **Interpretability:** Snippets are readable and auditable.

<figure>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-deepmath/output-example.png" style="width:700" alt="Output example: it contains a short python snippet as well as its output which is used in the reasoning process."/>
<figcaption><p>Figure 2: Output example where python code is generated, evaluated and the answer is inserted into the trace and used for context.</p></figcaption>
</figure>

## Training with GRPO

We fine-tune the model using **GRPO**, a reward-based optimization that balances:

- **Accuracy Reward:** +1 for correct answers.

- **Using code snippets:** +1 for generating code snippets, weighted 10:1 vs. the accuracy reward.

- **Length reduction:** shorter lengths are encouraged by limiting the GRPO completion candidates to 5k tokens.

- **Temperature Scheduling:** We implemented linear temperature scheduling (T=1.2 → T=0.7) to balance exploration and stability during training. This approach aims to enhance experimentation during the initial training phases, subsequently reducing the temperature as we refine our proficiency in the skill.

- **In-context Learning**: we include 4 solved examples where the trace contains agent calls and executor outputs, so the model learns the syntax and the call/response pattern.

- **Dataset**: we used [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset, the tool-usage subset. Note that GRPO only uses the <u>problem</u>, not the solution in the data. Choosing this dataset ensures problems benefit form tool use.

## Evaluation

We benchmarked DeepMath against baselines on four datasets. Metrics include:

- **majority@16** (robustness across samples).

- **Mean output length** (brevity).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/intel-deepmath/main-results.png" style="width:800" alt="Main results table."/>

**Key Insight:** DeepMath reduces output length by up to **66%** while improving accuracy on challenging datasets.

## Why It Matters

- **Accuracy:** Offloading computation reduces arithmetic errors.

- **Efficiency:** Shorter outputs mean faster inference and easier interpretability.

- **Safety:** Sandbox execution mitigates risks of running arbitrary code.

## Conclusion

DeepMath demonstrates a practical and lightweight way to combine a small executor with an LLM and to train the model to prefer short, computation-driven traces. Offloading deterministic computation reduces arithmetic and numerical errors and shortens traces, and GRPO fine-tuning further encourages concise, correct answers. The result is a more accurate and more interpretable math-solving agent without requiring a massive model or heavyweight external tools.

## Try It Yourself

Check out the GitHub repo and share your feedback! Contributions welcome. 🚀

<https://github.com/intel/DeepMath>.

## Citation

If you use DeepMath in your research, please cite:

```bibtex
@software{deepmath2025,
  author = {Fleischer, Daniel and Berchansky, Moshe and Wasserblat, Moshe},
  title = {DeepMath: A Lightweight Math Reasoning Agent for LLMs},
  year = {2025},
  publisher = {Intel AI Labs},
  url = {https://github.com/IntelLabs/DeepMath}
}
```

## Limitations & Future Work

- **Scope**: we focused on a small model and on mathematical reasoning.

- **Generalization**: evaluated on contest-style math; results may not transfer to open-ended mathematical creativity or formal proofs.

- Executing generated code is inherently risky. DeepMath uses strict sandboxing and resource limits, but any deployment should carefully manage attack surfaces and enforce rate limits.
