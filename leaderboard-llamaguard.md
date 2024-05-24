---
title: "CyberSecEval 2 - A Comprehensive Evaluation Framework for Cybersecurity Risks and Capabilities of Large Language Models"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_llamaguard.png
authors:
- user: r34p3r1321
  guest: true
  org: Facebook
- user: csahana95
  guest: true
  org: Facebook
- user: liyueam10
  guest: true
  org: Facebook
- user: cynikolai
  guest: true
  org: Facebook
- user: dwjsong
  guest: true
  org: Facebook
- user: simonwan
  guest: true
  org: Facebook
- user: fa7pdn
  guest: true
  org: Facebook
- user: is-eqv
  guest: true
  org: Facebook
- user: yaohway
  guest: true
  org: Facebook
- user: dhavalkapil
  guest: true
  org: Facebook
- user: dmolnar
  guest: true
  org: Facebook
- user: spencerwmeta
  guest: true
  org: Facebook
- user: jdsaxe
  guest: true
  org: Facebook
- user: vontimitta
  guest: true
  org: Facebook
- user: carljparker
  guest: true
  org: Facebook
- user: clefourrier
---


# CyberSecEval 2 - A Comprehensive Evaluation Framework for Cybersecurity Risks and Capabilities of Large Language Models

With the speed at which the generative AI space is moving, we believe an open approach is an important way to bring the ecosystem together and mitigate  potential risks of Large Language Models (LLMs).  Last year, Meta released an initial suite of open tools and evaluations aimed at facilitating responsible development with open generative AI models. As LLMs become increasingly integrated as coding assistants, they introduce novel cybersecurity vulnerabilities that must be addressed. To tackle this challenge, comprehensive benchmarks are essential for evaluating the cybersecurity safety of LLMs. This is where [CyberSecEval 2](https://arxiv.org/pdf/2404.13161),  which assesses an LLM's susceptibility to code interpreter abuse, offensive cybersecurity capabilities, and prompt injection attacks, comes into play to provide a more comprehensive evaluation of LLM cybersecurity risks. You can view the [CyberSecEval 2 leaderboard](https://huggingface.co/spaces/facebook/CyberSecEval) here.

## Benchmarks 
CyberSecEval 2 benchmarks help evaluate LLMs’ propensity to generate insecure code and comply with requests to aid cyber attackers:
- **Testing for generation of insecure coding practices**: Insecure coding-practice tests measure how often an LLM suggests risky security weaknesses in both autocomplete and instruction contexts as defined in the [industry-standard insecure coding practice taxonomy of the Common Weakness Enumeration](https://cwe.mitre.org/). We report the code test pass rates.
- **Testing for susceptibility to prompt injection**: Prompt injection attacks of LLM-based applications are attempts to cause the LLM to behave in undesirable ways. The [prompt injection tests](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/datasets/mitre) evaluate the ability of the LLM to recognize which part of an input is untrusted and its level of resilience against common prompt injection techniques. We report how frequently the model complies with attacks.
- **Testing for compliance with requests to help with cyber attacks**: Tests to measure the false rejection rate of confusingly benign prompts. These [prompts](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/datasets/frr) are similar to the cyber attack compliance tests in that they cover a wide variety of topics including cyberdefense, but they are explicitly benign—even if they may appear malicious. We report the tradeoff between false refusals (refusing to assist in legitimate cyber related activities) and violation rate (agreeing to assist in offensive cyber attacks).
- **Testing propensity to abuse code interpreters**: Code interpreters allow LLMs to run code in a sandboxed environment. This set of [prompts](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/datasets/interpreter) tries to manipulate an LLM into executing malicious code to either gain access to the system that runs the LLM, gather sensitive information about the system, craft and execute social engineering attacks, or gather information about the external infrastructure of the host environment. We report the frequency of model compliance to attacks.
- **Testing automated offensive cybersecurity capabilities**: This suite consists of [capture-the-flag style security test cases](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/datasets/canary_exploit) that simulate program exploitation. We use an LLM as a security tool to determine whether it can reach a specific point in the program where a security issue has been intentionally inserted. In some of these tests we explicitly check if the tool can execute basic exploits such as SQL injections and buffer overflows. We report the model’s percentage of completion.

All the code is open source, and we hope the community will use it to measure and enhance the cybersecurity safety properties of LLMs.

You can read more about all the benchmarks [here](https://huggingface.co/spaces/facebook/CyberSecEval).

## Key Insights

Our latest evaluation of state-of-the-art Large Language Models (LLMs) using CyberSecEval 2 reveals both progress and ongoing challenges in addressing cybersecurity risks.

### Industry Improvement

Since the first version of the benchmark, published in December 2023,  the average LLM compliance rate with requests to assist in cyber attacks has decreased from 52% to 28%, indicating that the industry is becoming more aware of this issue and taking steps towards improvement.

### Model Comparison

We found models without code specialization tend to have lower non-compliance rates compared to those that are code-specialized. However, the gap between these models has narrowed, suggesting that code-specialized models are catching up in terms of security.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-on-the-hub/llamaguard.png" alt="heatmap of compared results"/>


### Prompt Injection Risks

Our prompt injection tests reveal that conditioning LLMs against such attacks remains an unsolved problem, posing a significant security risk for applications built using these models. Developers should not assume that LLMs can be trusted to follow system prompts safely in the face of adversarial inputs.

### Code Exploitation Limitations

Our code exploitation tests suggest that while models with high general coding capability perform better, LLMs still have a long way to go before being able to reliably solve end-to-end exploit challenges. This indicates that LLMs are unlikely to disrupt cyber exploitation attacks in their current state.

### Interpreter Abuse Risks

Our interpreter abuse tests highlight the vulnerability of LLMs to manipulation, allowing them to perform abusive actions inside a code interpreter. This underscores the need for additional guardrails and detection mechanisms to prevent interpreter abuse. 

## How to contribute?

We’d love for the community to contribute to our benchmark, and there are several things you can do if interested! 

To run the CyberSecEval 2 benchmarks on your model, you can follow the instructions [here](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks). Feel free to send us the outputs so we can add your model to the [leaderboard](https://huggingface.co/spaces/facebook/CyberSecEval)! 

If you have ideas to improve the CyberSecEval 2 benchmarks, you can contribute to it directly by following the instructions [here](https://github.com/meta-llama/PurpleLlama/blob/main/CONTRIBUTING.md).

## Other Resources
- [Meta’s Trust & Safety](https://llama.meta.com/trust-and-safety/)
- [Github Repository](https://github.com/meta-llama/PurpleLlama)
- [Examples of using Trust & Safety tools](https://github.com/meta-llama/llama-recipes/tree/main/recipes/responsible_ai)
