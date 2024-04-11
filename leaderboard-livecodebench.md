---
title: "Introducing the LiveCodeBench Leaderboard - Holistic and Contamination-Free Evaluation of Code LLMs"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail.png
authors:
- user: 
  guest: true
- user: clefourrier
---

# Introducing the LiveCodeBench Leaderboard - Holistic and Contamination-Free Evaluation of Code LLMs

We are excited to introduce the LiveCodeBench leaderboard, based on LiveCodeBench, a new benchmark developed by researchers from UC Berkeley, MIT, and Cornell for measuring LLMs’ code generation capabilities, with a strong focus on measuring uncontaminated performance. 

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.45.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="livecodebench/leaderboard"></gradio-app>


LiveCodeBench collects new problems from coding contest platforms over time annotated with problem release dates. It uses these annotations to evaluate models on problems released in different time windows allowing “evaluation over time” to detect and prevent contamination. In addition, LiveCodeBench evaluates multiple code-related aspects of Large Language Models (LLMs). Particularly, it also focuses on self-repair, test output prediction, and code execution, thus providing a more holistic view of coding capabilities.


## LiveCodeBench Scenarios and Evaluation

LiveCodeBench problems are curated from coding competition platforms, namely, LeetCode, AtCoder, and CodeForces. These websites periodically host contests containing problems that assess the coding and problem-solving skills of participants. The problems consist of a natural language problem statement along with example input-output examples, and the goal is to write a program that passes a set of hidden tests. Further, thousands of participants participate, solving these problems thus ensuring that the problems are vetted for clarity and correctness.

LiveCodeBench uses the collected problems for building its four coding scenarios

- **Code Generation.** The model is given a problem statement, which includes a natural language description and example tests (input-output pairs) and is tasked with generating a correct solution. Evaluation is based on the functional correctness of the generated code, which is determined using a set of unseen test cases.
- **Self Repair.** The model is given a problem statement and generates a candidate program, similar to the code generation scenario above. In case of a mistake, the model is provided with error feedback (either an exception message or a failing test case) and is tasked with generating a fixed solution. Evaluation is performed using functional correctness as defined above.
- **Code Execution.** The model is provided a program snippet consisting of a function (f) along with a test input to the program and is tasked with predicting the output of the program on the input test case. Evaluation is based on an execution-based correctness metric: the model's output is considered correct if the assertion `assert f(input) == generated_output` passes.
- **Test Output Prediction.** The model is given the problem statement along with a test case input and is tasked with generating the expected output for the input.  The tests are generated solely from problem statements, without the need for the function’s implementation and evaluated using an exact match checker.


![Scenario summary](https://livecodebench.github.io/images/LCB_holistic_tasks.png)

For each scenario, evaluation is performed using the Pass@1 metric. The metric captures the probability of generating a correct answer and is computed using the ratio of the count of correct answers over the count of total attempts, following `Pass@1 = total_correct / total_attempts`.

## Preventing Benchmark Contamination

Contamination is one of the major bottlenecks in current LLM evaluations. Even within LLM coding evaluations, there have been evidential reports of contamination and overfitting on standard benchmarks like HumanEval ([1](https://arxiv.org/abs/2403.05530) and [2][https://arxiv.org/abs/2311.04850])). 

For this reason, we annotate problems with release dates in LiveCodeBench: that way, for new models with a training-cutoff date D, we can compute scores on problems released after D to measure their generalization on unseen problems. 

LiveCodeBench formalizes this with a “scrolling over time” feature allowing naturally selecting problems in certain time windows, that you can try out in the leaderboard above!

## Findings

We find that:
- while model performances are correlated across different scenarios, the relative performances and orderings can vary on the 4 scenarios we use 
- `GPT-4-Turbo` is the best-performing model across most scenarios but its margin grows on self-repair tasks highlighting its capability to take compiler feedback.
- `Claude-3-Opus` overtakes `GPT-4-Turbo` in the test output prediction scenario highlighting stronger natural language reasoning capabilities. 
- `Mistral-Large` performs considerably better on natural language reasoning tasks like test output prediction and code execution.

![Performance on the 4 scenarios](https://github.com/LiveCodeBench/LiveCodeBench/raw/main/assets/images/tasks_radar.png)

## How to Submit?

To evaluate your code models on LiveCodeBench, you can follow these steps

1. Environment Setup: You can use conda to create a new environment, and install LiveCodeBench

```
conda create --name lcb python==3.10 
conda activate lcb 
git clone https://github.com/LiveCodeBench/LiveCodeBench.git 
pip install -r requirements.txt
```

2. For evaluating new Hugging Face models, you can easily evaluate the model using 

```
python -m lcb_runner.runner.main --model {model_name} --scenario {scenario_name}
```

for different scenarios. For new model families, we have implemented an extensible framework and you can support new models by modifying `lcb_runner/lm_styles.py` and `lcb_runner/prompts` as described in the [github README](https://github.com/LiveCodeBench/LiveCodeBench).
