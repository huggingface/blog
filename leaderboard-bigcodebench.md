---
title: "BigCodeBench: Benchmarking Large Language Models on Solving Practical and Challenging Programming Tasks"
thumbnail: 
authors:
- user: terryyz
  guest: true
  org: bigcode
- user: ganler
  guest: true
  org: bigcode
- user: zijwang
  guest: true
  org: bigcode
- user: SivilTaram
  guest: true
- user: huybery
  guest: true
- user: Muennighoff
  guest: true
  org: bigcode
- user: dpfried
  guest: true
  org: bigcode
- user: harmdevries
  guest: true
  org: bigcode
- user: lvwerra
- user: clefourrier
---

# Introducing the BigCodeBench Leaderboard: Benchmarking Large Language Models on Solving Practical and Challenging Programming Tasks

[HumanEval](https://github.com/openai/human-eval) is a widely used benchmark for evaluating large language models (LLMs) on code generation tasks. One main reason is that it is easy to evaluate a condense function-level code snippet. There are growing concerns about the effectiveness of HumanEval in evaluating programming capabilities of LLMs. The main concern is that the tasks in HumanEval are too simple and may not be representative of real-world programming tasks.

While there have been some efforts to address this issue, they are either domain-specific, solution-specific, or agent-centric (sorry [DS-1000](https://github.com/HKUNLP/DS-1000), [ODEX](https://github.com/zorazrw/odex), and [SWE-bench](https://github.com/princeton-nlp/SWE-bench) 💔).
The community still lacks an easy-to-use benchmark that can fundamentally evaluate the programming capabilities of LLMs.

We are excited to announce the release of BigCodeBench, which evaluates LLMs on solving practical and challenging programming tasks. Specifically, BigCodeBench contain 1,140 function-level tasks to challenge LLMs to follow instruction and compose multiple function calls as tools from 139 libraries. To evaluate LLMs rigorously, each programming task encompasses 5.6 test cases with an average branch coverage of 99%.

Ready to deep dive into BigCodeBench? Let's get started! 🚀

## What do the tasks in BigCodeBench look like? 🕵️‍♂️

<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/tease.svg?raw=true" alt="png" style="display: block; margin-left: auto; margin-right: auto;">

```python
# We elaborate the above task with some test cases:

# Requirements SetUp
import unittest
from unittest.mock import patch
import http.client
import ssl
import socket

# Start the test
class TestCases(unittest.TestCase):

    # Mock the successful connection and assess the response content
    @patch('http.client.HTTPSConnection')
    def test_response_content(self, mock_conn):
        """ Test the content of the response. """
        mock_conn.return_value.getresponse.return_value.read.return_value = b'Expected Content'
        result = task_func('www.example.com', 443, '/content/path')
        self.assertEqual(result, 'Expected Content')

    # Mock the failed connection and assess the error handling
    @patch('socket.create_connection')
    @patch('http.client.HTTPSConnection')
    def test_ssl_handshake_error_handling(self, mock_conn, mock_socket):
        """ Test handling of SSL handshake errors. """
        mock_socket.side_effect = ssl.SSLError('SSL handshake failed')
        with self.assertRaises(ssl.SSLError):
            task_func('badssl.com', 443, '/test/path')

    # More test cases...
```
BigCodeBench contains complex and user-oriented instructions (e.g., clear functionality descriptions, input and output formats, error handling, and verified interactive examples) for each task. We do not describe the step-by-step instructions for the tasks, as we believe that _capable LLMs should be able to understand task descriptions from the user's perspective and solve the tasks in an open-ended manner_. To examine whether LLMs can follow implementation instructions, we verify specific features with test cases.

Another feature is that the tasks in BigCodeBench are designed to utilize diverse function calls (or APIs) from popular libraries. We do not restrict the specific function calls that the LLMs can use, as we believe that the capable LLMs should be able to _choose the appropriate function calls as tools and flexibly combine them to solve the tasks_. Instead of checking the specific function calls, we design the test cases as test harnesses to examine the expected program behaviors during the runtime.

To assess the LLMs' performance, we use Pass@1 with greedy decoding, which measures the percentage of tasks that the model can solve correctly with the first generated code snippet via curated test cases. This setup aligns with other representative benchmarks like [HumanEval](https://github.com/openai/human-eval) and [MBPP](https://github.com/google-research/google-research/tree/master/mbpp). As we notice that some LLMs are lazy to repeat the long code prompts (as discussed in the next section), we add the missing setup (e.g., import statements and global constants) back during the Pass@1 evaluation.

<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/depth-breadth.svg?raw=true" alt="png" style="display: block; margin-left: auto; margin-right: auto; width: 50%;">

To better understand the implementation complexity and tool-use diversity, we compare the tasks in BigCodeBench with those in representative benchmarks, including [APPS](https://github.com/hendrycks/apps), [DS-1000](https://github.com/HKUNLP/DS-1000), [ODEX](https://github.com/zorazrw/odex), [APIBench](https://github.com/ShishirPatil/gorilla/tree/main/data/apibench), [MBPP](https://github.com/google-research/google-research/tree/master/mbpp),  [NumpyEval](https://github.com/microsoft/PyCodeGPT/tree/main/cert/pandas-numpy-eval), [PandasEval](https://github.com/microsoft/PyCodeGPT/tree/main/cert/pandas-numpy-eval), [HumanEval](https://github.com/openai/human-eval), and [TorchDataEval](https://github.com/microsoft/PyCodeGPT/tree/main/apicoder/private-eval). We find that the tasks in BigCodeBench require more complex reasoning and problem-solving skills to implement comprehensive functionalities.

<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/bigcodebench_prompt.svg?raw=true" alt="png" style="display: block; margin-left: auto; margin-right: auto;">

As we can see from the task figure, the main target scenario is code completion (denoted as BigCodeBench-Complete), where LLMs are required to finish the implementation of the function based on the verbose instruction inside the docstring. However, considering the downstream applications such as multi-turn dialogue, users may tend to describe the requirements in a more conversational and less verbose manner. Instruction-tuned LLMs here come in handy, as they are trained to follow the natural-language-oriented instructions and generate code snippets accordingly. To test if the models are really capable enough to understand human intents to code, we create BigCodeBench-Instruct, a more challenging variant of BigCodeBench to particularly evaluate instruction-tuned LLMs.

## Where does the task come from? 🤔

<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/construct_pipeline.svg?raw=true" alt="png" style="display: block; margin-left: auto; margin-right: auto;">


We guarantee the quality of the tasks in BigCodeBench by using a systematic "Human-LLM collaboration process". 
We use [ODEX](https://github.com/zorazrw/odex) as "seed dataset": it contains short but realistic human intents and corresponding Python one-liners from Stack Overflow, and we utilize GPT-4 to enrich the one-liners into comprehensive function-level tasks. Then, 20 human experts (where most of them hold PhD degrees and 5+ years of Python programming experience) voluntarily ground GPT-4 in an execution-based sandbox, and continually instruct it to refactor the synthesized tasks and add test cases. Finally, the tasks and test cases are examined in the local environment, the tasks are pre-evaluated on other LLMs, and cross-checked with 7 additional human experts to ensure their quality. 

To assert overall quality, the authors sampled tasks for 11 human experts to solve, and the average human performance was 97%.

## How well do LLMs perform on BigCodeBench? 📊

We host the BigCodeBench leaderboard on both [Hugging Face Space](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) and [GitHub Pages](https://bigcode-bench.github.io/). Here, we use the Hugging Face leaderboard as an example.

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.36.1/gradio.js"
></script>

<gradio-app theme_mode="light" space="bigcode/bigcodebench-leaderboard"></gradio-app>

Interestingly, we observe that instruction-tuned LLMs like GPT-4 can omit the essential import statements of the given long prompts in BigCodeBench-Complete, which can lead to task failure due to the lack of proper module and constant definitions. Such behaviors are denoted as "model laziness" in long-context interactions, as [discussed in the community](https://community.openai.com/t/why-i-think-gpt-is-now-lazy/534332). <u>Compared to human performance, we find that LLMs perform significantly lower on BigCodeBench-Complete, and even lower on BigCodeBench-Instruct, with the best model (GPT-4o) achieving the calibrated Pass@1 of 61.1% and 51.1%, respectively.</u> In addition, we observe a big performance gap between the close LLMs and open ones.

While benchmark-level Pass@1 is a good metric to show the overall performance of the models, it is not informative enough to do the pairwise comparison between the models. Inspired by [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/), we employ Elo rating to rank the models on BigCodeBench-Complete. The Elo rating is a widely used method in chess to rank the players based on their performance in the games. We adapt the Elo rating to the programming tasks by treating each task as a game and each model as a player. The Elo rating is updated after each game based on the outcome of the game and the expected outcome of the game. We use the calibrated Pass@1 in each task as the outcome of the game and exclude the tie results. We set the initial Elo rating of the models to 1000, fit it using maximum likelihood estimation, and bootstrap with 500 iterations to get the final scores. <u>We see that GPT-4o outperforms the other models by a large margin, and LLMs like Gemini-1.5-Pro, GPT-4-Turbo, GPT-4, Claude-3-Opus are in the second tier.</u>

To help the community better understand how the models perform on each task, we provide the progress of solve rate, measured by calibrated Pass@1. On BigCodeBench-Complete, 151 tasks have not been solved by by all 71 models and 6 have been completely solved. For BigCodeBench-Instruct, there are 281 tasks remain unsolved and 14 fully solved by all models. As there is a significant number of unsolved tasks and only small number of fully solved tasks, we deem BigCodeBench as a challenging benchmark for LLMs.

## Great! So how can I evaluate my model on BigCodeBench? 🛠️

We make BigCodeBench easily accessible to the community by providing a simple and user-friendly evaluation framework, which can be downloaded via [PyPI](https://pydigger.com/pypi/bigcodebench). The prototype of the evaluation framework is based on [EvalPlus](https://github.com/evalplus/evalplus) for HumanEval+ and MBPP+ benchmarks. Different from EvalPlus, we take great effort to build a less bounded and more flexible execution environment  to support tasks with diverse library dependencies, and adapt it for `unittest` in the test harness of BigCodeBench.

To facilitate the evaluation, we provide pre-built Docker images for _code generation_ with [cuda-11.8.0](https://hub.docker.com/r/terryzho/bigcodebench-generate-cu11) and [cuda-12.1.1](https://hub.docker.com/r/terryzho/bigcodebench-generate-cu12), and [code execution](https://hub.docker.com/r/terryzho/bigcodebench-evaluate). Check out our [GitHub repository](https://github.com/bigcode-project/bigcodebench) to find more details on how to use the evaluation framework.

### Setup
```bash
# Install to use bigcodebench.evaluate
pip install bigcodebench --upgrade
# If you want to use the evaluate locally, you need to install the requirements
pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt

# Install to use bigcodebench.generate
# You are strongly recommended to install the [generate] dependencies in a separate environment
pip install bigcodebench[generate] --upgrade
```

### Code Generation
You are suggested to use `flash-attn` for generating code samples.
```bash
pip install -U flash-attn
```

To generate code samples from a model, you can use the following command:
```bash
bigcodebench.generate \
    --model [model_name] \
    --subset [complete|instruct] \
    --greedy \
    --bs [bs] \
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google] \
    --tp [gpu_number] \
    [--trust_remote_code] \
    [--base_url [base_url]]
```

The generated code samples will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples].jsonl`.

### Code Post-processing

LLM-generated text may not be compilable code for including natural language lines or incomplete extra code.
We provide a tool namely `bigcodebench.sanitize` to clean up the code:

```bash
# 💡 If you want to storing calibrated codes in jsonl:
bigcodebench.sanitize --samples samples.jsonl --calibrate
# Sanitized code will be produced to `samples-sanitized-calibrated.jsonl`

# 💡 If you do without calibration:
bigcodebench.sanitize --samples samples.jsonl
# Sanitized code will be produced to `samples-sanitized.jsonl`

# 💡 If you are storing codes in directories:
bigcodebench.sanitize --samples /path/to/vicuna-[??]b_temp_[??]
# Sanitized code will be produced to `/path/to/vicuna-[??]b_temp_[??]-sanitized`
```

### Code Evaluation

You are strongly recommended to use a sandbox such as [docker](https://docs.docker.com/get-docker/):

```bash
# mount the current directory to the container
docker run -v $(pwd):/app terryzho/bigcodebench-evaluate:latest --subset [complete|instruct] --samples samples.jsonl
```

## What's next?
We share a long-term roadmap to address the limitations of BigCodeBench, and
sustainably build with the community. We believe that [program-aided language models](https://arxiv.org/abs/2211.10435) for task
completion and reasoning provide the possibility towards artificial general intelligence. Our goal is to provide the community with the most open, reliable, and scalable evaluations to truly understand the fundamental capabilities of LLMs for programming, pinpointing the ways to unleash their power. Specifically, we plan to enhance the following aspects of BigCodeBench:

- **Multilingualism**: One of the main limitations is that BigCodeBench is Python-only and cannot be easily extended to other programming languages. As the function calls are mostly language-specific, we may not be able to find a package or library with exactly the same functionalities other than Python.

- **Rigorousness**: While we achieve high test coverage for the ground-truth solutions in BigCodeBench, it does not guarantee that any code generated by LLMs will be correctly assessed against existing test cases. Previous works like EvalPlus have attempted to extend the limited test cases by augmenting the input-output pairs via LLM- and mutation-based strategies. However, it is challenging to adapt EvaPlus to the test harness in BigCodeBench, as the harness only examines the expected program behaviors during the runtime (e.g., mocking tests).

- **Generalization**: One intuitive question is "How well do the models generalize to the unseen tools and tasks?" Current BigCodeBench only covers the common libraries and daily programming tasks. It will be more interesting to benchmark models on the programming tasks that use emerging libraries like [transformers](https://github.com/huggingface/transformers) and [langchain](https://github.com/langchain-ai/langchain).

- **Evolution**: Naturally, the libraries can be obsolete or updated, which means that the source code data for model training will constantly evolve. Thus, the models may not memorize function calls from a deprecated library version. This poses a challenge for any tool-dependent programming benchmarks to correctly examine the model capability without periodic updates. Another related concern is the test set contamination issue due to the evolving training data. 

- **Interaction**: Recent interests are around the concept of _LLMs as Agents_, which is deemed as a way towards artificial general intelligence. Specifically, LLMs will be grounded in a less constrained sandbox environment, where they can interact with any given applications, such as the web browser and terminal. The environment can help unlock the capabilities like [self-debugging](https://arxiv.org/pdf/2304.05128) and [self-reflection](https://arxiv.org/abs/2303.11366).


We are excited to see the community's feedback and contributions to build BigCodeBench in the long run 🤗

## Resources
We open-source all the artifacts of BigCodeBench, including the tasks, test cases, evaluation framework, and leaderboard. You can find them as follows:
- [`bigcodebench` GitHub Repository](https://github.com/bigcode-project/bigcodebench)
- [`bigcodebench` HF Data Viewer](https://huggingface.co/spaces/bigcode/bigcodebench-viewer)
- [`bigcodebench` HF Dataset](https://huggingface.co/datasets/bigcode/bigcodebench)
- [`bigcodebench` HF Leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard)
- [`bigcodebench` GitHub Pages Leaderboard](https://bigcode-bench.github.io/)

If you have any questions or suggestions, please feel free to open an issue in the repository or contact us via [terry.zhuo@monash.edu](mailto:terry.zhuo@monash.edu) or [contact@bigcode-project.org](mailto:contact@bigcode-project.org).

## Citation

If you find our evaluations useful, please consider citing our work
```bibtex
@article{bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Min Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and Brunner, Simon and Gong, Chen and Hoang, Thong and Zebaze, Armel Randy and Hong, Xiaoheng and Li, Wen-Ding and Kaddour, Jean and Xu, Ming and Zhang, Zhihan and Yadav, Prateek and Jain, Naman and Gu, Alex and Cheng, Zhoujun and Liu, Jiawei and Liu, Qian and Wang, Zijian and Lo, David and Hui, Binyuan and Muennighoff, Niklas and Fried, Daniel and Du, Xiaoning and de Vries, Harm and Von Werra, Leandro},
  year={2024}
}
```