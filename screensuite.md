---
title: "Introducing ScreenSuite - The most comprehensive evaluation suite for GUI Agents!"
thumbnail: /blog/assets/screensuite/thumbnail.png
authors:
  - user: a-mahla
  - user: m-ric
  - user: thomwolf
---
## We’re releasing ScreenSuite, the most comprehensive evaluation suite for GUI Agents!

**TL;DR**

Over the past few weeks, we’ve been working tirelessly on making GUI agents more open, accessible and easy to integrate. Along the way, we created the largest benchmarking suite for GUI agents performances 👉 let us introduce [ScreenSuite](https://github.com/huggingface/screensuite).

We are very excited to share it with you today: [ScreenSuite](https://github.com/huggingface/screensuite) is the most comprehensive and easiest way to evaluate [Vision Language Models](https://huggingface.co/blog/vlms) (VLMs)across many agentic capabilities!

### WTF is a GUI Agent?

<div>
  <video controls style="margin-bottom:0;">
    <source src="https://os-world.github.io/static/videos/main.mp4" type="video/mp4">
  </video>
  <p style="color:grey;margin-top:0;"><i>GUI Agents in action - courtesy of <a href="https://os-world.github.io/">OSWorld<a></i></p>
</div>

In short, an AI Agent is a robot that acts in the virtual world. ([more thorough definition here](https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents)) 

In particular, a “GUI Agent” is an agent that lives in a GUI. Think “an agent that can do clicks and navigate on my desktop or my phone”, à la Claude Computer Use.

This means in essence that the AI model powering the agent will be given a task like “Fill the rest of this Excel column”, along with screen captures of the GUI. Using this information, it will then decide to take action on the system : `click(x=130, y=540)` to open a web browser, `type(”Value for XYZ in 2025")`, `scroll(down=2)` to read further… To see a GUI agent in action, you can try our [Open Computer Agent](https://huggingface.co/spaces/smolagents/computer-agent), powered by [Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct).

A good GUI agent will be able to navigate a computer just like we would, thus unlocking all computer tasks : scrolling through Google Maps, editing a file, buying an item online. This involves a variety of capabilities that can be hard to evaluate.

### Introducing ScreenSuite 🥳

The literature, for instance [Xu et al. (2025)](https://arxiv.org/abs/2412.04454) or [Qin et al. (2025)](https://arxiv.org/abs/2501.12326), generally splits GUI agent abilities amongst several categories:

1. Perception: correctly perceiving the informations displayed on screen
2. Grounding: understanding the positioning of elements - this is paramount to click the correct place
3. Single step actions: solving instructions correctly over one action
4. Multi-step agents: solving a higher-level goal through several actions in a GUI environment.

So our first contribution is to **gather and unify a comprehensive suite of 13 benchmarks spanning the ensemble of these GUI agent capabilities.**

If you look at the last category listed above, evaluating Multi-step agentic capabilities is especially challenging as it requires virtual machines to run the agent’s environment, be it Windows, Android, Ubuntu... To address this, we provide support both for [E2B desktop](https://github.com/e2b-dev/desktop) remote sandboxes, and we created from scratch a new option to easily launch Ubuntu or Android virtual machines in Docker!

Full markdown table (visualize it on [https://huggingface.co/new-blog](https://huggingface.co/new-blog))

| **Category**                 | **Benchmark**                                                                                    | **Environment** | **Sample count**    |
| ---------------------------- | ------------------------------------------------------------------------------------------------ | --------------- | ------------------- |
| **Perception / Grounding 👁️** | [ScreenQA-Short](https://github.com/google-research-datasets/screen_qa)                          | Mobile          | 8.4k                |
|                              | [ScreenQA-Complex](https://github.com/google-research-datasets/screen_qa/tree/main/complex_qa)   | Mobile          | 11.8k               |
|                              | [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2/tree/main)              | Desktop         | 1.3k                |
|                              | [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)                   | Desktop         | 1.6k                |
|                              | [WebSRC](https://huggingface.co/datasets/X-LANCE/WebSRC_v1.0)                                    | Web             | 52k                 |
|                              | [VisualWebBench](https://huggingface.co/datasets/visualwebbench/VisualWebBench)                  | Web             | 1.5k                |
| **Single-Step Actions 🎯**    | [Showdown-clicks](https://huggingface.co/datasets/generalagents/showdown-clicks)                 | Web             | 0.6k                |
|                              | [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) | Mobile          | 3k                  |
|                              | [Multimodal-Mind2web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web)                | Web             | 6.4k                |
| **Multi-Step Agents 🐾**      | [AndroidWorld](https://github.com/google-research/android_world) (incl MobileMiniWob)            | Mobile          | 116 tasks, infinite |
|                              | [OSWorld](https://os-world.github.io/)                                                           | Desktop         | 369                 |
|                              | [BrowseComp](https://openai.com/index/browsecomp/)                                               | Web             | 1.27k               |
|                              | [GAIA-Web](https://huggingface.co/gaia-benchmark)                                                | Web             | 132                 |
|                              | [Mind2Web-Live](https://osu-nlp-group.github.io/Mind2Web/)                                       | Web             | 208                 |

**Implementation details**

We’ve carefully designed our benchmark suite with modularity and consistency in mind, ensuring strong alignment across tasks and environments. When required, especially for online benchmarks, we leverage [smolagents](https://github.com/serain/smolagents) as framework layer to streamline agent execution and orchestration.

To support reproducibility and ease of use, we’ve built custom Dockerized containers that allow local deployment of full **Ubuntu Desktop** or **Android** environments.

Unlike many existing GUI benchmarks that rely on accessibility trees* or other privileged metadata alongside visual input, our stack is intentionally **vision-only**. While this can result in different scores on some established leaderboards, it creates a more realistic and challenging setup, one that better reflects how humans perceive and interact with graphical interfaces.

### How do existing VLMs compare? 📊



We’ve evaluated leading VLMs on the benchmark, and managed to recover existing scores - with the caveat that we evaluate on vision only, causing some differences, see below.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/screensuite/scores_screensuite.png"/>
</div>

>[!NOTE]
> 💡 Note that ScreenSuite does not intend to exactly reproduce benchmarks published in the industry: we wanted to evaluate “agents that act on a GUI, taking only the screenshots as input”. As a result, on benchmarks like Mind2Web where other benchmarks gave the agent a view of information rich context like DOM or accessibility tree, our evaluation setting is much harder, thus we don’t recover the same scores. But this does not conflict with our goal is of comparing VLMs on vision-only agentic ability.

– All agentic frameworks (Android World, OSWorld, GAIAWeb, Mind2Web) use [smolagents](https://github.com/huggingface/smolagents) and rely solely on **vision**, without any accessibility tree or DOM added (in contrast with evaluation settings reported in other sources).
– **Mind2Web (Multimodal)** originally used **element-name-based multi-choice selection** based on the accessibility tree and screenshots, but was later adapted to **click precision within bounding boxes** using **vision only**, which significantly increases task difficulty.

### Start your custom evaluation in 30s ⚡️

Head to [the repository](https://github.com/huggingface/screensuite).

1. Clone the repository with submodules: `git clone --recurse-submodules git@github.com:huggingface/screensuite.git`
2. Install the package: `uv sync --extra submodules --python 3.11`
3. Run `python run.py` 
    - Alternatively, run `python examples/run_benchmarks.py` for more fine-grained control, like running evaluations for several models in parallel.

>[!NOTE]
> The multistep benchmarks **requires a bare-metal machine** to run and deploy desktop/mobile* environment *emulators (see [README.md](https://github.com/huggingface/screensuite/blob/main/README.md))

### Next steps 🚀

Running consistent and meaningful evaluations easily allows the community to quickly iterate and make progress in this field, as we’ve seen with [Eleuther LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness), the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/) and the [Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard).

We hope to see much more capable open models in the coming month that can run a wide range of tasks reliably and even run locally!