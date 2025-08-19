---
title: "TextQuests: How Good are LLMs at Text-Based Video Games?" 
thumbnail: /blog/assets/textquests/thumbnail.gif
authors:
- user: justinphan3110
  org: cais
  guest: true
- user: clefourrier
---

# TextQuests: How Good are LLMs at Text-Based Video Games?

The rapid advancement of Large Language Models (LLMs) has enabled remarkable progress on established academic and industrial benchmarks. Knowledge benchmarks, such as MMLU and GPQA, are now largely saturated, and frontier models are making significant progress on expert evaluations like [HLE](lastexam.ai). However, this success in static, knowledge-based tasks does not always translate to effectiveness in dynamic, interactive settings, the kind of environment in which we would want effective assistants and AI agents to perform well. Developing robust methodologies for evaluating LLMs as autonomous agents in complex, exploratory environments remains a significant challenge.

Two core avenues exist to evaluate autonomous agents: either use real-world environments and a limited set of specific skills, such as tool use or coding capabilities, or use simulated open-world environments. The latter better captures an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context, while being easy to evaluate.
While this direction is still developing, it has seen growing interest through benchmarks such as [Balrog](https://balrogai.com), ARC-AGI, and demonstrations of models like Claude and Gemini playing Pokémon. Building on this emerging vein of work, we introduce [TextQuests](https://huggingface.co/spaces/cais/textquests).


<div style="display: flex; justify-content: center; margin: -3.5rem 0;">
  <img 
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/textquests/textquests_logo.png" 
    alt="TextQuests Logo" 
    style="width: 30px; height: 30px;"
  >
</div>

## TextQuests

TextQuests is a benchmark built upon 25 classic [Infocom](https://en.wikipedia.org/wiki/Infocom) interactive fiction games. These once-popular text-based video games, which can take human players over 30 hours and require hundreds of precise actions to solve, provide a compelling testbed for the challenges of agentic reasoning. They demand that an agent demonstrate:

- **Long-Context Reasoning:** Agents must devise and execute multi-step plans by reasoning over a long and continuously growing history of actions and observations, relying solely on their intrinsic capabilities without the aid of external tools.

- **Learning through Exploration:** The games require agents to learn from experience, interrogating their own failures and making incremental improvements through trial-and-error as they explore the unknown world.

Success in these games requires an agent to build understanding over a long gameplay session. This allows for a more direct and accurate assessment of the LLM itself as the reasoning backbone of an AI agent system.

<figure class="image text-center">
    <figcaption> Examples showing the diverse reasoning challenges in TextQuests.

  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/textquests/textquests_examples.png" alt="Text watermark">
</figure>


## Evaluations

For each model, we conduct two distinct evaluation runs: one with access to the game's official hints (```With Clues```) and one without (```No Clues```). Each run is executed for a maximum of 500 steps and stops early if the agent successfully completes the game. To handle the growing context, the full game history is maintained without truncation throughout the run. This long-context evaluation is computationally feasible due to the prompt caching inherent in modern LLM inference frameworks. We employ two main evaluation metrics:

  - **Game Progress.** The Game Progress metric is calculated based on a series of labeled checkpoints representing necessary objectives on the path to finishing a game.

  - **Harm.** To assess the ethical behavior of the agents, we measure Harm by tracking specific in-game actions that are considered harmful to some degree. This score is then averaged across all games to evaluate an agent's overall tendency to perform such actions.

<figure class="image text-center">
  <figcaption> LLMs performance on TextQuests.
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/textquests/textquests_results.png" alt="results">
  
</figure>

## Discussion

**Long-context Reasoning.** During evaluation, the context window can exceed 100K tokens, requiring LLMs to consistently perform precise reasoning and planning over a vast history of observations and clues to effectively progress. As the context length grows, we observe that current models often hallucinate about prior interactions, such as believing they have already picked up an item when they have not or getting stuck navigating in a loop. Furthermore, similar to observations in [Gemini 2.5 Plays Pokémon](https://arxiv.org/abs/2507.06261), LLM agents show an increased tendency to repeat actions from their history rather than synthesizing novel plans as the context lengthens. These long-context failures are particularly stark in tasks requiring spatial reasoning. For instance, in <u><em>Wishbringer</em></u>, most LLMs struggled to navigate back down a cliff after climbing it. The solution simply required reversing the sequence of directions used to ascend—information available in the context history—indicating a fundamental difficulty in building and utilizing a mental map. Similarly, all frontier LLMs struggle in navigating the infamous Maze in <u><em>Zork I</em></u>.


<figure class="image text-left">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/textquests/textquests_fail.png" alt="reasoning">
  <figcaption>
    Examples of long context reasoning failures in TextQuests. <strong>Left:</strong> In <u><em>Zork I</em></u>, tested LLMs failed to correctly recall information from its history, hallucinating that it dropped a matchbook in the <code>Studio</code> instead of the <code>Atlantis Room</code>. <strong>Right:</strong> In <u><em>Wishbringer</em></u>, LLMs often fail to retrieve and reverse their own ascent path from in-context history to navigate down a cliff successfully.
  </figcaption>
</figure>

**Dynamic Thinking.** An agent's overall effectiveness is defined by both its task success and its operational efficiency. For LLM agents, efficiency is closely tied to the number of output or reasoning tokens it generates, which directly impacts inference cost and latency. Models that utilize more test-time compute generally achieve higher performance. However, this trend starts to diminish after a certain budget. This consideration is important as many exploratory steps in TextQuests (for example, navigation steps) are intermediate and can be successfully executed without a large reasoning depth.

<figure class="image text-left">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/textquests/textquests_dynamic_thinking.png" alt="thinking">
  <figcaption>
    A comparison of output and reasoning token efficiency across state-of-the-art LLMs on TextQuests. Since many exploratory steps are intermediate and do not require a full reasoning budget, an ideal LLM agent should be efficient and dynamic with its reasoning effort while still maintaining consistent performance.
  </figcaption>
</figure>

In closing, TextQuests is an evaluation of how well models can consistently progress through a series of classic interactive fiction games that were once popular among human players. We hope that open-sourcing TextQuests helps researchers better understand and assess the current capabilities of LLM agents in challenging exploratory environments. Open-source model builders are welcome to submit to [TextQuests Leaderboard](https://huggingface.co/spaces/cais/textquests) by sending us an email at [agibenchmark@safe.ai](mailto:agibenchmark@safe.ai)



## Citations
```
@misc{phan2025textquestsgoodllmstextbased,
  title={TextQuests: How Good are LLMs at Text-Based Video Games?}, 
  author={Long Phan and Mantas Mazeika and Andy Zou and Dan Hendrycks},
  year={2025},
  eprint={2507.23701},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2507.23701}, 
}
```
