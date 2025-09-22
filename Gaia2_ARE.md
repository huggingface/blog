---
title: "Gaia2 and ARE: Empowering the community to study agents " 
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_mare_gaia2.png
authors:
- user: clefourrier
  org: OpenEvals
- user: gmialon
  guest: true
  org: meta-agents-research-environments
- user: mlcu
  guest: true
  org: meta-agents-research-environments
- user: mortimerp9
  guest: true
  org: meta-agents-research-environments
- user: xcid
- user: tfrere
- user: evijit
  org: hfmlsoc
- user: RomainFroger
  guest: true
  org: meta-agents-research-environments
- user: dheeraj7596
  guest: true
  org: meta-agents-research-environments
- user: CarolinePascal
  org: lerobot
- user: upiter
  guest: true
  org: meta-agents-research-environments
---

# Gaia2 and ARE: Empowering the Community to Evaluate Agents

In an ideal world, AI agents would be reliable assistants. When given a query, they would easily manage ambiguity in instructions, construct step-by-step plans, correctly identify necessary resources, execute those plans without getting sidetracked, and adapt to unexpected events, all while maintaining accuracy and avoiding hallucinations.
However, developing agents and testing these behaviors is no small feat: if you have ever tried to debug your own agent, you’ve probably observed how tedious and frustrating this can be. Existing evaluation environments are tightly coupled with the tasks they evaluate, lack real-world flexibility, and do not reflect the messy reality of open-world agents: simulated pages never fail to load, events don’t spontaneously emerge, and asynchronous chaos is absent.

That’s why we’re very happy to introduce Gaia2, the follow-up to the agentic benchmark GAIA, allowing analysis of considerably more complex behaviors. Gaia2 is released with the open Meta Agents Research Environments (ARE) framework to run, debug and evaluate agents. ARE simulates complex real world-like conditions and can be customized to further study agents behaviors. Gaia2 dataset is released under CC by 4.0 license, and ARE under MIT license.

![Gaia2 Budget Scaling curves](https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/fig1_budget_scaling_curves.png)


## Gaia2: Agentic Evaluation on Real Life Assistant Tasks

[GAIA](https://arxiv.org/abs/2311.12983) is an agentic benchmark published in 2023, with 3 levels of information retrieval questions requiring tools, web browsing, and reasoning to solve. In 2 years, the easiest levels have become too easy for models, and the community is coming close to solving the hardest questions, so it was time for an entirely new and harder agent benchmark! 

Here comes Gaia2, a follow up to GAIA, going way beyond it in terms of capabilities studied! 

Where GAIA was read-only, Gaia2 is now a read-and-write benchmark, focusing on interactive behavior and complexity management. Agents are now evaluated not only on search and retrieval, but also on instruction following over ambiguous or time-sensitive queries, in a noisy and environment with controlled failures - reflecting real-world conditions more than any other simulated environment. We want to test how agents manage tools or APIs that sometimes do not work, plan successions of actions with very specific time frames, and adapt to new events - a whole new range of complexity!


To do this, we use the following task groups (thanks to 1000 brand new human-created scenarios):
- **Execution**: Multi-step instruction following and tool-use (e.g., contact updates)  
- **Search**: Cross-source information gathering (e.g., friend cities from WhatsApp)  
- **Ambiguity Handling**: Clarification of conflicting requests (e.g., scheduling conflicts)  
- **Adaptability**: Response to changes in the simulation (e.g., updating an email using follow up information)
- **Time/temporal Reasoning**: Time-sensitive actions (e.g., cab orders after 3-minute delays)
- **Agent-to-Agent Collaboration**: Communication between agents without direct API access  
- **Noise Tolerance**: Robustness to API failures and environmental instability  

In the spirit of GAIA, scenarios do not require specialized knowledge: humans should in principle be able to get 100%, which allows easy debugging for model developers.

Want to explore the benchmark? Check out our [dataset](https://huggingface.co/datasets/meta-agents-research-environments/Gaia2), which you can better display in our demo [here](https://huggingface.co/spaces/meta-agents-research-environments/demo).

### How does Gaia2 run? 

Gaia2 runs with ARE, an execution environment, where an agent of your choice has access to a combination of applications and associated pre-populated data. 

For Gaia2, we created a **smartphone mock-up** environment, simulating what a human would use in their daily life. It contains real-world applications such as messaging (Email), utilities (Calendar, Contacts, Shopping, a FileSystem, …), and a chat interface to talk to the agent. All applications are also accessible to the agents through tool calling. Last but not least, the demo also contains a simulated persona’s history of conversations and app interactions.

All agent interactions are automatically recorded as **structured traces** during execution for deep dives and analysis: they include tool calls, API responses, model thoughts, timing metrics (e.g., response latency), user interactions, and so forth - and can all be exported as JSON.

![ARE](https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/fig2_structure_of_are.png)


### Results

For reference, we compare a range of large open and closed source models: Llama 3.3-70B Instruct, Llama-4-Maverick, GPT-4o, Qwen3-235B-MoE, Grok-4, Kimi K2, Gemini 2.5 Pro, Claude 4 Sonnet, and GPT-5 in all reasoning modes.

All models are evaluated using the same setup (a uniform ReAct loop for consistency, temperature of 0.5, generation limit of 16K tokens), with a combination of model-as-a-judge (Llama 3.3 Instruct 70B) and exact-match evaluation depending on the particular task. All 101 tools (and the general environment description) are provided in the system prompt. 

![Results](https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/fig9_Gaia2_scores_per_capability.png)


Among the evaluated models, the **highest-scoring model** overall as of September 2025 is GPT-5 with high reasoning, and the best open source model is Kimi K2. 

Some capabilities appear to be already **close to solved** by the best models: execution of simple tool calls and instruction following (`execution`), and overall `search` (as we could have guessed from current results on GAIA). The ambiguity, adaptability, and noise splits remain **challenging** for now for all models, and it’s interesting to see that performance on what were considered complex agentic tasks (instruction following and search) is not a good proxy for performance on closer-to-real-world tasks. Last but not least, the hardest split for all models at the moment is the `time` one: **it’s very hard at this moment for models to correctly handle time-sensitive actions** (though this could likely be mitigated by the use of specialised tools and better temporal reasoning). Detailed analysis of these results can be found in the paper.

However, we believe it’s important to **push reporting beyond raw scores**: if the model is correct but took several thousand tokens to reach the correct solution, or ran for several hours, it is “not as good” as a model which succeeded orders of magnitude faster. We therefore also normalize scores for cost, quantified as the average number of LLM calls and output tokens (which both define a cost-performance Pareto frontier). In the paper you’ll find score vs monetary cost and time.

![Pareto](https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/fig12_calls_tokens_vs_score_pareto_frontier.png)


### Compare with your favorite models! Evaluating on Gaia2

If you want to evaluate your model on Gaia2, you can follow these steps:

First, install Meta's Agent Research Environment in your Python environment of choice (uv, conda, virtualenv, ...)

```bash
pip install meta-agents-research-environments
```

Then, run the benchmark for all configurations: execution, search, adaptability, time and ambiguity. Don't forget to upload all results to the hub with the hf_upload kwarg!

```bash
are-benchmark run --hf meta-agents-research-environments/Gaia2     --split validation --config CONFIGURATION     --model YOUR_MODEL --model_provider YOUR_PROVIDER     --agent default     --max_concurrent_scenarios 2     --scenario_timeout 300     --output_dir ./monitored_test_results     --hf_upload YOUR_HUB_DATASET_TO_SAVE_RESULTS 
```

Run the oracle to get your aggregated score file

```bash
are-benchmark judge --hf meta-agents-research-environments/Gaia2     --split validation --config CONFIGURATION     --agent default     --max_concurrent_scenarios 2     --scenario_timeout 300     --output_dir ./monitored_test_results --hf_upload YOUR_HUB_DATASET_TO_SAVE_RESULTS 
```

Finally, add all the relevant information about your model in the README, and share it on the leaderboard to centralize Gaia2 traces [here](https://huggingface.co/spaces/meta-agents-research-environments/leaderboard)!

## Beyond Gaia2: study your agents with ARE

Beyond benchmark scenarios, you can use Gaia2 apps and content in ARE to see if the model is able to correctly solve less verifiable tasks such as loading emails, writing  follow-ups, adding events to the calendar or booking meetings - in sum, providing the perfect setup to **evaluate your AI assistants through interaction**! 

You can also easily customise the environment, by 1) **connecting your tools** (via MCP or directly ) to test your agents on it; 2) **implementing your own scenarios**, including defining **trigger or timed events** (eg: after 2 minutes, the Mail app will receive a new email from Contact), to see how the agent is able to adapt to an evolving environment 

(As the agents are by default `json agents`, they can’t mess up your machine, unless of course you connect them to external apps with unsafe rights. So, operate with caution when adding your own apps or using untrusted MCPs) 

Here are several use cases that we’ve used ARE for:
- **Vibe-check any agent** on real or simulated data, to study a variety of setups, with their own rules, tools, content, and verifications
- Test agent **tool calling and orchestration capabilites**, either with local apps or MCP tools
- Generate your own tool-calling trace to **fine-tune tool calling models**
- Easily gather and **reproduce existing agentic benchmarks** in a unified framework
- Debug and **study agent to agent interactions on the fly** within the user interface 
- **Study model limitations** in noisy environments (with API timeouts and ambiguity)


We recorded 3 videos so you can check some of these use cases (but of course, we hope the community gets creative with ARE :hugging_face:). For these videos, we use the default demo described above, which contains the simulated life of Linda Renne, PhD student in machine learning. 

### 1) Testing an agent on a simple task: event organisation

To test how good the default model is at event organisation, let’s plan a birthday party! 

We first ask the agent to text everyone in the Renne family about the user’s 30th birthday party on November 7. The default universe has 21 contacts in the list, including 5 Renne family members - Linda, the simulation “owner”, George and Stephie, her parents, Anna her sister, and Morgan her grandfather. The agent successfully goes through the contact list, finds the four family members, and texts them. 

Next, we ask the agent to create a calendar invite and add them as invitees. The agent remembers the above context! It creates a calendar invite on the correct date and correctly adds the family members to it.


<video controls width="800">
    <source src="https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/demo_base.mov" type="video/mp4">
    Your browser does not support the video tag.
</video>


### 2) Understanding agents: deep diving the traces

ARE also allows us to check the traces behind the actions taken by the agent. 
Upon opening the Agent logs tool on the left, we can see the system prompt, the chain of thought, multi-step actions taken with the tools called, and the outcomes as neatly organised logs. Everything can be exported as json if you want to consult things offline!

<video controls width="800">
    <source src="https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/demo_traces.mov" type="video/mp4">
    Your browser does not support the video tag.
</video>


### 3) Playing around and extending the demo: Connecting the agent to your own MCPs

In this last example, we connect ARE to a remote robot arm via MCP, so it can gesture things to us, then ask the agent to answer our yes or no questions by waving the robot arm! Here’s what it looks like.

<video controls width="800">
    <source src="https://huggingface.co/spaces/meta-agents-research-environments/demo/resolve/main/blog_assets/demo_robot_short.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

But these examples are only very simple starting points, and we’re really looking towards what you’ll build! (For more advanced users, you can even directly install and edit the Meta-ARE code [here](https://github.com/facebookresearch/meta-agents-research-environments).)

## Conclusion

Gaia2 and ARE are new research tools that we hope will empower anyone to easily build more reliable and adaptable AI agents - by allowing easy experiments, making real-world evaluation accessible to anyone, as well as improving trust through transparent, reproducible benchmarks and debuggable traces.

We’d love to see what you will do with this project!
