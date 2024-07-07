---
title: "Our Transformers Code Agent beats the GAIA benchmark 🏅"
thumbnail: /blog/assets/beating-gaia/thumbnail.jpeg
authors:
  - user: m-ric
  - user: sergeipetrov
---

## TL;DR

After some experiments, we were impressed by the performance of Transformers Agents to build agentic systems, so we wanted to see how good it was! We tested using a [Code Agent built with the library](https://github.com/aymeric-roucher/GAIA) on the GAIA benchmark, arguably the most difficult and comprehensive agent benchmark… and ended up on top!


## GAIA: a tough benchmark for Agents

**What are agents?**

In one sentence: an agent is any system based on an LLM that can call external tools or not, depending on the need for the current use case and iterate on further steps based on the LLM output. Tools can include anything from a Web search API to a Python interpreter.

> For a visual analogy: all programs could be described as graphs. Do A, then do B. If/else switches are forks in the graph, but they do not change its structure. We define **agents** as the systems where the LLM outputs will change the structure of the graph. An agent decides to call tool A or tool B or nothing, it decides to run one more step or not: these change the structure of the graph. You could integrate an LLM in a fixed workflow, as in [LLM judge](https://huggingface.co/papers/2310.17631), without it being an agent system, because the LLM output will not change the structure of the graph

Here is an illustration for two different system that perform [Retrieval Augmented Generation](https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain): one is the classical, its graph is fixed. But the other is agentic, one loop in the graph can be repeated as needed.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/beating_gaia/classical_vs_agentic_rag.png" alt="Classical vs Agentic RAG" width=90%>
</p>


Agent systems give LLMs superpowers. For more detail, read[ our earlier blog post on the release of Transformers Agents 2.0](https://huggingface.co/blog/agents).

[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) is the most comprehensive benchmark for agents. The questions in GAIA are very difficult and highlight certain difficulties of LLM-based systems.

Here is an example of a tricky question:

> Which of the fruits shown in the 2008 painting "Embroidery from Uzbekistan" were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film "The Last Voyage"? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit.

You can see this question involves several difficulties:

- Answering in a constrained format.
- Multimodal abilities to read the fruits from the image
- Several informations to gather, some depending on the others:
    * The fruits on the picture
    * The identity of the ocean liner used as a floating prop for “The Last Voyage”
    * The October 1949 breakfast menu for the above ocean liner
- The above forces the correct solving trajectory to use several chained steps.

Solving this requires both high-level planning abilities and rigorous execution, which are precisely two areas where LLMs struggle.

Therefore, it’s an excellent test set for agent systems!

On GAIA’s[ public leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard), GPT-4-Turbo does not reach 7% on average. The top submission is (was) an Autogen-based solution with a complex multi-agent system that makes use of OpenAI’s tool calling functions, it reaches 40%.

**Let’s take them on. 🥊**

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/beating_gaia/prepare_for_battle.gif" alt="Let's fight" width=70%>
</p>


## Building the right tools 🛠️

We used three main tools to solve GAIA questions:

**a. Web browser**

For web browsing, we mostly reused the Markdown web browser from [Autogen team’s submission](https://github.com/microsoft/autogen/tree/gaia_multiagent_v01_march_1st/samples/tools/autogenbench/scenarios/GAIA/Templates/Orchestrator). It comprises a `Browser` class storing the current browser state, and several tools for web navigation, like `visit_page`, `page_down` or `find_in_page`. This tool returns markdown representations of the current viewport. Using markdown compresses web pages information a lot, which  could lead to some misses,  compared to other solutions like taking a screenshot and using a vision model. However, we found that the tool was overall performing well without being too complex to use or edit.

Note: we think that a good way to improve this tool in the future would be to to load pages using selenium package rather than requests. This would allow us to load javascript (many pages cannot load properly without javascript) and accepting cookies to access some pages.

**b. File inspector**

Many GAIA questions rely on attached files from a variety of type, such as `.xls`, `.mp3`, `.pdf`, etc. These files need to be properly parsed.. Once again, we use Autogen’s tool since it works really well.

Many thanks to the Autogen team for open-sourcing their work. It sped up our development process by weeks to use these tools! 🤗

**c. Code interpreter**

We will have no need for this since our agent naturally generates and executes Python code: see more below.


## Code Agent 🧑‍💻


### Why a Code Agent?

As shown by[ Wang et al. (2024)](https://huggingface.co/papers/2402.01030), letting the agent express its actions in code has several advantages compared to using dictionary-like outputs such as JSON. For us, the main advantage is that **code is a very optimized way to express complex sequences of actions**. Arguably if there had been a better way to rigorously express detailed actions than our current programming languages, it would have become a new programming language!

Consider this example given in their paper:


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/beating_gaia/code_vs_json.png" alt="Code agents are just more intuitive than JSON" width=100%>


It highlights several advantages of using code:


- Code actions are **much more concise** than JSON.
    * Need to run 4 parallel streams of 5 consecutive actions ? In JSON, you would need to generate 20 JSON blobs, each in their separate step; in Code it’s only 1 step.
    * On average, the paper shows that Code actions require 30% fewer steps than JSON, which amounts to an equivalent reduction in the tokens generated. Since LLM calls are often the dimensioning cost of agent systems, it means your agent system runs are ~30% cheaper.
- Code enables to re-use tools from common libraries
- Using code gets better performance in benchmarks, due to two reasons:
    * It’s a more intuitive way to express actions
    * LLMs have lots of code in their training data, which possibly makes them more fluent in code-writing than in JSON writing.

We confirmed these points during our experiments on[ agent_reasoning_benchmark](https://github.com/aymeric-roucher/agent_reasoning_benchmark).

From our latest experiments of building transformers agents, we also observed additional advantages:



- It is much easier to store an element as a named variable in code. For example, need to store this rock image generated by a tool for later use?
    * No problem in code: using “rock_image = image_generation_tool(“A picture of a rock”)” will store the variable under the key “rock_image” in your dictionary of variables. Later the LLM can just use its value in any code blob by referring to it again as “rock_image”.
    * In JSON you would have to do some complicated gymnastics to create a name under which to store this image, so that the LLM later knows how to access it again. For instance, save any output of the image generation tool under “image_{i}.png”, and trust that the LLM will later understand that image_4.png is the output of the tool call that precedes it in memory? Or let the LLM also output a “output_name” key to choose under which name to store the variable, thus complicating the structure of your action JSON?
- Agent logs are considerably more readable.


### Implementation of Transformers Agents’ CodeAgent

The thing with LLM generated code is that it can be really unsafe to execute as is. If you let an LLM write and execute code without guardrails, it could hallucinate anything: for instance that all your personal files need to be erased by copies of the Dune lore, or that this audio of you singing the Frozen theme needs to be shared on your blog!

So for our agents, we had to make code execution secure. The usual approach is top-down: “use a fully functional python interpreter, but forbid certain actions”.

To be more safe, we preferred to go the opposite way, and **build a LLM-safe Python interpreter from the ground-up**. Given a Python code blob provided by the LLM, our interpreter starts from the [Abstract Syntax Tree representation](https://en.wikipedia.org/wiki/Abstract_syntax_tree) of the code given by the [ast](https://docs.python.org/3/library/ast.html) python module.  It executes the tree nodes one by one, following the tree structure, and stops at any operation that was not explicitly authorised

For example, an `import` statement will first check if the import is explicitly mentioned in the user-defined list of `authorized_imports`: if not, it does not execute. We include a default list of built-in standard Python functions, comprising for instance `print` and `range`.  Anything outside of it will not be executed except explicitly authorized by the user. For instance, `open` (as in `with open("path.txt", "w") as file:`) is not authorized.

When encountering a function call (`ast.Call`), if the function name is one of the user-defined tools, the tool is called with the arguments to the call. If it’s another function defined and allowed earlier, it gets run normally.

We also do several tweaks to help with LLM usage of the interpreter:

- We cap the number of operations in execution to prevent infinite loops caused by issues in LLM-generated code: at each operation, a counter gets incremented, and if it reaches a certain threshold the execution is interrupted
- We cap the number of lines in print outputs to avoid flooding the context length of the LLM with junk. For instance if the LLM reads a 1M lines text files and decides to print every line, at some point this output will be truncated, so that the agent memory does not explode.


## Basic multi-agent orchestration

Web browsing is a very context-rich activity, but most of the retrieved context is actually useless. For instance, in the above GAIA question, the only important information to get is the image of the painting "Embroidery from Uzbekistan". Anything around it, like the content of the blog we found it on, is generally useless for the broader task solving.

To solve this, using a multi-agent step makes sense! For example, we can create a manager agent and a web search agent. The manager agent should solve the higher-level task, and assign specific web search task to the web search agent. The web search agent should return only the useful outputs of its search, so that the manager is not cluttered with useless information.

We created exactly this multi-agent orchestration in our workflow:
- The top level agent is a [ReactCodeAgent](https://huggingface.co/docs/transformers/main/en/main_classes/agent#transformers.ReactCodeAgent). It natively handles code since its actions are formulated and executed in Python. It has access to these tools:
    - `file_inspector` to read text files, with an optional `question` argument to not return the whole content of the file but only return its answer to the specific question based on the content
    - `visualizer` to specifically answer questions about images.
    - `search_agent` to browse the web. More specifically, this Tool is just a wrapper around a Web Search agent, which is a JSON agent (JSON still works well for strictly sequential tasks, like web browsing where you scroll down, then navigate to a new page, and so on). This agent in turn has access to the web browsing tools:
        - `informational_web_search`
        - `page_down`
        - `find_in_page`
        - … (full list [at this line](https://github.com/aymeric-roucher/GAIA/blob/a66aefc857d484a051a5eb66b49575dfaadff266/gaia.py#L107))

This embedding of an agent as a tool is a naive way to do multi-agent orchestration, but we wanted to see how far we could push it - and it turns out that it goes quite far!


## Planning component 🗺️

There is now [an entire zoo](https://arxiv.org/pdf/2402.02716) of planning strategies, so we opted for a relatively simple plan-ahead workflow. Every N steps we generate two things:

- a summary of facts we know or we can derive from context and facts we need to discover
- a step-by-step plan of how to solve the task given fresh observations and the factual summary above

The parameter N can be tuned for better performance on the target use cas: we chose N=2 for the manager agent and N=5 for the web search agent.

An interesting discovery was that if we do not provide the previous version of the plan as input, the score goes up. An intuitive explanation is that it’s common for LLMs to be strongly biased towards any relevant information available in the context. If the previous version of the plan is present in the prompt, an LLM is likely to heavily reuse it instead of re-evaluating the approach and re-generating a plan when needed.

Both the summary of facts and the plan are then used as additional context to generate the next action. Planning encourages an LLM to choose a better trajectory by having all the steps to achieve the goal and the current state of affairs in front of it.


## Results 🏅

[Here is the final code used for our submission.](https://github.com/aymeric-roucher/GAIA)

We get 44.2% on the validation set: so that means Transformers Agent’s ReactCodeAgent is now #1 overall, with 4 points above the second! **On the test set, we get 33.3%, so we rank #2, in front of Microsoft Autogen’s submission, and we get the best average score on the hardcore Level 3 questions.**

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/beating_gaia/leaderboard.png" alt="We did it!" width=100%>

This is a data point to support that [Code actions work better](https://huggingface.co/papers/2402.01030). Given their efficiency, we think Code actions will soon replace JSON/OAI format as the standard for agents writing their actions.

LangChain and LlamaIndex do not support Code actions out of the box to our knowledge, Microsoft's Autogen has some support for Code actions (executing code in [docker containers](https://github.com/microsoft/autogen/blob/57ec13c2eb1fd227a7976c62d0fd4a88bf8a1975/autogen/code_utils.py#L350)) but it looks like an annex to JSON actions. So Transformers Agents is the only library to make this format central!


## Next steps

We hope you enjoyed reading this blog post! And the work is just getting started, as we’ll keep improving Transformers Agents, along several axes:

- **LLM engine:** Our submission was done with GPT-4o (alas), **without any fine-tuning**. Our hypothesis is that using a fine-tuned OS model would allow us to get rid of parsing errors, and score a bit higher!
- **Multi-agent orchestration:** our is a naive one, with more seamless orchestration we could probably go a long way!
- **Web browser tool:** using the `selenium` package, we could have a web browser that passes cookie banners and loads javascript, thus allowing us to read many pages that are for now not accessible.
- **Improve planning further:** We’re running some ablation tests with other options from the literature to see which method works best. We are planning to give a try to alternative implementations of existing components and also some new components. We will publish our updates when we have more insights!

Keep an eye on Transformers Agents in the next few months! 🚀

And don’t hesitate to reach out to us with your use cases, now that we have built internal expertise on Agents we’ll be happy to lend a hand! 🤝