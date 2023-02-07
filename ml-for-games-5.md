---
title: "Generating Stories: AI for Game Development #5"
thumbnail: /blog/assets/124_ml-for-games/thumbnail5.png
authors:
- user: dylanebert
---

<h1>Generating Stories: AI for Game Development #5</h1>

<!-- {blog_metadata} -->
<!-- {authors} -->
 

**Welcome to AI for Game Development!** In this series, we'll be using AI tools to create a fully functional farming game in just 5 days. By the end of this series, you will have learned how you can incorporate a variety of AI tools into your game development workflow. I will show you how you can use AI tools for:

1. Art Style
2. Game Design
3. 3D Assets
4. 2D Assets
5. Story

Want the quick video version? You can watch it [here](https://www.tiktok.com/@individualkex/video/7190364745495678254). Otherwise, if you want the technical details, keep reading!

<!-- TODO: Update video link -->

**Note:** This post makes several references to [Part 2](https://huggingface.co/blog/ml-for-games-2), where we used ChatGPT for Game Design. Read Part 2 for additional context on how ChatGPT works, including a brief overview of language models and their limitations.

## Day 5: Story

In [Part 4](https://huggingface.co/blog/ml-for-games-4) of this tutorial series, we talked about how you can use Stable Diffusion and Image2Image as a tool in your 2D Asset workflow.

In this final part, we'll be using AI for Story. First, I'll walk through my [process](#process) for the farming game, calling attention to :warning: **Limitations** to watch out for. Then, I'll talk about relevant technologies and [where we're headed](#where-were-headed) in the context of game development. Finally, I'll [conclude](#conclusion) with the final game.

### Process

**Requirements:** I'm using [ChatGPT](https://openai.com/blog/chatgpt/) throughout this process. For more information on ChatGPT and language modeling in general, I recommend reading [Part 2](https://huggingface.co/blog/ml-for-games-2) of the series. ChatGPT isn't the only viable solution, with many emerging competitors, including open-source dialog agents. Read ahead to learn more about [the emerging landscape](#the-emerging-landscape) of dialog agents.

1. **Ask ChatGPT to write a story.** I provide plenty of context about my game, then ask ChatGPT to write a story summary.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt1.png" alt="ChatGPT for Story #1">
</div>

ChatGPT then responds with a story summary that is extremely similar to the story of the game [Stardew Valley](https://www.stardewvalley.net/). 

> :warning: **Limitation:** Language models are susceptible to reproducing existing stories.

This highlights the importance of using language models as a tool, rather than as a replacement for human creativity. In this case, relying solely on ChatGPT would result in a very unoriginal story.

2. **Refine the results.** As with Image2Image in [Part 4](https://huggingface.co/blog/ml-for-games-4), the real power of these tools comes from back-and-forth collaboration. So, I ask ChatGPT directly to be more original.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt2.png" alt="ChatGPT for Story #2">
</div>

This is already much better. I continue to refine the result, such as asking to remove elements of magic since the game doesn't contain magic. After a few rounds of back-and-forth, I reach a description I'm happy with. Then, it's a matter of generating the actual content that tells this story.

3. **Write the content.** Once I'm happy with the story summary, I ask ChatGPT to write the in-game story content. In the case of this farming game, the only written content is the description of the game, and the description of the items in the shop.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt3.png" alt="ChatGPT for Story #3">
</div>

Not bad. However, there is definitely no help from experienced farmers in the game, nor challenges or adventures to discover.

4. **Refine the content.** I continue to refine the generated content to better fit the game.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt4.png" alt="ChatGPT for Story #4">
</div>

I'm happy with this result. So, should I use it directly? Maybe. Since this is a free game being developed for an AI tutorial, probably. However, it may not be straightforward for commercial products, having potential unintended legal, ethical, and commercial ramifications.

> :warning: **Limitation:** Using outputs from language models directly may have unintended legal, ethical, and commercial ramifications.

Some potential unintended ramifications of using outputs directly are as follows:
- <u>Legal:</u> The legal landscape surrounding Generative AI is currently very unclear, with several ongoing lawsuits.
- <u>Ethical:</u> Language models can produce plagiarized or biased outputs. For more information, check out the [Ethics and Society Newsletter](https://huggingface.co/blog/ethics-soc-2).
- <u>Commercial:</u> [Some](https://www.searchenginejournal.com/google-says-ai-generated-content-is-against-guidelines/444916/) sources have stated that AI-generated content may be deprioritized by search engines. This [may not](https://seo.ai/blog/google-is-not-against-ai-content) be the case for most non-spam content, but is worth considering. Tools such as [AI Content Detector](https://writer.com/ai-content-detector/) can be used to check whether content may be detected as AI-generated. There is ongoing research on language model [watermarking](https://arxiv.org/abs/2301.10226) which may mark text as AI-generated.

Given these limitations, the safest approach may be to use language models like ChatGPT for brainstorming but write the final content by hand.

5. **Scale the content.** I continue to use ChatGPT to flesh out descriptions for the items in the store.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/chatgpt5.png" alt="ChatGPT for Story #5">
</div>

For my simple farming game, this may be an effective approach to producing all the story content for the game. However, this may quickly run into scaling limitations. ChatGPT isn't well-suited to very long cohesive storytelling. Even after generating a few item descriptions for the farming game, the results begin to drift in quality and fall into repetition.

> :warning: **Limitation:** Language models are susceptible to repetition.

To wrap up this section, here are some tips from my own experience that may help with using AI for Story:
- **Ask for outlines.** As mentioned, quality may deteriorate with long-form content. Developing high-level story outlines tends to work much better.
- **Brainstorm small ideas.** Use language models to help flesh out ideas that don't require the full story context. For example, describe a character and use the AI to help brainstorm details about that character.
- **Refine content.** Write your actual story content, and ask for suggestions on ways to improve that content. Even if you don't use the result, it may give you ideas on how to improve the content.

Despite the limitations I've discussed, dialog agents are an incredibly useful tool for game development, and it's only the beginning. Let's talk about the emerging landscape of dialog agents and their potential impact on game development.

### Where We're Headed

#### The Emerging Landscape

My [process](#process) focused on how ChatGPT can be used for story. However, ChatGPT isn't the only solution available. [Character.AI](https://beta.character.ai/) provides access to dialog agents that are customized to characters with different personalities, including an [agent](https://beta.character.ai/chat?char=9ZSDyg3OuPbFgDqGwy3RpsXqJblE4S1fKA_oU3yvfTM) that is specialized for creative writing.

There are many other models which are not yet publicly accessible. Check out [this](https://huggingface.co/blog/dialog-agents) recent blog post on dialog agents, including a comparison with other existing models such as [Google's LaMDA](https://arxiv.org/abs/2201.08239) and [Bard](https://blog.google/technology/ai/bard-google-ai-search-updates/), [Meta's BlenderBot](https://arxiv.org/abs/2208.03188), [DeepMind's Sparrow](https://arxiv.org/abs/2209.14375), and [Anthropic's Assistant](https://arxiv.org/abs/2204.05862). It remains to be seen how these models will compete and play out in the near future.

While many prevalent contenders are closed-source, there are also open source dialog agent efforts, such as [LAION's OpenAssistant](https://github.com/LAION-AI/Open-Assistant), reported efforts from [CarperAI](https://carper.ai), and the open source release of [Google's FLAN-T5 XXL](https://huggingface.co/google/flan-t5-xxl). These can be combined with open source tools like [LangChain](https://github.com/hwchase17/langchain), which allow language model inputs and outputs to be chained, helping to work toward open dialog agents.

Just as the open source release of Stable Diffusion has given rapid rise to a wide variety of innovations that have inspired this series, the open source community will be key to exciting language-centric applications in game development that are yet to be seen. To keep up with these developments, feel free to follow me on [Twitter](https://twitter.com/dylan_ebert_). In the meantime, let's talk about what some of these potential developments are.

#### In Game Development

**NPCs:** Aside from the clear uses of language models and dialog agents in the game development workflow, there is an exciting in-game potential for this technology that has not yet been realized. The most clear case of this is AI-powered NPCs. There are already startups built around the idea. Personally, I don't quite see how language models, as they currently are, can be applied to create compelling NPCs. However, I definitely don't think it's far off. I'll let you know.

**Controls.** What if you could control a game by talking to it? This is actually not too hard to do right now, though it hasn't been put into common practice. Would you be interested in learning how to do this? Stay tuned.

### Conclusion

Want to play the final farming game? Check it out [here](https://individualkex.itch.io/).

<!-- TODO: Update link to game in spaces -->

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/game.png" alt="Final Farming Game">
</div>

Thank you for reading the AI for Game Development series! This series is only the beginning of AI for Game Development at Hugging Face, with more to come. Have questions? Want to get more involved? Join the [Hugging Face Discord](https://hf.co/join/discord)!