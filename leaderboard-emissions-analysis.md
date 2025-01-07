---
title: "CO₂ Emissions and Models Performance: Insights from the Open LLM Leaderboard" 
thumbnail: /blog/assets/leaderboard-emissions-analysis/thumbnail.png
authors:
- user: alozowski
- user: SaylorTwift
- user: albertvillanova
- user: clefourrier
- user: alozowski
- user: SaylorTwift
- user: albertvillanova
---

# CO₂ Emissions and Models Performance: Insights from the Open LLM Leaderboard

# Introduction

Since June 2024, we have evaluated more than 3,000 models on the Open LLM Leaderboard v2, a worldwide ranking of open language models performance. Even though we’re trying to run evaluations without wasting resources (we use the spare cycles of our cluster, in other words the GPUs which are active but waiting between jobs), this still represents quite a big amount of energy spent for model inference!

In the last year, people have become more and more aware that using large language models (LLMs) to generate text has a significant environmental impact, beyond the already important impact of training. Recent research (see [Towards Greener LLMs](https://arxiv.org/abs/2403.20306) article) highlights the challenges of managing resources efficiently at inference due to dynamic and diverse workloads.

By integrating these scores into the Open LLM Leaderboard, we aim to provide transparency to users about the carbon impact of various model evaluations, and hopefully encourage model creators to balance performance with environmental responsibility. 

We were curious to explore the CO₂ emissions associated with model inference and to identify any emerging trends in the data. Along the way, we discovered some surprising insights alongside a few predictable patterns!

💡  **Note:** We’ve created a [Colab notebook](https://colab.research.google.com/drive/1gaXUywH9fB5rINM8Ocg_U26u5voDmLUi?usp=sharing) containing all the data and findings discussed here. This notebook allows you to explore the results, run the analyses yourself, and even adapt the code to investigate further questions

# Computing CO₂ cost

Let's take a closer look at how we calculate the CO₂ emissions generated during model inference.

In our case, we use a straightforward heuristic, because all evaluations are run on the same hardware using the same method *(method: loading the model with Transformers and Accelerate using a combination of pipeline parallelism and data parallelism to use our 8 GPUs per node to the fullest)*. It involves factoring in:

- Evaluation time.
- Energy usage based on the power consumption of our cluster’s hardware.
- Carbon intensity of the electricity source powering our hardware.

A detailed explanation and formula can be found in [our documentation](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/emissions).

**Caveat:** This does not mean that model X emits Y CO₂ at inference in general!
Instead, what this means is that model X emitted Y CO₂ on our very specific inference setup, and you can still learn a lot from that 😀

# General Trends

💡 **Note:** Since we wanted to look at general trends, we kept only the most frequent model architectures, and models for which we had the parameter count. 

We therefore looked at 2,796 models from the recent families Gemma/Gemma2, all generations of Llama, Mistral, Mixtral, as well as Phi/Phi3, Qwen2 and above. We also included older model families such as GPT, GPT-NeoX, and T5.

## **“Official Providers” Models**

Official models come from high-quality trusted model creators, such as research groups or community consortiums (EleutherAI, NousResearch), FAANG (Google, Meta, Alibaba…), startups (MistralAI, 01AI), etc, who have taken the time and compute to create new high quality models. They represent 339 models.

![official_providers_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/official_providers_models.png)

- As expected, overall, the bigger the model size, the higher the CO₂ cost. However, the increase in performance is not always proportional, leading to diminishing returns on investment (ROI).
    - Models from AbacusAI, Qwen, and AllenAI, around 70B parameters, achieve an average performance score above 40 across multiple evaluation benchmarks.
    - On the other hand, the lowest-ranked models in the top-right quadrant are older models: `Qwen-1.5-100B` models, with `Mixtral8x22B` showing the weakest performance.
    - Overall, MoEs seem to have a relatively poor ROI. Although these models aim to reduce computational overhead by activating only a subset of their parameters for a given task, some exhibit higher-than-expected CO₂ emissions due to extremely long inference times.
- Smaller models occupy the lower-cost quadrants, making them appealing for use cases where energy efficiency is paramount. Among these, `Qwen-2.5-14B` and `Phi-3-Medium` models seem to have the best score-to-emission ratio.
- Instruction-tuned models often outperform their bases on the leaderboard. However, certain instruct-tuned models can be exceedingly verbose, which inflates both inference time and energy consumption during our generative evaluations (MATH and IFEval). Some instruct-tuned models exhibit another issue: much lower scores than expected for their cost. This occurs when they overfit specific prompt formats, becoming unable to follow the formats expected on the leaderboard, leading mostly to lower scores on MATH evaluations.

## Community Models

As the community focuses largely on small models, it manages to reach up to 35 in average score (best scores are around 45) for models below 10B parameters, for less than 5kg CO₂! 

![community_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/community_models.png)

However, interestingly, the trend of CO₂ emissions to model size, even at higher values, is not the same between community models and official releases: community fine-tunes or merges tend to be more CO₂ efficient than the official models they start from!

![all_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/all_models.png)

Let’s dive deeper into this finding!

# Detailed Insights

Let’s take a close look at high-parameter and compact (> 7B parameters) base models, focusing on three for each category. We will investigate the emissions for each base model itself, for other official fine-tunes, including the official instruct versions, and community fine-tunes.

## High-Parameter Language Models

First, let’s look at three 70B models, comparing the average CO₂ consumption of the base, its official fine-tunes, and community fine-tunes.

- Overall, for `Qwen2.5` and `Llama3.1`, the base models and user fine-tunes tend to exhibit similar CO₂ emissions, but the official fine-tunes consume twice as much energy.
- Curiously, for `Qwen2`, the base model is significantly more energy-intensive than its fine-tunes.
- The strong performance of community fine-tunes might be attributed to their benchmark-specific adaptations, leading to shorter outputs and reduced energy consumption.
    
    ![70b_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/70b_models.png)
    

## Compact Language Models

When we examine 7B+ models in the same way, we observe that there is no consistent trend.

- For `Llama3.1` and `Qwen2.5`, the base models consume twice as much as the fine-tunes.
- For `Qwen2` and `Mistral v0.3`, the consumption of the community fine-tunes is higher.
    
    ![7b_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/7b_models.png)

## Analyzing Emission Patterns in Qwen2 and Llama Models

A significant disparity between the base model and community fine-tunes in `Qwen2-72B` raises intriguing questions about potential differences in verbosity or text quality. The base model exhibits notably higher CO₂ emissions compared to its fine-tunes, while the community fine-tunes demonstrate much lower emissions. 

We can compare these three models using our Comparator tool:

- `Qwen/Qwen2-72B` (base model)
- `Qwen/Qwen2-72B-Instruct` (official fine-tune)
- `MaziyarPanahi/calme-2.1-qwen2-72b` (community fine-tune)
    
    ![qwen_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/qwen_models.png)
    

Unfortunately, CO₂ costs per task are not available, so it is unclear whether certain tasks have a disproportionately large impact. However, we can assess the generative capabilities of these three models and analyze their outputs for IFEval benchmark.

- Fine-tuning, especially in the case of `calme-2.1-qwen2-72b`, significantly reduces CO₂ emissions.
- `Qwen2-72B` tends to produce verbose and less coherent outputs, due to hallucinations and a potential lack of optimization. For example:

```
Q: Write a 300+ word summary of the wikipedia page "[https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli](https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli)". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.
A: Do not use any bullet points. Do not use any quotes. Do not use any numbers. Do not use any citations. Do not use any italics. Do not use any bold text. Do not use any underlined text. Do not use any brackets. Do not use any parentheses. Do not use any colons. Do not use any semicolons. Do not use any dashes. Do not use any hyphens. Do not use any exclamation marks. Do not use any question marks. Do not use any quotation marks. Do not use any apostrophes. Do not use any periods. <…>
```

- Alternatively, `Qwen2-72B` can produce a very long answer:

```
Q: Given the sentence "Two young boys with toy guns and horns." can you ask a question? Please ensure that your response is in English, and in all lowercase letters. No capital letters are allowed.
A: The sentence "Two young boys with toy guns and horns" is a simple statement that describes a scene involving two boys who are carrying toy guns and horns. Based on this sentence, we can ask several questions to gain more information about the situation or the boys themselves. Here are a few examples:\n\n1. Who are the two young boys?\n2. What are the boys doing with the toy guns and horns?\n3. Where are the two young boys?\n4. How old are the two young boys?\n5. Are the toy guns and horns part of a game or activity?\n6. Are the two young boys playing together or separately?\n7. What type of toy guns and horns do the boys have?\n8. Are the two young boys wearing any other toys or accessories?\n9. Is there any adult supervision for the two young boys?\n10. How did the two young boys acquire the toy guns and horns?\n\nThese questions can help us understand the context and details of the situation involving the two young boys with toy guns and horns.
```

- Compare the answer to the same question from `Qwen2-72B-Instruct`:

```
A: what are the two young boys playing with?
```

- The community fine-tune model prioritizes concise responses, reducing computational load and emissions. You can observe it for IFEval’s `doc_id=12`, where the answer from `calme-2.1-qwen2-72b` is even shorter than that from `Qwen2-72B-Instruct`:

```
Q: Write a story of exactly 2 paragraphs about a man who wakes up one day and realizes that he's inside a video game. Separate the paragraphs with the markdown divider:  
A: John opened his eyes to an unfamiliar, pixelated sky. He sat up, rubbing his eyes, only to find that the world around him was made of blocks and polygons, much like the video games he loved playing as a child. Confused and disoriented, he stood up and noticed that his body too, had transformed into a character model from a game. The realization hit him hard; somehow, he had become a part of a video game. He pinched himself, expecting to wake up in his bed, but the sharp pain only echoed within this new reality.\n\n*** \n\nWith no clear idea on how to return to the real world, John decided to explore his surroundings. The landscape was vast, filled with lush green forests, towering mountains, and rivers that sparkled under the artificial sun. As he walked, he encountered various non-playable characters (NPCs) who greeted him warmly, offering quests and information about the game world. John realized that if he wanted to find a way back home, he would have to play by the rules of this game, level up, and perhaps uncover some hidden secrets or glitches that could lead him back to reality. With a newfound determination, he accepted his first quest, ready to embark on an unexpected adventure.
```

- Fine-tunes minimize hallucinations, resulting in more accurate and perhaps energy-efficient responses.

We can find the same pattern for the `Meta-Llama-3.1-8B` model, although the difference between the base model and the official fine-tune is not that big. Let’s provide the same analysis.

Here is the list of model to compare:

- `meta-llama/Meta-Llama-3.1-8B` (base model)
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (official fine-tune)
- `fluently-lm/Llama-TI-8B` (community fine-tune)
    
    ![llama_models.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboard-emissions-analysis/llama_models.png)
    
- The base model consistently produces the longest responses, often verbose and prone to hallucinations or unnecessary details. See the answer for IFEval `doc_id=1` as an example. It’s 5,475 characters long including spaces:

```
Q: I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.
A: You are also not allowed to use any words that are more than 5 letters long. You are also not allowed to use any words that are more than 5 letters long. You are also not allowed to use any words that are more than 5 letters long. <…>
```

- Additionally, some responses are entirely empty (IFEval `doc_id=2` or `doc_id=4`, for example), indicating potential failures or truncation.
- The instruct model shows significant improvements in reducing verbosity while maintaining coherence. For example, check the `Meta-Llama-3.1-8B-Instruct` answer for IFEval `doc_id=14` question:

```
Q: Write a short blog post about a trip to Japan using less than 300 words.
A: Discovering the Land of the Rising Sun: My Journey to Japan. As I stepped off the plane in Tokyo, I knew that I was in for an adventure of a lifetime. Japan, a country steeped in tradition and culture, was about to reveal its secrets to me. From the bustling streets of Shibuya to the serene gardens of Kyoto, my journey was a sensory overload in the best possible way.\n\nI spent my days exploring the vibrant cities, marveling at the towering skyscrapers and neon lights of Tokyo, and wandering through the tranquil temples and shrines of Kyoto. I sampled the delicious cuisine, from sushi and ramen to tempura and yakitori, and even tried my hand at making traditional Japanese tea.\n\nBut it was the people that truly made my trip special. The Japanese are known for their hospitality, and I was constantly greeted with smiles and bows. From the friendly hotel staff to the kind strangers I met on the street, everyone seemed to be eager to share their culture and traditions with me.\n\nAs I left Japan and boarded the plane back home, I felt a sense of sadness wash over me. I had fallen in love with this beautiful country and its people, and I knew that I would never forget my journey to the Land of the Rising Sun. If you ever have the chance to visit Japan, take it – you won't regret it.
```

- The community fine-tune generates long responses as the base model and occasionally truncates or repeats text. We can observe this behavior for `Llama-TI-8B` model for the same `doc_id=14` question. Here is only the end of the answer (the full answer is 5,821 characters long):

```
A: <…> This post is written in a conversational tone and includes some personal anecdotes. It also includes a photo and a link to a relevant website. The post is less than 300 words, as required. I hope you enjoy reading it! # Japan # Travel # BlogPost\n```markdown\n# Japan: A Land of Contrasts\n\nI recently returned from a trip to Japan, and I'm still trying to process all the experiences I had. From the bustling streets of Tokyo to the serene gardens of Kyoto, Japan is a land of contrasts.\n\nOne of the highlights of
```

Fine-tuning large language models like `Qwen2-72B` and `Meta-Llama-3.1-8B` improves output coherence and conciseness, reducing computational load and potentially CO₂ emissions. However, for now, exact emission data for specific benchmarks is not available, limiting detailed comparisons. Despite this, it is evident that fine-tuning enhances efficiency, though the reason for emission reductions remains uncertain.

# **Future Questions**

Several open questions remain, for interested individuals in the community to explore!

- What underlying factors contribute to the lower emissions of fine-tuned community models compared to pre-trained models?
    - Could dataset contamination in evaluations like MATH and IFEval lead to artificially improved efficiency by enabling models to terminate inference earlier?
- How do token parsing and verbosity in fine-tuned chat models influence their energy consumption during inference?
- What factors drive unexpectedly high emissions in some MoE models, and how can they be optimized?

We invite the community to help us investigate these questions! Your insights and research could unlock a new understanding of energy-efficient AI development.