---
title: "Egyptian and Nigerian English join the Open ASR Leaderboard"
thumbnail: /blog/assets/open-asr-leaderboard-liva/thumbnail.png
authors:
- user: bezzam
- user: ashleymo2025
  guest: true
  org: liva-ai
- user: aonorii
  guest: true
  org: liva-ai
---

# Egyptian and Nigerian English join the Open ASR Leaderboard

A few months ago, we added private data to the Open ASR Leaderboard: 11 splits from Appen and DataOceanAI covering scripted and conversational speech from five accents (blog). The community response was great, with many **highlighting the importance of held-out sets for trustworthy evaluations**. Since then, we added one more private split for Indian-accented English with Voice Arena (blog).

We’re excited to announce a new collaboration with [Liva AI](https://huggingface.co/liva-ai) to add even more private data with two new accents: **Egyptian** 🇪🇬 and **Nigerian** 🇳🇬.

| Dataset | Accent | Duration [h] | Style | Transcription |
| --- | --- | --- |  --- | --- |
| Liva AI Conversational EG | Egyptian | 1.34 | Conversational, spontaneous | Punctuated, cased, with disfluencies. |
| Liva AI Conversational NG | Nigerian | 1.55 | Conversational, spontaneous | Punctuated, cased, with disfluencies. |

This is part of our continued effort to improve representation in ASR evals (more accents and more languages) and to improve trust (new private data to minimize benchmaxxing, and new data providers). We’re excited to see the community continuing to push ASR performance for the world 🤗

## Richer accent breakdown in the Private tab

With these new accents, we can now present a richer breakdown of accents, based on [Kachru’s Three Circles](https://en.wikipedia.org/wiki/World_Englishes#Kachru%27s_Three_Circles_of_English) (kudos to Hume AI for this idea in their [leaderboard](https://huggingface.co/spaces/HumeAI/asr-leaderboard)). The Open ASR Leaderboard will now report:
1. US vs. non-US (*UK, Australia, Canada, India, Nigeria, Egypt*)
2. Inner Circle (*US, UK, Australia, Canada*) vs. Outer & Expanding Circles (*India, Nigeria, Egypt*)

As more accents join, we’ll split the Outer and Expanding Circles.

<div align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/private_data_accent_breakdown.png" width="1024px" alt="thumbnail" />
</div>

## About Liva AI

Liva AI collects multimodal interaction data through consumer products used by real people. Because Liva owns both the collection environment and distribution, it can design products around specific model gaps across domains, industries, languages, modalities, and interaction types. This gives it access to natural, high-context behavior that emerges through real product use, rather than artificially elicited or task-based interactions.

