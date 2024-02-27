---
title: "TTS Arena: Benchmarking Text-to-Speech Models in the Wild"
thumbnail: /blog/assets/arenas-on-the-hub/thumbnail_tts.png 
authors:
- user: mrfakename
  guest: true
- user: reach-vb
- user: clefourrier
- user: Wauplin
- user: ylacombe
- user: main-horse
  guest: true
- user: sanchit-gandhi
---

# TTS Arena: Benchmarking Text-to-Speech Models in the Wild

Automated measurement of the quality of text-to-speech (TTS) models is very difficult. Assessing the naturalness and inflection of a voice is a trivial task for humans, but it is much more difficult for AI. This is why today, we’re thrilled to announce the TTS Arena. Inspired by [LMSys](https://lmsys.org/)'s [Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) for LLMs, we developed a tool that allows anyone to easily compare TTS models side-by-side. Just submit some text, listen to two different models speak it out, and vote on which model you think is the best. The results will be organized into a leaderboard that displays the community’s highest-rated models.

## Motivation

The field of speech synthesis has long lacked an accurate method to measure the quality of different models. Objective metrics like WER (word error rate) are unreliable measures of model quality, and subjective measures such as MOS (mean opinion score) are typically small-scale experiments conducted with few listeners. As a result, these measurements are generally not useful for comparing two models of roughly similar quality. To address these drawbacks, we are inviting the community to rank models in an easy-to-use interface. By opening this tool and disseminating results to the public, we aim to democratize how models are ranked and to make model comparison and selection accessible to everyone.

## The TTS Arena

Human ranking for AI systems is not a novel approach. Recently, LMSys applied this method in their [Chatbot Arena](https://arena.lmsys.org/) with great results, collecting over 300,000 rankings so far. Because of its success, we adopted a similar framework for our leaderboard, inviting any person to rank synthesized audio.

The leaderboard allows a user to enter text, which will be synthesized by two models. After listening to each sample, the user will vote on which model sounds more natural. Due to the risks of human bias and abuse, model names will be revealed only after a vote is submitted.

## Selected Models

We selected several SOTA (State of the Art) models for our leaderboard. While most are open-source models, we also included several proprietary models to allow developers to compare the state of open-source development with proprietary models.

The models available at launch are:
- ElevenLabs (proprietary)
- MetaVoice
- OpenVoice
- OpenAI (proprietary)
- Pheme
- WhisperSpeech
- XTTS

Although there are many other open and closed source models available, we chose these because they are generally accepted as the highest-quality publicly available models.

## The TTS Leaderboard

The results from Arena voting will be made publicly available in a dedicated leaderboard. Note that it will be initially empty until sufficient votes are accumulated, then models will gradually appear. As raters submit new votes, the leaderboard will automatically update.

Similar to the Chatbot Arena, models will be ranked using an algorithm similar to the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system), commonly used in chess and other games.



## Conclusion

We hope the [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) proves to be a helpful resource for all developers. We'd love to hear your feedback! Please do not hesitate to let us know if you have any questions or suggestions by sending us an [X/Twitter DM](https://twitter.com/realmrfakename), or by opening a discussion in [the community tab of the Space](https://huggingface.co/spaces/TTS-AGI/TTS-Arena/discussions).

## Credits

Special thanks to all the people who helped make this possible, including [Clémentine Fourrier](https://twitter.com/clefourrier), [Lucian Pouget](https://twitter.com/wauplin), [Yoach Lacombe](https://twitter.com/yoachlacombe), [Main Horse](https://twitter.com/main_horse), and the Hugging Face team. In particular, I’d like to thank [VB](https://twitter.com/reach_vb) for his time and technical assistance. I’d also like to thank [Sanchit Gandhi](https://twitter.com/sanchitgandhi99) and [Apolinário Passos](https://twitter.com/multimodalart) for their feedback and support during the development process.
