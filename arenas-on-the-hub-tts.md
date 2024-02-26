---
title: "TTS Arena: Benchmarking TTS Models in the Wild"
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

# TTS Arena: Benchmarking TTS Models in the Wild

Automated measurement of the quality of text-to-speech (TTS) models is quite difficult. Although testing the naturalness and inflection of a voice is a trivial task for humans, it is much more difficult for AI. This is why today, we’re thrilled to announce the TTS Arena. Inspired by LMSys's Chatbot Arena for LLMs, we developed a tool that allows individuals to easily compare TTS models side-by-side. Just submit text, listen to two models speak it, and vote on which model is the best. The results are organized into a leaderboard that displays the community’s highest-rated models.

## Motivation
The field of speech synthesis has long lacked an accurate method to measure the quality of different models. Objective metrics like WER (word error rate) are unreliable measures of model quality, and subjective measures such as MOS (mean opinion score) are typically small-scale experiments conducted with few listeners. As a result, these measurements are generally not useful for comparing two models of roughly similar quality. To address these drawbacks, we are inviting the community to rank models in an easy-to-use interface, and opening it up to the public in order to make both the opportunity to rank models, as well as the results, more easily accessible to everyone.

## The Arena
Human ranking for AI systems is not a novel approach. Recently, LMSys applied this method in their [Chatbot Arena](https://arena.lmsys.org/) with positive results, resulting in over 300,000 rankings. Because of its success, we adopted a similar framework for our leaderboard, inviting humans to rank synthesized audio.

The voting system is an important aspect of the leaderboard. Due to the risks of human bias and abuse, model anonymity is key. This is why we reveal model names only after a user submits their vote.

## The Models
We selected several SOTA (State of the Art) models for our leaderboard. While most are open source models, we also included several proprietary models to allow developers to compare the state of open source development with proprietary models.

The models available at launch are:
- ElevenLabs (proprietary)
- MetaVoice
- OpenVoice
- OpenAI (proprietary)
- Pheme
- WhisperSpeech
- XTTS

Although there are many other open and closed source models available, we chose these because they are generally accepted as the highest-quality publicly available models.

## The Leaderboard
We will display the results from Arena voting in a publicly accessible leaderboard. Although it will initially be empty, models will gradually appear as they accumulate enough votes. As raters submit new votes, the leaderboard will automatically update with the latest results.

Similar to the Chatbot Arena, the models on the leaderboard are ranked using the [Elo rating](https://en.wikipedia.org/wiki/Elo_rating_system) system, a system commonly used by game players.



## Conclusion
We hope the TTS Arena proves to be a helpful resource for all developers. We would love to hear your feedback. Please also do not hesitate to let us know if you have any questions or suggestions by sending an X/Twitter DM [here](https://twitter.com/realmrfakename) or by opening a discussion in [the Space](https://huggingface.co/spaces/ttseval/TTS-Arena).

## Credits

Special thanks to all the people who helped make this possible, including [Clémentine Fourrier](https://twitter.com/clefourrier), [Lucian Pouget](https://twitter.com/wauplin), [Yoach Lacombe](https://twitter.com/yoachlacombe), [Main Horse](https://twitter.com/main_horse), and the Hugging Face team. In particular, I’d like to thank [VB](https://twitter.com/reach_vb) for his time and technical assistance. I’d also like to thank [Sanchit Gandhi](https://twitter.com/sanchitgandhi99) and [Apolinário Passos](https://twitter.com/multimodalart) for their feedback and support during the development process.
