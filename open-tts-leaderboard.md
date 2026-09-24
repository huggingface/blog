---
title: "Open TTS Leaderboard: Scalable Evaluation for Multilingual Text-to-Speech and Voice Cloning" 
thumbnail: /blog/assets/open-tts-leaderboard/thumbnail.png
authors:
- user: bezzam
- user: Steveeeeeeen
- user: eustlb
- user: mrfakename
  guest: true
---


# Open TTS Leaderboard: Scalable Evaluation for Multilingual Text-to-Speech and Voice Cloning

The pace of open-source text-to-speech (TTS) model releases has been incredible. On the Hugging Face Hub (as of Sep 24, 2026) there are more than 7.9K TTS models available 🚀

**Evaluation, however, hasn't kept pace: it remains fragmented and unstandardized.** The gold standard is human preference scores such as MOS or MUSHRA (more on metrics). To this end, several arena-based leaderboards have established themselves as useful reference points for the community:

1. [TTS Arena v2](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2)
2. [Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice)
3. [Voice Arena](https://voicearena.com/tts-leaderboard)

These arenas compare models by presenting users with TTS outputs from two models, and asking them to choose one over the other. After collecting a sufficient number of votes, an [Elo score](https://en.wikipedia.org/wiki/Elo_rating_system) is computed to rank models, typically with the Bradley–Terry model (see [Voice Arena methodology](https://voicearena.com/tts-methodology)).

While human preference is the ultimate decider, **arenas cannot scale to keep up with the pace of TTS releases**. This may partly explain why open-source models are underrepresented on arena-style leaderboards: as of Sep 24, 2026, only 16 of the 92 models on [Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice) are open-weights, with a similar skew on [Voice Arena](https://voicearena.com/tts-leaderboard). This likely reflects practical factors: adding an API model requires little more than an API key, whereas an open model must be hosted and served by the arena operator, and commercial providers have more reason to seek placement than open-source authors. Moreover, no arena can ensure that the same voters with the same criteria of “better” can consistently evaluate models over time. Even the preferences of a single person change over time (“A man cannot step into the same river twice” as famously said by Heraclitus).

To this end, we've built the [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard), which uses objective metrics to evaluate models on complementary aspects of performance:

1. **Intelligibility**: word/character error rate (WER and CER) between the prompt and the generated audio's transcript, using Qwen3 ASR (top ranking open-source model on the Open ASR Leaderboard).
2. **Speed**: inverse real-time factor (RTFx) for batched offline inference on an H200 GPU, and time-to-first-audio (TTFA) for quantifying streaming batch size 1 latency on an H200 GPU and CPU.
3. **Speaker similarity** by computing the cosine similarity (SIM) between WavLM speaker embeddings of the generated audio and the reference clip.

By relying on objective metrics **evaluating a model drops from a couple weeks (for collecting votes) to a couple hours** ⚡

Importantly, the Open TTS Leaderboard does not replace human preference ranking. ASR-based WER provides a proxy for intelligibility, while speaker similarity estimates voice identity preservation. Neither directly measures naturalness, expressiveness, or listener preference. Nevertheless, they can even inform voting-based leaderboards which models to include in their evaluations.

The next few sections give an overview of main features of the Open TTS Leaderboard. Our intention with this leaderboard is for it to be **shaped by the community**; we want to hear your feedback so the evaluations stay relevant and insightful.

## Multilingual + voice cloning evaluation

From the default view of the leaderboard, models are ranked by macro-average WER on the English splits of [Seed TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval) ([paper](https://huggingface.co/papers/2406.02430)) and [CV3 Eval](https://github.com/QwenAudio/CV3-Eval) (zero shot) ([paper](https://huggingface.co/papers/2505.17589)).

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/english_table.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/english_pareto.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/english_bars.png" width="1024px" alt="thumbnail" />
</div>


[microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B), [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), and [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) lead the pack on English WER when averaged on these two splits, while the Pareto plots visualize which models strike a good balance between WER, batched inference (RTFx), and size.

English performance doesn't necessarily translate to other languages. Multiple languages can be toggled to rank models on multilingual performance. Seed TTS Eval only has audio for English and Chinese, so the other languages are simply the score on CV3 Eval (zero shot). Note that Chinese, Japanese, and Korean are character-based languages and so character error rate (CER) is reported, and the “Average WER” across languages is a macro-average across languages.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/multilingual.png" width="1024px" alt="thumbnail" />
</div>

[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro), and [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) are strong multilingual models.

By toggling “Voice cloning”, the models that support this functionality (on the selected languages) can be compared. 

Moreover, a SIM column for speaker similarity now appears in the table, as well as two more Pareto plots for visualizing the tradeoff between SIM, batched inference, and size.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/voice_clone_table.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/voice_clone_pareto.png" width="1024px" alt="thumbnail" />
</div>



## Compare and vote on TTS outputs

Numbers only tell part of the story, and as mentioned earlier **human preference is the ultimate decider**. From the “Listen” tab, you can compare the generated outputs that are behind the metrics, to find which model(s) you prefer! 

Pick the **language/dataset** you're interested in, whether you want to compare **voice cloning**, and optionally pick the models or listen to outputs from a random selection.

**The “Listen” tab fills an important gap in existing TTS leaderboards: a space to explore model outputs of various models.**

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/listen_tab.png" width="1024px" alt="thumbnail" />
</div>

You can even give feedback on the generated outputs. As we collect more votes from the community, we may include this data on the leaderboard. **So vote! But please login with your HF account to help us weed out spam/bots.**

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/listen_outputs.png" width="1024px" alt="thumbnail" />
</div>

## Evaluating streaming performance

The “Streaming” tab compares the streaming capabilities. Models are ranked by TTFA (time-to-first-audio), which quantifies how long a user waits after probing a model in order to obtain audio that can be played. This is important for voice agents and other interactive apps. 

For streaming models (✅ under “Streaming API”) it's the time until the first audio chunk arrives. For non-streaming models, it's the time until the whole utterance is generated, because playback can't start any earlier. Every model runs one audio at a time (batch size 1), on the same 50 English prompts from CV3-Eval, on the same hardware and in its default voice. We drop the first 3 runs as warm-up and report the median TTFA across the rest.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/streaming_table.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/streaming_bars.png" width="1024px" alt="thumbnail" />
</div>

The default view compares performance on an H200 GPU. Results are also available for CPU for a small (but growing) set of models!

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/streaming_cpu.png" width="1024px" alt="thumbnail" />
</div>

[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) is a great model for streaming on both GPU and CPU!


## Conclusion
The goal of the Open TTS Leaderboard is not only to keep up with the incredible pace of TTS model releases, but to be shaped by the community; we want to hear your feedback so the evaluations stay relevant and insightful. Let us know which datasets, models, and metrics you want to see! 

For now, we've focused on:
1. **Open-source models**, to put forward many great models that have been neglected by arena-style evaluations.
2. **Multilingual**, since English performance is not a suitable proxy for other languages.

We will soon open-source the evaluation scripts, much like the Open ASR Leaderboard [repo](https://github.com/huggingface/open_asr_leaderboard), so that you can directly provide your feedback and suggestions via GitHub Issues and PRs! Let's shape TTS evaluations together 🤗
