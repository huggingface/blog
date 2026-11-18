---
title: "Rumble-2-Flash: an open 0.5B speech model for 40+ languages"
thumbnail: /blog/assets/rumble-2-flash/thumbnail.png
authors:
- user: reson8-thomas
  guest: true
  org: Reson8
- user: MarktHart
  guest: true
  org: Reson8
---

# Rumble-2-Flash: an open 0.5B speech model for 40+ languages

Today we're open sourcing [Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash) under the Apache 2.0 license. It's a 0.5B-parameter speech recognition model for 40+ languages that competes with models 3-60x its size.

- **Runs anywhere**: native support in 🤗 Transformers and vLLM, no custom code
- **40+ languages**: automatic language detection
- **Output control**: numbers as digits or words, fillers, verbatim repeats, forced language
- **Extremely fast**: an hour of audio in about half a second, with over 7,000 RTFx on full batches
- **Holds up on real phone calls**: on par with NVIDIA's Nemotron 3.5 Transcribe (30B) on a 302-hour call-centre benchmark

## Quickstart

<details>
<summary>Transcribe with 🤗 Transformers</summary>

```bash
uv pip install torch transformers accelerate
```

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = "Reson8/Rumble-2-Flash"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(
    audio="https://hf.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    language="English",  # omit to detect automatically
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512)
generated = output[:, inputs["input_ids"].shape[-1]:]
print(processor.decode(generated, return_format="transcription_only")[0])  # return_format="parsed" if detecting language
```

`audio` accepts a local path, a URL, a NumPy array, or a list of these.

</details>

## How it's built

Rumble-2-Flash uses the Qwen3-ASR architecture for day-0 support. If your stack runs Transformers or vLLM, it runs Rumble-2-Flash, with no forked runtime or `trust_remote_code`.

To find out how small a model could get before accuracy dropped off, we distilled Rumble-2, our larger production model, into it with on-policy distillation. The student doesn't train on transcripts the teacher wrote. It transcribes the audio itself, and the teacher grades that transcript token by token: at every position, it gives its own probability for what should come next, and the student learns to match it. Because the student learns from its own outputs, it gets corrected on exactly the mistakes it makes, with feedback on every token instead of one score per clip.

On-policy distillation normally requires teacher and student to share a tokenizer, and ours don't: choosing Qwen3-ASR meant taking its vocabulary too. [GOLD](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) removes that restriction. It merges tokens where the two tokenizers split the text differently, so both models are compared over the same text. Tokens that exist in both vocabularies are compared directly, and the rest by their sorted probabilities.

<figure class="image text-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/gold.png" alt="On-policy distillation with GOLD: the student transcribes, the teacher scores every token, and GOLD aligns the two tokenizers">
  <img class="hidden dark:block" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/gold-dark.png" alt="On-policy distillation with GOLD: the student transcribes, the teacher scores every token, and GOLD aligns the two tokenizers">
  <figcaption>On-policy distillation with GOLD</figcaption>
</figure>

## Results

### Accuracy for its size

On the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) multilingual track, Rumble-2-Flash averages a **6.96% error rate** across Chinese, German, French, Italian, Spanish, Portuguese, Hindi, Dutch and English. That's better than Meta's omniASR-LLM 7B, a model 15x larger, and ahead of Whisper large-v3. On the English track it averages **6.6% WER**.

<figure class="image text-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/size-vs-wer-en.png" alt="Average error rate against model size for open-weight models on the Open ASR Leaderboard multilingual track, including English">
  <img class="hidden dark:block" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/size-vs-wer-en-dark.png" alt="Average error rate against model size for open-weight models on the Open ASR Leaderboard multilingual track, including English">
  <figcaption>The best score for its size</figcaption>
</figure>

### Speed

Rumble-2-Flash is also fast. Its throughput depends on how full the batches are: on the larger datasets in HF's Open ASR Leaderboard, where batches stay full, it reaches **over 7,000 RTFx**. That's an hour of audio in about half a second.

### Real phone calls

Clean audio is easy. Phone calls are compressed and noisy, and often have speakers with very different accents. On [AppTek's call-centre benchmark](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) (302 hours of helpline calls, 14 English accents, scored with AppTek's own script), Rumble-2-Flash reaches **7.86% WER**, on par with NVIDIA's Nemotron 3.5 Transcribe (30B) at 7.78%. That's well ahead of Qwen3-ASR 0.6B (8.7%), from the same model family, and close to Qwen3-ASR 1.7B (7.4%), the only model in the comparison that does better.

<figure class="image text-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/apptek.png" alt="WER on the AppTek call-centre benchmark">
  <img class="hidden dark:block" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/apptek-dark.png" alt="WER on the AppTek call-centre benchmark">
  <figcaption>Real phone calls, 14 English accents</figcaption>
</figure>

## Try it

The weights are on the Hub at [Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash). It's also available through our [API](https://www.reson8.dev).

We're very curious what you'll build with it. Tell us in the [Discussions tab](https://huggingface.co/Reson8/Rumble-2-Flash/discussions) what you've made, what you like, and what you'd like to see changed in a next version.
