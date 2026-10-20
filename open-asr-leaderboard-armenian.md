---
title: "Armenian Joins the Open ASR Leaderboard"
thumbnail: /blog/assets/open-asr-leaderboard-armenian/thumbnail.png
authors:
- user: bezzam
- user: mariam-av
  guest: true
  org: Metric-AI
- user: alexshah
  guest: true
  org: Metric-AI
- user: Hrant
  guest: true
  org: Metric-AI
---

# Armenian Joins the Open ASR Leaderboard

*Metric AI and Hugging Face add Armenian 🇦🇲 to the Multilingual tab of the Open ASR Leaderboard*

The [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) lets you compare speech recognition systems on the same data with the same scoring. Its Multilingual tab started with European languages, and [Hindi recently joined it](https://huggingface.co/blog/open-asr-leaderboard-global-south). Today we are adding **Armenian**.

Armenian is spoken by millions of people in Armenia and across a large diaspora. It has its own alphabet, its own punctuation and rich morphology, and there is not much public speech data for it. Multilingual models usually list Armenian as supported, but it is hard to find a fair, side-by-side comparison of how well they actually transcribe it.

This release covers:

1. Two public Armenian test sets, [FLEURS](https://huggingface.co/datasets/google/fleurs) (with reviewed reference corrections) and Common Voice 26, hosted in a leaderboard-ready format at [Metric-AI/open-asr-leaderboard-multilingual-datasets](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets).
2. A dedicated Armenian text normalizer in the [Open ASR Leaderboard repository](https://github.com/huggingface/open_asr_leaderboard), so that formatting differences are not counted as recognition errors.
3. Results for 17 open and closed models, available on the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

## Why Armenian needs its own normalizer

WER is only as good as the text normalization applied before scoring. The leaderboard's basic multilingual normalizer works well for languages written in Latin script with European punctuation, but Armenian breaks several of its assumptions. If normalization is wrong, the leaderboard ends up comparing formatting habits, not recognition quality.

A few examples of what makes Armenian different:

- **Its own punctuation.** Armenian has its own full stop, comma and quotation marks. Its question, exclamation and emphasis marks are written *inside* the word, on the stressed vowel, rather than at the end of the sentence. Models and annotators also often use similar-looking Latin characters instead of the Armenian ones.
- **Numbers with endings.** Armenian is highly inflected, and numbers take grammatical endings just like words. The same year or percentage can be written with digits, with words, or as a mix of both. A model should not win or lose points depending on which of these forms it happens to choose.
- **Mixed scripts.** Armenian speech often includes English and Russian names and terms, so Latin and Cyrillic words need to be kept rather than removed.

The Armenian normalizer maps all of these variants to a single canonical form before scoring. It is applied automatically to Armenian in the multilingual evaluation scripts, so every model on the leaderboard is scored the same way.

## The test sets

Both sets are Eastern Armenian read speech, from the test splits of established public datasets:

| Dataset | Source | Clips | Duration | Clip length (mean / median) | Style | Transcriptions |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| [FLEURS](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/fleurs_hy) | `google/fleurs`, `hy_am`, test split | 932 | 3.0 h | 11.6 s / 10.8 s | Read, Wikipedia sentences | Punctuated, cased; reviewed corrections applied |
| [MCV](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/mcv_hy) | Common Voice Scripted Speech 26.0, `hy-AM`, test split | 6,892 | 10.8 h | 5.6 s / 5.5 s | Read, crowd-sourced | Punctuated, cased |

**FLEURS with corrected references.** A benchmark is only as good as its references. We manually reviewed the Armenian FLEURS test transcripts and found a significant number of errors, including typos, missing words and text that did not match the audio. All 314 reviewed corrections, out of 932 recordings, are published in [Metric-AI/fleurs-corrections](https://huggingface.co/datasets/Metric-AI/fleurs-corrections) and applied to the leaderboard version. This way models are not penalized for correctly transcribing what was actually said.

**Common Voice 26.** We use the version 26.0 Armenian test split of Mozilla Common Voice, converted to mono 16 kHz audio. The clips are short and recorded by volunteers on their own microphones, which gives more variety in speakers and recording conditions than FLEURS.

## Results

The full Armenian results for all 17 models are on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), in the **🌍 Multilingual** tab (see [below](#getting-evaluated) for how to select Armenian).

Some observations:

- **A small dedicated model is still competitive.** NVIDIA's Armenian [FastConformer](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) has about 115M parameters, is by far the fastest model evaluated, and is among the most accurate. It is also the only model that scores better on MCV than on FLEURS. Its model card lists Common Voice among its training data, so MCV is in-domain for it, and its MCV score should be read with that in mind.
- **Multilingual open models have improved a lot for Armenian.** The [Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B-v2) LLM variants are close to the best systems overall, while `whisper-large-v3` is above 40% WER on both sets.
- **An LLM decoder helps most at small sizes, but costs speed.** At 300M parameters, the LLM variant of Omnilingual ASR reduces FLEURS WER from 18.13 to 7.04 compared to the CTC variant. At 7B, the gap is much smaller (5.88 vs 3.61). CTC models are 7–23 times faster, so the right choice depends on whether accuracy or throughput matters more.
- **MCV is harder than FLEURS for almost every model.** The clips are shorter (5.6 s vs 11.6 s on average), which gives the model less context, and they are recorded by many volunteers on many different microphones. Systems that are close on FLEURS spread out more on MCV.
- **"Supported" does not mean "usable."** Several widely used systems produce WER close to 50% on Armenian. If you are building an Armenian product, measure first instead of relying on a supported-languages list.

## Limitations

These two test sets are a first step, not a full picture of Armenian ASR.

- **Read speech only.** FLEURS and Common Voice are both read speech, and ASR results usually depend heavily on the domain: a model that does well on read speech may not do well on conversational or noisy audio. Low WER here does not mean low WER on call-center audio, interviews or spontaneous conversation.
- **Eastern Armenian only.** Western Armenian, spoken mostly in the diaspora, and regional dialects are not covered. Many dialects do not have a single accepted written form, which makes it hard to create references and score against them.
- **Normalization has its limits.** The normalizer removes formatting differences, but it cannot handle every valid spelling variant or every way of writing a name in another script.

## Getting evaluated

To add a model to the leaderboard, follow the usual process on the [Open ASR Leaderboard GitHub](https://github.com/huggingface/open_asr_leaderboard):

1. Open a pull request. A [model checklist](https://github.com/huggingface/open_asr_leaderboard/blob/main/.github/PULL_REQUEST_TEMPLATE.md#new-model-checklist) will appear. Include results on the Armenian sets if your model supports the language. The multilingual evaluation scripts already load the Armenian datasets and the normalizer when the language is set to `hy`.
2. We will verify the results.
3. Confirm the results we obtained.

On the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), Armenian is in the **🌍 Multilingual** tab. Like English, it is not selected by default under "Languages", so the default multilingual ranking does not change for existing models. Check "Armenian" to include it in the average, or choose it from the **Language dataset breakdown** dropdown to see the FLEURS and MCV columns separately.

## What comes next

Adding Armenian to the Open ASR Leaderboard makes it visible next to other languages, on a leaderboard model developers already track. We hope this encourages more models to be evaluated on Armenian and makes improvements easier to measure.

Next, we want to add more realistic Armenian test data: spontaneous and conversational speech, code-switching between Armenian, English and Russian, and domain-specific audio. For a deeper look at Armenian ASR across more models and domains, see [ArmBench-ASR](https://huggingface.co/blog/Metric-AI/armbench-asr), a dedicated Armenian benchmark.
