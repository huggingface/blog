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

Armenian is spoken by millions of people in Armenia and across a large diaspora. It has its own alphabet, its own punctuation and rich morphology, and there is not much public speech data for it. Multilingual models usually list Armenian as supported, but it is hard to find a fair, side-by-side comparison of how well they actually transcribe it. Earlier this year we released [ArmBench-ASR](https://huggingface.co/blog/Metric-AI/armbench-asr), a dedicated Armenian benchmark. This post brings what we learned there to a leaderboard the community already follows.

This release covers:

1. Two public Armenian test sets, [FLEURS](https://huggingface.co/datasets/google/fleurs) (with reviewed reference corrections) and [Common Voice 26](https://mozilladatacollective.com/datasets/cmqinsp2y00xsnr07ra8gxfoi), hosted in a leaderboard-ready format at [Metric-AI/open-asr-leaderboard-multilingual-datasets](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets).
2. A dedicated `ArmenianTextNormalizer` in the [Open ASR Leaderboard repository](https://github.com/huggingface/open_asr_leaderboard), so that formatting differences are not counted as recognition errors.
3. Results for 17 open and closed models.

## Why Armenian needs its own normalizer

WER is only as good as the text normalization applied before scoring. The leaderboard's basic multilingual normalizer works well for languages written in Latin script with European punctuation. Armenian breaks several of its assumptions. In ArmBench-ASR, the gap between strict and normalized scores showed that punctuation, capitalization, spacing and spelling conventions make up a large share of what looks like recognition errors. If normalization is wrong, the leaderboard ends up comparing formatting habits, not recognition quality.

These are the main Armenian-specific issues:

- **Punctuation marks inside words.** Armenian has its own full stop (`։`), comma (`՝`) and quotation marks (`« »`). Its question (`՞`), exclamation (`՜`) and emphasis (`՛`) marks are placed *on* the stressed vowel inside the word, not at the end of the sentence. If you treat them as ordinary punctuation and split on them, one word turns into two. If you keep them, a correct word with a missing intonation mark counts as an error.
- **Look-alike characters.** Models and annotators use ASCII lookalikes interchangeably with Armenian punctuation: `:` instead of `։`, a backtick or apostrophe instead of `՝` or `՛`, and many different dashes and hyphens. Each of these needs to map to one canonical form.
- **Ligatures.** The conjunction "and" is written with the ligature `և`, which some systems output as the two letters `եվ`. Both should count as the same word.
- **Numbers with suffixes.** Armenian is highly inflected, and numbers carry case endings: a year can be written as `2020-ին` or `2020 թ.`, or spelled out in words with the suffix attached. "Five percent", "5%" and "5 %-ով" all have to resolve to the same tokens. Without this, a model that writes numbers as digits loses to one that spells them out (or the other way round) for reasons unrelated to what it heard.
- **Mixed scripts.** Armenian speech regularly includes English and Russian names and terms. The normalizer keeps Armenian, Latin and Cyrillic letters, so a correctly transcribed foreign name is not silently deleted.

The `ArmenianTextNormalizer` applies Unicode NFKC, standardizes Armenian punctuation, quotes, apostrophes and dashes, lowercases, removes punctuation (while keeping clock times like `10։30` intact), and maps written and spoken forms of numbers to a single canonical representation. It is used automatically for `hy` in the multilingual evaluation scripts, so every model on the leaderboard is scored the same way.

```python
from normalizer import ArmenianTextNormalizer

normalizer = ArmenianTextNormalizer()
normalizer(reference)  # applied identically to references and predictions
```

## The test sets

Both sets are Eastern Armenian read speech, from the test splits of established public datasets:

| Dataset | Source | Clips | Duration | Clip length (mean / median) | Style | Transcriptions |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| [FLEURS](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/fleurs_hy) | `google/fleurs`, `hy_am`, test split | 932 | 3.0 h | 11.6 s / 10.8 s | Read, Wikipedia sentences | Punctuated, cased; reviewed corrections applied |
| [MCV](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/mcv_hy) | Common Voice Scripted Speech 26.0, `hy-AM`, test split | 6,892 | 10.8 h | 5.6 s / 5.5 s | Read, crowd-sourced | Punctuated, cased |

**FLEURS with corrected references.** A benchmark is only as good as its references. While building ArmBench-ASR, we manually reviewed the Armenian FLEURS test transcripts and found a significant number of errors, including typos, missing words and text that did not match the audio. All 314 reviewed corrections, out of 932 recordings, are published in [Metric-AI/fleurs-corrections](https://huggingface.co/datasets/Metric-AI/fleurs-corrections) and applied to the leaderboard version. This way models are not penalized for transcribing correctly what was actually said.

**Common Voice 26.** Common Voice is now distributed through the [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmqinsp2y00xsnr07ra8gxfoi). We use the version 26.0 Armenian test split, converted to mono 16 kHz audio. The clips are short and recorded by volunteers on their own microphones, which gives more variety in speakers and recording conditions than FLEURS.

## Results

The table below shows the leaderboard results for Armenian, sorted by the average of the two test sets (lower is better).

| Model | FLEURS WER | MCV WER | Average WER | RTFx |
| :--- | ---: | ---: | ---: | ---: |
| [elevenlabs/scribe_v2](https://elevenlabs.io/speech-to-text) | **3.22** | 7.60 | **5.41** | – |
| [nvidia/stt_hy_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) | 6.80 | **4.94** | 5.87 | **2879.4** |
| [facebook/omniASR-LLM-7B-v2](https://huggingface.co/facebook/omniASR-LLM-7B-v2) | 3.61 | 10.04 | 6.83 | 42.6 |
| [facebook/omniASR-LLM-3B-v2](https://huggingface.co/facebook/omniASR-LLM-3B-v2) | 4.38 | 9.48 | 6.93 | 45.2 |
| [facebook/omniASR-LLM-1B-v2](https://huggingface.co/facebook/omniASR-LLM-1B-v2) | 4.74 | 9.98 | 7.36 | 46.8 |
| gemini/gemini-3.5-transcribe | 4.79 | 11.74 | 8.27 | – |
| [facebook/omniASR-CTC-7B-v2](https://huggingface.co/facebook/omniASR-CTC-7B-v2) | 5.88 | 12.04 | 8.96 | 310.9 |
| [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | 6.36 | 11.61 | 8.99 | 91.6 |
| [facebook/omniASR-CTC-3B-v2](https://huggingface.co/facebook/omniASR-CTC-3B-v2) | 6.49 | 12.00 | 9.25 | 527.6 |
| [facebook/omniASR-LLM-300M-v2](https://huggingface.co/facebook/omniASR-LLM-300M-v2) | 7.04 | 12.05 | 9.55 | 47.7 |
| openai/gpt-transcribe | 7.23 | 12.21 | 9.72 | – |
| [facebook/omniASR-CTC-1B-v2](https://huggingface.co/facebook/omniASR-CTC-1B-v2) | 9.55 | 16.83 | 13.19 | 763.9 |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 13.43 | 26.57 | 20.00 | 973.6 |
| [facebook/omniASR-CTC-300M-v2](https://huggingface.co/facebook/omniASR-CTC-300M-v2) | 18.13 | 27.79 | 22.96 | 1084.2 |
| assemblyai/universal-3-5-pro | 45.23 | 50.38 | 47.81 | – |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 43.03 | 57.04 | 50.04 | 73.4 |
| modulate/multilingual | 43.84 | 57.43 | 50.64 | – |

RTFx is not reported for API-based systems, since their speed depends on the provider's infrastructure rather than the model alone.

Some observations:

- **A small dedicated model is still competitive.** NVIDIA's Armenian FastConformer has about 115M parameters, is the fastest model in the table by a wide margin, and ranks second overall. It is also the only model that scores better on MCV than on FLEURS. Its model card lists Common Voice among its training data, so MCV is in-domain for it, and its MCV score should be read with that in mind. On FLEURS it is behind several larger multilingual models. ArmBench-ASR showed the same pattern: on Common Voice, Armenian-specific models were among the best systems, ahead of much larger general ones.
- **Multilingual open models have improved a lot for Armenian.** The Omnilingual ASR LLM variants are within 2 points of the best average, and `omniASR-LLM-7B-v2` has the best open-model FLEURS score (3.61). For comparison, `whisper-large-v3` is above 40% WER on both sets.
- **An LLM decoder helps most at small sizes, but costs speed.** At 300M parameters, the LLM variant of Omnilingual ASR reduces FLEURS WER from 18.13 to 7.04 compared to the CTC variant. At 7B, the gap is much smaller (5.88 vs 3.61). CTC models are 7–23 times faster, so the right choice depends on whether accuracy or throughput matters more.
- **MCV is harder than FLEURS for almost every model.** The clips are shorter (5.6 s vs 11.6 s on average), which gives the model less context, and they are recorded by many volunteers on many different microphones. Systems that are close on FLEURS spread out more on MCV.
- **"Supported" does not mean "usable."** Several widely used systems produce WER close to 50% on Armenian. If you are building an Armenian product, measure first instead of relying on a supported-languages list.

## Limitations

These two test sets are a first step, not a full picture of Armenian ASR.

- **Read speech only.** FLEURS and Common Voice are both read speech. In ArmBench-ASR, results depended heavily on the domain: every model performed much worse on conversational speech from films than on read speech, and a model that does well on one domain may not do well on another. Low WER here does not mean low WER on call-center audio, interviews or spontaneous conversation.
- **Eastern Armenian only.** Western Armenian, spoken mostly in the diaspora, and regional dialects are not covered. Many dialects do not have a single accepted written form, which makes it hard to create references and score against them.
- **Normalization has its limits.** The normalizer removes formatting differences, but it cannot handle every valid spelling variant or every way of writing a name in another script. Strict and normalized scores should not be compared with each other.

## Getting evaluated

To add a model to the leaderboard, follow the usual process on the [Open ASR Leaderboard GitHub](https://github.com/huggingface/open_asr_leaderboard):

1. Open a pull request. A [model checklist](https://github.com/huggingface/open_asr_leaderboard/blob/main/.github/PULL_REQUEST_TEMPLATE.md#new-model-checklist) will appear. Include results on the Armenian sets if your model supports the language. The multilingual evaluation scripts already load the Armenian datasets and the normalizer when the language is set to `hy`.
2. We will verify the results.
3. Confirm the results we obtained.

On the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), Armenian is in the **🌍 Multilingual** tab. Like English, it is not selected by default under "Languages", so the default multilingual ranking does not change for existing models. Check "Armenian" to include it in the average, or choose it from the **Language dataset breakdown** dropdown to see the FLEURS and MCV columns separately.

## What comes next

Adding Armenian to the Open ASR Leaderboard makes it visible next to other languages, on a leaderboard model developers already track. We hope this encourages more models to be evaluated on Armenian and makes improvements easier to measure.

Next, we want to add more realistic Armenian test data: spontaneous and conversational speech, code-switching between Armenian, English and Russian, and domain-specific audio. For a broader view of Armenian ASR today, including more models and domains, see [ArmBench-ASR](https://huggingface.co/blog/Metric-AI/armbench-asr) and its [interactive leaderboard](https://metric-ai-armbench-asr.static.hf.space/).
