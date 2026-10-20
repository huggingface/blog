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

Armenian is spoken by millions of people in Armenia and across a large diaspora, yet there is not much public speech data for it, and few fair side-by-side comparisons of how well ASR models transcribe it. Today, Armenian joins the **🌍 Multilingual** tab of the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), with:

- two public test sets, FLEURS (with corrected references) and Common Voice 26, in a [leaderboard-ready format](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets);
- a dedicated Armenian text normalizer in the [evaluation code](https://github.com/huggingface/open_asr_leaderboard);
- results for 17 open and closed models, available on the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

This post explains what makes Armenian hard to evaluate and what we did about it.

## What makes Armenian different

Armenian has its own alphabet, its own punctuation and rich morphology: words, and even numbers, take grammatical endings. The language also has two standard forms, Eastern and Western, and everyday speech differs noticeably from the written language. Speakers also mix in English and Russian words all the time.

None of this is a problem for a listener, but each of these points creates differences between a correct transcript and its reference text. A word-level metric like WER counts every one of those differences as an error.

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/armenian-features.png" alt="Five features of Armenian that affect ASR scoring, with examples" width="90%">
</p>

## Armenian-specific normalization

Before scoring, the leaderboard normalizes both the reference and the prediction, so that only real recognition errors are counted. The generic multilingual normalizer was designed for Latin-script European languages and handles Armenian poorly, so we added a dedicated one. It:

- maps Armenian punctuation, and the Latin look-alikes that models often produce, to a single form before removing it, including the marks written inside words;
- treats the two spellings of "and" (one ligature letter or two letters) as the same word;
- converts numbers, years, percentages and clock times to one canonical form, whether they are written in digits or words, and with or without grammatical endings;
- keeps Latin and Cyrillic words instead of deleting them.

The effect is large. In each example below, the prediction is a correct transcription, and only its formatting differs from the reference:

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/normalization-effect.png" alt="Strict vs normalized WER on six correctly transcribed sentences" width="90%">
</p>

Without normalization, a model that writes "five percent" in words instead of "5%" is penalized for reasons unrelated to what it heard. Across a whole test set, these formatting differences can easily outweigh the actual recognition differences between models.

## Dataset diversity

Both public test sets are Eastern Armenian read speech:

| Dataset | Clips | Duration | Style | Notes |
| :--- | ---: | ---: | :--- | :--- |
| [FLEURS](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/fleurs_hy) | 932 | 3.0 h | Read Wikipedia sentences | 314 references [manually corrected](https://huggingface.co/datasets/Metric-AI/fleurs-corrections) |
| [Common Voice 26](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/mcv_hy) | 6,892 | 10.8 h | Crowd-sourced read speech | Short clips, many speakers and microphones |

Even these two similar sets rank models differently. NVIDIA's small Armenian [FastConformer](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) is 9th on FLEURS but 1st on Common Voice. Its model card lists Common Voice in its training data, so that set is in-domain for it. Most other models find Common Voice harder: its clips are half as long, which gives less context, and they are recorded on many different devices.

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/fleurs-vs-mcv.png" alt="FLEURS vs Common Voice WER for the top 12 models, with rank on each set" width="90%">
</p>

Real Armenian audio is much more varied than read speech. Our private Armenian test sets cover domains that public data does not: <!-- TODO: confirm with the team how much can be disclosed about the private sets --> expressive literary recitation with background music, film and TV dialogue with background noise, and news narration. On those sets, every model we tested has much higher error rates than on read speech, and a model that leads on one domain does not necessarily lead on another. The public sets are a useful, reproducible baseline, but a low WER on them does not guarantee a low WER on conversations, call-center audio or broadcast media.

## Challenges with conventional WER

Normalization fixes formatting, but some differences are not formatting. Everyday speech uses colloquial forms that references usually write in the literary form. A foreign word can be correctly written in Latin or in Armenian letters. Some words have more than one accepted spelling. A rule-based normalizer cannot safely decide which of these are acceptable, because the same rule would also hide real errors.

One option is **LLM-assisted scoring**: after normalization, an LLM reviews each remaining mismatch against the reference and decides whether it is an acceptable variant or a real error. Paraphrases and wrong words still count, and the final score is still computed from the edit counts, not by the LLM itself.

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/wer-layers.png" alt="One sentence scored with strict, normalized and LLM-assisted WER" width="90%">
</p>

The Open ASR Leaderboard reports normalized WER, which is fully deterministic and reproducible. LLM-assisted scores, along with more models and domains, are available in [ArmBench-ASR](https://huggingface.co/blog/Metric-AI/armbench-asr), our dedicated Armenian ASR leaderboard.

## Getting evaluated

On the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), open the **🌍 Multilingual** tab. Armenian is not selected by default, so check "Armenian" under "Languages", or choose it from the **Language dataset breakdown** dropdown to see each test set separately.

To add a model, open a pull request on the [Open ASR Leaderboard GitHub](https://github.com/huggingface/open_asr_leaderboard) and follow the [model checklist](https://github.com/huggingface/open_asr_leaderboard/blob/main/.github/PULL_REQUEST_TEMPLATE.md#new-model-checklist). The multilingual evaluation scripts load the Armenian test sets and normalizer automatically when the language is set to `hy`.

## What comes next

We want to bring more realistic Armenian audio to the leaderboard: spontaneous conversation, code-switching between Armenian, English and Russian, domain-specific speech, and eventually Western Armenian. We hope having Armenian next to other languages on a leaderboard the community already follows encourages more models to support it well, and makes their progress easier to measure.
