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

Today we are adding Armenian to the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). This release brings:

1. **Armenian in the 🌍 Multilingual tab**, next to the other languages on the leaderboard, with results for 17 open and closed models.
2. **Two public test sets**, FLEURS (with corrected references) and Common Voice 26, in the leaderboard's [evaluation format](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets).
3. **Armenian-specific normalization**, so that the reported WER reflects differences in what models transcribed, rather than differences in how they format text.

This does not solve Armenian ASR evaluation. It gives it a public, reproducible baseline.

## What makes Armenian different

Several properties of Armenian create differences between a transcript and its reference that have nothing to do with recognition:

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/armenian-features.png" alt="Five features of Armenian that affect ASR scoring, with examples" width="90%">
</p>

- **Punctuation and orthography.** Armenian has its own punctuation marks, and its question, exclamation and emphasis marks are written inside the word. The word "and" has two accepted spellings: a single ligature letter, or two separate letters. Models also often produce similar-looking Latin characters instead of Armenian ones.
- **Numbers.** Armenian is highly inflected, and numbers take grammatical endings like other words. The same number can be written in digits or in words, with the ending attached either way. Elsewhere, different endings usually mean different words, and they should be counted as errors.
- **Code-switching.** Speakers often use English and Russian words, which can appear in Latin, Cyrillic or Armenian letters.
- **Eastern and Western Armenian.** The language has two standard forms. Both public test sets are Eastern Armenian, so Western Armenian is not covered yet.

## Armenian-specific normalization

Before scoring, the leaderboard normalizes both the reference and the prediction to reduce differences caused by formatting and orthographic conventions rather than speech recognition errors. Because generic multilingual normalization handles Armenian poorly, we implemented a dedicated Armenian normalizer. It:

- maps Armenian punctuation, and its Latin look-alikes, to a single form before removing it, including the marks written inside words;
- treats the two spellings of "and" as the same word;
- converts numbers, years, percentages and clock times to one canonical form, whether they are written in digits or words;
- keeps Latin and Cyrillic words instead of deleting them, without trying to match them to Armenian spellings.

The normalizer does not merge different word forms in general. Outside of numbers, a different grammatical ending is still counted as an error.

To show the effect, each example below has a prediction that differs from the reference only in punctuation, capitalization, or how the same spoken words are written:

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/normalization-effect.png" alt="Strict vs normalized WER on six sentences that differ only in formatting" width="90%">
</p>

Without normalization, a model that writes "five percent" in words instead of "5%" is penalized for something that has nothing to do with what it heard. With a shared normalizer, every model is scored by the same rules, so differences in formatting style do not decide the ranking.

## Results on two independent test sets

Both test sets are Eastern Armenian read speech, but they come from different sources and were recorded in different ways:

| Dataset | Clips | Duration | Style | Notes |
| :--- | ---: | ---: | :--- | :--- |
| [FLEURS](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/fleurs_hy) | 932 | 3.0 h | Read Wikipedia sentences | 314 references [manually corrected](https://huggingface.co/datasets/Metric-AI/fleurs-corrections) |
| [Common Voice 26](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/mcv_hy) | 6,892 | 10.8 h | Crowd-sourced read speech | Short clips, many volunteer speakers and microphones |

Every model is run with the same settings on both sets, and both are scored with the same Armenian normalizer and WER implementation. Any difference between the two columns therefore comes from the data, not from the evaluation setup.

<p align="center">
  <img src="/blog/assets/open-asr-leaderboard-armenian/fleurs-vs-mcv.png" alt="FLEURS vs Common Voice WER for the top 12 models, with rank on each set" width="90%">
</p>

- **Rankings depend on the test set.** NVIDIA's small Armenian [FastConformer](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) is 9th on FLEURS but 1st on Common Voice. Omnilingual ASR 7B goes from 2nd to 5th, and Gemini from 5th to 7th. On a single test set, these models would look very different from how they look on the other.
- **Some results are stable.** Six of the twelve models keep the same rank on both sets, which makes their relative position more trustworthy.
- **Common Voice has higher WER for almost every model.** The two sets differ in content, speakers, recording conditions and clip length, so this gap cannot be attributed to any single factor.

This is why the leaderboard reports both sets, rather than just one.

Both sets are read speech, though, and performance on read speech does not necessarily predict performance on conversational or broadcast audio. To close this gap, we plan to add private Armenian test sets to the leaderboard with speech from three domains:

- **Movies:** conversational dialogue, often with background noise and music.
- **News:** broadcast-style narration.
- **Poems:** expressive literary recitation, often with background music.

The goal is to evaluate how models transcribe realistic, noisy speech with background sound, which read-speech test sets do not capture. Like the leaderboard's other private sets, they will be held out to limit benchmark-specific optimization, and the Hugging Face team will run the evaluation on them. The public FLEURS and Common Voice sets remain the baseline anyone can reproduce.

## Limits of normalized WER

Some differences are not formatting. Everyday speech uses colloquial forms that references usually write in the literary form, a foreign name can be correctly written in Latin or in Armenian letters, and some words have more than one accepted spelling. A rule-based normalizer cannot reliably separate these from real errors, so the leaderboard still counts them, in the same way for every model.

Normalized WER provides a deterministic and reproducible basis for comparing models, but it cannot account for every legitimate spelling or colloquial variation. We explore complementary evaluation approaches, including LLM-assisted scoring, in [ArmBench-ASR](https://metric-ai-armbench-asr.static.hf.space/index.html), our dedicated Armenian ASR benchmark.

## How to use it

On the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), open the **🌍 Multilingual** tab. Armenian is not selected by default, so check "Armenian" under "Languages", or choose it from the **Language dataset breakdown** dropdown to see each test set separately.


## Conclusion

Armenian now has a public, reproducible baseline on an established multilingual benchmark: two public test sets, one consistent scoring method, and results anyone can inspect and rerun. Next, we want to extend it to more realistic speech, such as spontaneous conversation, code-switching and Western Armenian. If you work on Armenian speech, we would love to see your models on the leaderboard.
