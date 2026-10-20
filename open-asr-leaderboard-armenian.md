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
  <img src="assets/open-asr-leaderboard-armenian/armenian-features.png" alt="Five features of Armenian that affect ASR scoring, with examples" width="90%">
</p>

- **Punctuation and orthography.** Armenian has its own punctuation marks, and its question, exclamation, and emphasis marks are written inside the word. In digital text, the word "and" can be represented either as a single ligature character or as two separate characters. Models also often use ASCII punctuation marks instead of their Armenian equivalents.
- **Numbers.** Armenian is a highly inflected language, and numbers take grammatical endings like other words. The same number can be written in digits or in words, with the ending attached either way.
- **Code-switching.** Speakers often use English and Russian words, which can appear in Latin, Cyrillic or Armenian letters.
- **Eastern and Western Armenian.** The language has two standard forms. Both public test sets are mainly in Eastern Armenian, so Western Armenian is not yet covered.

## Armenian-specific normalization

Before scoring, the leaderboard normalizes both the reference and the prediction to reduce differences caused by formatting and orthographic conventions rather than speech recognition errors. Because generic multilingual normalization handles Armenian poorly, we implemented a dedicated Armenian normalizer. It:

- removes punctuation;
- removes capitalization;
- treats the two spellings of "and" as the same word;
- converts numbers, years, percentages and clock times to one canonical form, whether they are written in digits or words;
- keeps Latin and Cyrillic words instead of deleting them, without trying to match them to Armenian spellings;
- does not merge formal and informal auxiliary verb forms.

For example, without normalization, a model that writes "five percent" in words instead of "5%" is penalized for something that has nothing to do with what it heard. With a shared normalizer, every model is scored by the same rules, so differences in formatting style do not decide the ranking.

## Results on two independent test sets

Both test sets are mostly Eastern Armenian read speech, but they come from different sources and were recorded in different ways:

| Dataset | Clips | Duration | Style | Notes |
| :--- | ---: | ---: | :--- | :--- |
| [FLEURS](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/fleurs_hy) | 932 | 3.0 h | Read Wikipedia sentences | All references [manually checked and corrected](https://huggingface.co/datasets/Metric-AI/fleurs-corrections) |
| [Common Voice 26](https://huggingface.co/datasets/Metric-AI/open-asr-leaderboard-multilingual-datasets/viewer/mcv_hy) | 6,892 | 10.8 h | Crowd-sourced read speech | Short clips, many volunteer speakers and microphones |

Every model is run with the same settings on both sets, and both are scored with the same Armenian normalizer and WER implementation. Any difference between the two columns therefore comes from the data, not from the evaluation setup.

<p align="center">
  <img src="assets/open-asr-leaderboard-armenian/fleurs-vs-mcv.png" alt="FLEURS vs Common Voice WER for the top 10 models, with rank on each set" width="90%">
</p>

- **Rankings depend on the test set.** NVIDIA's small Armenian [FastConformer](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) is 9th on FLEURS but 1st on Common Voice. Omnilingual ASR 7B is 2nd on FLEURS, but 5th on MCV. On a single test set, these models would look very different from how they look on the other.
- **Common Voice has higher WER for almost every model.** The two sets differ in content, speakers, recording conditions and clip length, so this gap cannot be attributed to any single factor.

This is why the leaderboard reports both sets, rather than just one.

Both sets are read speech, though, and performance on read speech does not necessarily predict performance on conversational or broadcast audio. To close this gap, we plan to add private Armenian test sets to the leaderboard with speech from three domains:

- **Movies:** conversational dialogue, often with background noise and music.
- **News:** broadcast-style narration.
- **Poems:** expressive literary recitation, often with background music.

The goal is to evaluate how models transcribe realistic, noisy speech with background sound, which read-speech test sets do not capture. Like the leaderboard's other private sets, they will be held out to limit benchmark-specific optimization, and the Hugging Face team will run the evaluation on them. The public FLEURS and Common Voice sets remain the baseline anyone can reproduce.

## Limits of normalized WER

Some differences are not formatting. Everyday speech uses colloquial forms that references usually write in the literary form, a foreign name can be correctly written in Latin or in Armenian letters, and some words have more than one accepted spelling. A rule-based normalizer cannot reliably separate these from real errors, so the leaderboard still counts them, in the same way for every model.

Normalized WER provides a deterministic and reproducible basis for comparing models, but it cannot account for every legitimate spelling or colloquial variation. Metric AI explores complementary evaluation approaches, including LLM-assisted scoring, in [🇦🇲 ArmBench-ASR](https://metric-ai-armbench-asr.static.hf.space/index.html), a dedicated Armenian ASR benchmark.

## How to use it

On the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), open the **🌍 Multilingual** tab. Armenian is not selected by default, so check "Armenian" under "Languages", or choose it from the **Language dataset breakdown** dropdown to see each test set separately.

## What comes next

- **Private datasets**: Adding the Movies, News, and Poems datasets.
- **Code-switching:** Including speech where speakers naturally alternate between Armenian and other commonly used languages, particularly English and Russian.
- **More diverse conversations:** Expanding the dataset to include informal discussions, interviews, customer-service calls, meetings, podcasts, and conversations with multiple speakers, including interruptions and overlapping speech.
- **Specialized domains:** Adding speech from fields such as healthcare, law, finance, education, technology, and government, where technical vocabulary and proper names can make transcription more difficult.
- **Speaker diarization:** Assessing whether systems can correctly identify who is speaking and when, especially in recordings with multiple or overlapping speakers.
- **Additional transcription tasks:** Evaluating both verbatim and cleaned-up transcripts, including punctuation, capitalization, removal of disfluencies, and conversion of spoken language into readable text.
- **Timestamp and segmentation accuracy:** Evaluating how accurately systems assign timestamps to words and utterances, divide longer recordings into segments, and handle pauses and speaker transitions.

Together, these additions would make the benchmark better reflect real-world speech recognition applications, where performance depends on more than achieving a low word error rate (WER) on short, single-speaker recordings.