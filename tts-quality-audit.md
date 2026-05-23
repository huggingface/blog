---
title: "Quality Checks on Popular TTS Datasets"
thumbnail: /blog/assets/tts-quality-audit/thumbnail.png
authors:
- user: manuML
---

# Quality Checks on Popular TTS Datasets

I built a tool that runs 13 signal-level checks on audio files to flag common data engineering issues before training. It runs on CPU, requires no ML model, and works directly on HuggingFace datasets. I used it to audit 1,500 samples across three widely-used TTS/ASR datasets all three datasets in under 3 minutes on a free Google Colab CPU. 

```bash
pip install audio-data-quality-toolkit
```

---

## Datasets

| Dataset | Description | Sample Rate | Total Size |
|---------|------------|-------------|------------|
| **LibriTTS-R** (clean) | Restored version of LibriTTS using Google's Miipher speech restoration model. Multi-speaker audiobooks, 2,456 speakers. | 24kHz | 585h |
| **MLS English** | Multilingual LibriSpeech. Derived from LibriVox audiobooks, segmented and aligned. | 16kHz | 44.5Kh |
| **LibriSpeech** (clean) | Standard ASR benchmark. Read English speech from LibriVox audiobooks. | 16kHz | 960h |

All three derive from LibriVox audiobook recordings but differ in how they were processed, segmented, and cleaned.

## Results (500 samples per dataset)

| Dataset | Clean % | Avg Score | Grade A | Grade B | Main Failures | Audit Time |
|---------|---------|-----------|---------|---------|---------------|------------|
| **LibriTTS-R** | 98% | 8.9/10 | 242 | 258 | Background noise (2%) | 44s |
| **MLS English** | 81% | 9.6/10 | 499 | 1 | Silence (19%) | 98s |
| **LibriSpeech** | 89% | 9.4/10 | 409 | 91 | Noise (6%), Loudness (5%) | 33s |

**Total: 1,500 files audited in 2.9 minutes on a free Colab CPU.**

### Per-check pass rates

| Check | LibriTTS-R | MLS English | LibriSpeech |
|-------|-----------|-------------|-------------|
| Background noise | 98% | 100% | 94% |
| Clipping | 100% | 100% | 100% |
| Silence | 100% | 81% | 100% |
| Sample rate | 100% | 100% | 100% |
| Duration | 100% | 100% | 100% |
| Loudness | 100% | 100% | 95% |
| Metallic artifacts | 100% | 100% | 100% |
| Repetition | 100% | 100% | 100% |
| Channel | 100% | 100% | 100% |
| Upsampling | 100% | 100% | 100% |
| Transcript ratio | -- | 100% | 100% |

<p align="center">
  <img src="https://huggingface.co/spaces/manuML/audio-data-quality-tool/resolve/main/per_check_datasets.png" alt="Per-check pass rates heatmap" width="800"/>
</p>

<p align="center">
  <img src="https://huggingface.co/spaces/manuML/audio-data-quality-tool/resolve/main/duration_vs_quality_score.png" alt="Duration vs quality score" width="600"/>
</p>
---

## Findings

**Quality score and pass rate measure different things.** MLS English has the highest average score (9.6/10, with 499 of 500 files graded A) but the lowest overall pass rate (81%). LibriTTS-R has a lower average score (8.9) but the highest pass rate (98%). This happens because MLS files have excellent signal quality -- almost no noise, clipping, or loudness issues -- but many contain untrimmed silence from audiobook chapter boundaries. The score captures signal quality; the pass rate captures training readiness. Both are useful, for different purposes.

**Silence is the dominant issue in MLS.** 19% of MLS files have leading or trailing silence exceeding 1 second, or internal gaps exceeding 3 seconds. This is consistent with MLS being derived from long-form LibriVox recordings that were segmented but not aggressively trimmed. For TTS training, these pauses consume compute without contributing useful signal. Applying a voice activity detector to trim boundaries before training would likely improve data efficiency.

**LibriSpeech "clean" has moderate noise and loudness variation.** 6% of LibriSpeech files fall below 15dB SNR and 5% fall outside EBU R128 loudness tolerance (-23 LUFS +/- 10dB). The noise failures are not severe -- most affected files still score above 8/10 -- but they indicate that the "clean" label reflects transcript accuracy more than signal quality. For ASR training this is usually fine; for TTS fine-tuning, filtering on SNR may help.

**LibriTTS-R benefits from speech restoration.** The Miipher model used to restore LibriTTS-R appears effective. Only 2% of files fail any check, all on borderline background noise in quiet speech passages. This is consistent with the LibriTTS-R paper, which showed improved MOS scores over the original LibriTTS across most speakers.

**None of the datasets achieve a 100% overall pass rate.** Across 1,500 files, 162 failed at least one check. This is expected -- no automated pipeline perfectly segments and cleans hundreds of hours of audio. The practical implication is that filtering before training is worth doing, especially for smaller fine-tuning runs where individual sample quality has more impact. With a full audit running in under 3 minutes on free CPU, there's no reason to skip it.

**Clipping, channel issues, upsampling, and metallic artifacts are absent.** All three datasets pass these checks at 100%. This suggests that the common failure modes in crowd-sourced or web-scraped audio (upsampled sample rates, stereo/phase issues, codec artifacts) are not a concern for these curated audiobook datasets. These checks would likely be more useful on datasets like Common Voice or GigaSpeech.

---

## Using the tool

```python
from datasets import load_dataset
from audio_qa import audit_hf_dataset

ds = load_dataset("blabble-io/libritts_r", "clean",
                  split="train.clean.100", streaming=True)
report = audit_hf_dataset(ds, max_samples=500)
print(report.summary())

# Actionable outputs
report.export_clean_manifest("clean.txt")   # file list for training
report.to_csv("qa_report.csv")              # per-file breakdown
clean_ds = report.filter_hf_dataset(ds)     # filtered HF dataset
```

The toolkit runs 13 checks per file: SNR estimation, clipping detection, silence analysis, sample rate validation, duration bounds, loudness (LUFS), metallic artifact detection, repetition detection, channel issues, upsampling detection, transcript-audio ratio, duplicate fingerprinting, and a composite quality score (0-10).

### Speed

All checks are numpy/scipy/librosa operations -- no ML inference, no GPU. In the audit above, all 1,500 files across three HuggingFace streaming datasets completed in 2.9 minutes on a free Colab CPU:

| Dataset | Files | Audit Time | Per File |
|---------|-------|-----------|----------|
| LibriTTS-R | 500 | 44s | ~88ms |
| MLS English | 500 | 98s | ~196ms |
| LibriSpeech | 500 | 33s | ~66ms |

Processing time scales linearly with file count. Extrapolated estimates:

| Files | Time (Colab free, CPU) | Time (local, 4 workers) |
|-------|----------------------|------------------------|
| 500 | ~30s-1.5 min | ~15-30s |
| 1,000 | ~1-3 min | ~30s-1 min |
| 10,000 | ~10-30 min | ~3-8 min |

For local directories, `check_directory` uses parallel workers (default 4) to process files concurrently. For HuggingFace streaming datasets, `audit_hf_dataset` processes sequentially since samples are streamed one at a time. In both cases, the bottleneck is audio loading and FFT computation, not network or disk I/O.

By comparison, running NISQA or UTMOS on the same 500 files would require GPU access and PyTorch inference time per sample. DataSpeech additionally requires an LLM call per sample for the description step.

### How this relates to existing tools

There are several established tools for audio quality assessment, each designed for a different purpose. Understanding where this toolkit fits requires looking at what each tool actually computes.

**NISQA** (Mittag et al., Interspeech 2021) is a no-reference perceptual quality predictor. It uses a CNN operating on mel-spectrogram segments to extract frame-level features, followed by a self-attention mechanism to model time dependencies, and attention pooling to aggregate into a single MOS prediction (1-5 scale). It also predicts four quality dimensions: noisiness, coloration, discontinuity, and loudness. NISQA was trained end-to-end on 81 crowdsourced datasets with subjective ratings. It requires PyTorch and benefits from GPU acceleration. NISQA answers: "How would a human listener rate this audio's perceptual quality?"

**UTMOS** (Saeki et al., Interspeech 2022) takes a different approach. It fine-tunes large self-supervised learning (SSL) models (wav2vec 2.0, HuBERT) as feature extractors, adding frame-level BLSTMs with contrastive loss, listener-dependent embeddings, and phoneme encoding. These "strong learners" are ensembled with "weak learners" (classical regressors on mean-pooled SSL embeddings) through multi-stage stacking. UTMOS achieved the top score in the VoiceMOS Challenge 2022. It also predicts MOS on a 1-5 scale but tends to correlate better with human judgments on synthetic speech than NISQA. It requires PyTorch and a GPU. UTMOS answers: "How natural does this synthetic speech sound to a human?"

**PESQ** (ITU-T P.862) is an intrusive/reference-based metric. It requires both the degraded signal and a clean reference, aligning them in time and computing perceptual differences. PESQ scores range from -0.5 to 4.5. It runs on CPU but cannot be used when no reference exists which is the case for most dataset curation tasks. PESQ answers: "How much has this signal been degraded compared to a known clean original?"

**DataSpeech** (Hugging Face, 2024) is an annotation pipeline for Parler-TTS. It computes speaking rate, SNR, reverberation, and pitch statistics, then uses an LLM to generate natural language descriptions like "A female speaker delivers a slightly expressive speech at a moderate pace in a clean environment." These descriptions are used as conditioning inputs for Parler-TTS training. DataSpeech requires GPU for both the annotation models and the LLM description step. It answers: "What are the acoustic characteristics of this audio for conditioned TTS?"

**audio-qa** (this toolkit) does not predict perceptual quality or generate descriptions. It runs deterministic signal-level checks -- FFT-based upsampling detection, RMS-gated SNR estimation, LUFS measurement, chromagram fingerprinting for duplicates, autocorrelation for repetition and outputs a pass/fail per check plus a composite score. Everything runs on CPU with numpy, scipy, and librosa. It answers: "Does this file have data engineering problems that would hurt training?"

The key distinction is between perceptual assessment and data engineering assessment. A file can score 4.5/5 on NISQA (sounds great to humans) but still have 3 seconds of leading silence, a misaligned transcript, or be upsampled from 8kHz all of which waste compute or introduce noise during training. Conversely, a file can fail all 13 checks here but still sound perfectly fine to a listener. These tools are complementary: use NISQA/UTMOS to assess output quality of your trained model, use audio-qa to clean the training data going in.

| Tool | Method | Input | GPU | Output | Question |
|------|--------|-------|-----|--------|----------|
| **NISQA** | CNN + self-attention on mel-specs | Audio only | Yes | MOS 1-5 + 4 dimensions | Perceptual quality |
| **UTMOS** | Fine-tuned SSL ensemble + BLSTMs | Audio only | Yes | MOS 1-5 | Naturalness of synthetic speech |
| **PESQ** | Time-aligned perceptual comparison | Audio + reference | No | Score -0.5 to 4.5 | Degradation vs reference |
| **DataSpeech** | SNR/pitch/rate + LLM descriptions | Audio + text | Yes | NL descriptions | Acoustic characteristics |
| **audio-qa** | Deterministic signal checks (FFT, RMS, LUFS) | Audio (+text optional) | No | Score 0-10 + pass/fail + manifest | Training readiness |

---

**Links:**
[GitHub](https://github.com/EmmanuelleB985/audio-data-quality-tool) --
[HuggingFace Space](https://huggingface.co/spaces/manuML/audio-data-quality-tool) --
[Colab Notebook](link_to_your_colab) --
[PyPI](https://pypi.org/project/audio-data-quality-toolkit/) --
MIT licensed. Contributions welcome.