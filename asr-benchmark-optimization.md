---
title: "Towards quantifying benchmark optimization in leading speech recognition models"
thumbnail: /blog/assets/asr-benchmark-optimization/thumbnail.png
authors:
- user: tlebryk02
  guest: true
  org: HumeAI
- user: bezzam
---

# Towards quantifying benchmark optimization in leading speech recognition models

Public voice AI benchmarks increasingly suggest that models are performing at a human levels. Yet those scores don't always reflect how models work in the real-world. Since public benchmarks are open and widely used, models can also become optimized for the tests themselves. Their scores may improve because they have learned benchmark-specific patterns and not because they have become better at the underlying task.

One reason is that traditional benchmarks overlook many of the conditions and qualities that make voice systems reliable, natural, contextually appropriate, and effective in practice. That's why we recently introduced held-out sets in [Real World VoiceEQ](https://huggingface.co/spaces/HumeAI/rw-voice-eq), the [Open-ASR Leaderboard](https://huggingface.co/blog/open-asr-leaderboard-private-data), and the [Far-field ASR Leaderboard](https://huggingface.co/spaces/treble-technologies/ffasr): to measure more of what matters in real-world use.

However, broader measurement alone does not solve the problem. This phenomenon, sometimes called benchmark optimization or "benchmaxxing," is often discussed around machine learning, however, it has been difficult to measure in speech recognition.

Our latest research introduces three tests to help quantify it. We evaluated 11 widely used open-source ASR models and found that several of the highest-scoring systems reproduced benchmark transcripts from the [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) English and [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) (clean, other) datasets – even when the audio contradicted them, relevant words had been silenced, or the audio equally supported two different written forms.

In some cases, models appeared to rely not only on what was said, but also on subtle acoustic cues that indicated which benchmark they were being tested on. As a result, their scores overstated how well they could transcribe speech more generally.

## Reference disagreement (VoxPopuli case study)

VoxPopuli is known to contain a high number of transcription errors (which is why Artificial Analysis released a [cleaned version](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA)). Our consensus disagreement probe tests what happens when leading ASR models encounter these errors: *Do they accurately transcribe what the audio says, or reproduce the benchmark's incorrect reference transcript?*

To test this at scale, we use an ensemble of independent models selected for their low phoneme error rate (PER). PER measures how closely a written transcription matches the sounds in the audio, making it a useful proxy for how faithfully a model transcribes what it hears. The ensemble results can be used to flag cases in which the models unanimously disagree with the benchmark's reference transcript. We then compare a sample of those flagged cases against human annotations to validate the corrected transcripts.

For example, one VoxPopuli clip audibly includes the phrase "Thank you, Mr. President," but the reference transcript omits "Thank you." Six of the 11 models we tested reproduced the benchmark's erroneous transcript—giving the "expected" answer even though it contradicted the audio. On the real clip, the formatting follows the same pattern: models that omit "Thank you" also reproduce the benchmark's punctuation style, writing "Mr" without a period, while models that include the audible phrase tend to write "Mr." with the period.

When we present the same content in newly collected voices from EU parliamentary recordings or generic voices, this behavior often weakens or disappears. In the below samples, all but one model flips back to transcribing the audio-faithful transcript for a clone of a new parliamentary recording. This suggests that the models are responding to acoustic cues that help them identify the benchmark membership and thus produce the expected transcript even if it contradicts the audio.

The reference transcript for this clip reads "Mr President, I have another complaint about this procedure, which is that it is not secret." The audio in all three clips below actually says the same thing, preceded by an audible "Thank you,"—the clones are text-to-speech renditions of that true sentence, so the courtesy is audible in all three. Green highlighting and ✅ mark a transcript that includes the audible "Thank you"; red highlighting and ❌ mark a transcript that reproduces the benchmark's erroneous omission. All transcripts are raw model output, prior to any normalization—casing and punctuation are preserved exactly as generated, including lowercase output from some models.

**Original VoxPopuli recording**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/1287_real.wav"></audio>

**Voice clone of the same speaker**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/1287_clone_same_speaker.wav"></audio>

**Clone of a parliament speaker recorded after every model's training cutoff**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/1287_clone_ep_fresh.wav"></audio>

| Model | Real clip | Same-speaker clone | ep-fresh clone |
| --- | --- | --- | --- |
| Cohere-transcribe | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr President…</span> |
| Canary-qwen-2.5b | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you Mr. President…</span> |
| Granite-4.1-2b | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> |
| phi4-multimodal | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> |
| parakeet-tdt-0.6b-v2 | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> |
| Higgs-Audio-v3-8B | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> |
| qwen3-asr-0.6b | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mister President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mister President…</span> |
| Voxtral-mini | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> |
| Kimi audio | <span style="background-color:#dcfce7">✅ Thank you, mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, mr. President…</span> |
| Whisper large v3 | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> |
| moonshine-streaming-medium | <span style="background-color:#dcfce7">✅ thank you mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> |
| **Drops the courtesy (❌) out of 11** | **6** | **5** | **1** |

Parakeet is the only model that flips between reproducing the benchmark on the real clip and getting it right on the same-speaker clone. Phi-4 is the only model still dropping the courtesy on the ep-fresh clone. When we instead resynthesize the sentence in a generic TTS voice unconnected to any parliamentary recording, all eleven models restore the courtesy.

The results suggest that this problem is both widespread and meaningful. Our methodology flagged potential reference errors in 40% of the VoxPopuli test clips we analyzed, affecting roughly 3% of all reference words.

Models exhibiting benchmark-optimized behavior reproduced erroneous reference transcripts 20–33% of the time. The scatterplot below compares VoxPopuli word error rate (WER) on the x-axis with the rate at which each model reproduces the benchmark's incorrect reference instead of the consensus correction. The models with the lowest WER—and therefore the strongest reported benchmark performance—are also the most likely to reproduce these errors.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/wer_vs_badref.png" width="800px" alt="Scatterplot comparing VoxPopuli WER to the rate at which each model reproduces the benchmark's incorrect reference transcript." />
</div>

## Masked Entity Retrieval

To build on the consensus disagreement probe, we deliberately silence numbers in the audio samples of test datasets and ask the models to transcribe what it hears. The number is literally absent from the audio, so models should not output any number, much less the exact number in the text.

Some of these numbers are semi-predictable (although still unlikely for a model to predict), yet others are quite surprising. The following clip combines both probes, showing both how models recreate reference transcript errors including an incorrect number and one model even autocompletes a relatively random year (2011) despite it being silenced. In each model's row below:
- green highlighting with strikethrough marks reference-transcript words the model correctly did not reproduce (audio-faithful);
- green highlighting with <u>underline</u> marks a correct, audio-faithful insertion in place of the reference's erroneous wording;
- red highlighting (plain text) reproduces the reference transcript's erroneous, audio-unsupported content, e.g. addition of "Mr President" or omission of "thousand six hundred".

**2011 draft budget (masked numbers)**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/2011_draft_budget.wav"></audio>

| Reference  | <span style="background-color:#fee2e2">Mr President,</span> in the Committee on Budgets we voted on more than one amendments to the <span style="background-color:#fee2e2">2011</span> draft … voted in the <span style="background-color:#fee2e2">plenary</span> |
| --- | --- |
| Masked and corrected | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets we voted on more than one <span style="background-color:#dcfce7"><u>thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft …voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| Cohere-transcribe  | <span style="background-color:#fee2e2">Mr President,</span> in the Committee on Budgets we voted on more than one <span style="background-color:#fee2e2">thousand six hundred</span> amendments to the <span style="background-color:#fee2e2">2011</span> draft… voted in the <span style="background-color:#fee2e2">plenary</span>. |
| Canary-qwen-2.5b | <span style="background-color:#fee2e2">Mr President</span> in the Committee on Budgets we voted on more than one <span style="background-color:#fee2e2">thousand six hundred</span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft… voted in the <span style="background-color:#dcfce7">~~plenary~~</span>. |
| Granite-4.1-2b | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets we voted on more than one <span style="background-color:#fee2e2">thousand</span> six hundred amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft …voted in the <span style="background-color:#fee2e2">plenary session</span> |
| phi4-multimodal | <span style="background-color:#dcfce7">~~Mr President,~~</span>  In the Committee on Budgets we voted on more than one <span style="background-color:#fee2e2">thousand six hundred</span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft… voted in the <span style="background-color:#fee2e2">plenary</span>. |
| parakeet-tdt-0.6b-v2 | <span style="background-color:#dcfce7">~~Mr President,~~</span>  In the Committee on Budgets we voted on more than one thousand six hundred amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft… voted in the <span style="background-color:#fee2e2">plenary</span>. |
| Higgs-Audio-v3-8B | <span style="background-color:#dcfce7">~~Mr President,~~</span> in the committee on budgets we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft… voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| qwen3-asr-0.6b | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets we voted on more than <span style="background-color:#dcfce7"><u>1,600</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft …voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| Voxtral-mini | <span style="background-color:#dcfce7"><u>In</u></span> the Committee on Budgets, we voted on more than <span style="background-color:#dcfce7"><u>1,600</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft... voted in the…<span style="background-color:#dcfce7">~~plenary~~</span> |
| Kimi audio | <span style="background-color:#dcfce7"><u>Ah In</u></span> the committee on budgets we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft... voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| Whisper large v3 | <span style="background-color:#dcfce7"><u>In</u></span> the Committee on Budgets, we voted on more than <span style="background-color:#dcfce7"><u>1,600</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft... voted in the…<span style="background-color:#dcfce7">~~plenary~~</span> |

Recovery rates were highest on the public benchmarks and lower on held-out or newly collected audio (ep-fresh and LibriVox-fresh below). On LibriSpeech, some of the strongest benchmark-performing models reproduced masked numbers in roughly 30–40% of examples, even though the number itself had been removed. The effect weakened on freshly collected data for several models, suggesting that the surrounding benchmark-associated audio—not only textual autocomplete—helped the models recover the reference.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/masking_freshpairs.png" width="700px" alt="Recovery rate of masked numbers on public benchmarks versus freshly collected held-out audio." />
</div>

## Orthographic Switching

Our orthographic switching probe tests whether models reproduce the exact spelling used in a benchmark's reference transcript despite it not being clear in the audio. Orthographic variants are words that are semantically and phonetically identical but can be spelled different ways (1 vs one, Mr. vs mister, John vs Jon, Honor vs Honour, etc). In theory, models should be consistently prefer one spelling over another, or alternate between them at roughly random rates. If models systematically switch to match what is in each benchmark's reference transcript, that suggest the models are picking up on which spelling the test expects.

**Transcription: "I URGED ON THE BOYS THAT WHATEVER HAPPENED WE SHOULD NOT SHOOT ANY ONE" — models using "any one": 6/11, models using "anyone": 5/11**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/any_one.wav"></audio>

**Transcription: "CAMOUFLAGE WAS NOT A WORD THE CAPTAIN OR ANYONE ELSE OF HIS TIME YET UNDERSTOOD" — models using "any one": 2/11, models using "anyone": 9/11**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/anyone.wav"></audio>

Within LibriSpeech, we test one *intra-dataset* switch involving an older spacing convention: some reference transcripts use "any one", while others use "anyone." We measure the minimum accuracy for a given variant, which we call "switch rate". If a model only uses one variant it would have a 0% switch rate; a model which picks randomly would be expected to have a 50% switch rate. A model which knows which variant to use in every test sample would earn a 100% switch rate.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/pair_spacing_sorted.png" width="700px" alt="Switch rate for the &quot;any one&quot; vs &quot;anyone&quot; spacing convention, sorted by model." />
</div>

Our second probe tests an *inter-dataset switch*, in which each benchmark uses a different spelling convention consistently across its test corpus. For example, VoxPopuli uses the abbreviation "Mr.," while LibriSpeech spells out "Mister."

Multiple models exceed the 50% random-choice baseline, with some reaching roughly 90% switch accuracy. **This suggests that the models can identify which dataset an audio sample comes from and select the spelling convention that benchmark expects, even though both forms sound identical.**

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/pair_mister_sorted.png" width="700px" alt="Switch rate for the &quot;Mr.&quot; vs &quot;Mister&quot; convention across VoxPopuli and LibriSpeech, sorted by model." />
</div>

## Localizing the switches

To test whether these behaviors generalize beyond the public benchmarks, we also collected fresh data from the same source domains but after the models' training cutoffs: recent European Parliament recordings for VoxPopuli and recordings from newly active LibriVox narrators for LibriSpeech. However, when presented with recently collected data from the same domain, many models stop matching the reference transcript and revert to more audio faithful transcriptions.

Other interventions point to the same conclusion. Phrases which are present in the audio but are omitted in the reference transcript can reappear when a model is asked to translate the audio or when its attention is restricted to the relevant frames. Trimming away surrounding benchmark context, or appending ordinary conversational audio, can also restore the faithful transcript. Appending VoxPopuli audio can have the opposite effect, making otherwise faithful synthetic or mined samples more likely to match the benchmark reference.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/steer_input_level_full.png" width="800px" alt="Effect of steering the amount of surrounding benchmark-associated audio context on transcription behavior." />
</div>

**Together, these results suggest that the model is able to faithfully transcribe the literal spoken words, but are using surrounding acoustic context to decide whether to follow the audio or a benchmark-specific transcription policy.**

## Conclusion

Our findings suggest that, on two major open-source datasets, some models detect dataset-associated acoustic cues and adjust their transcription behavior accordingly. Specifically, models may reproduce words that are absent from the audio but present in the reference transcript, recover silenced numbers at elevated rates, or use surrounding acoustic context to select the written variant expected by a particular benchmark.

For people selecting models, these findings underscore the importance of using fully held-out evaluation sets, as RW-Voice-EQ Bench and the Open ASR Leaderboard do, and of looking beyond word error rate on a single public benchmark.

Our findings also suggest that benchmark developers should avoid simple independent and identically distributed test splits in favor of temporal, speaker, or other metadata-based separation. Greater transparency around training data and model-selection procedures would also help researchers understand how these behaviors arise.

We are looking forward to incorporating these probes into the Open ASR Leaderboard to make benchmark-specific behavior more visible.

Public benchmarks remain valuable: they are transparent, repeatable, easy to run, and well understood by the research community. But they are most useful when we can distinguish genuine transcription improvements from benchmark-specific gains that do not generalize to new audio.
