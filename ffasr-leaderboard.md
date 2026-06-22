---
title: "Introducing the FFASR Leaderboard: Benchmarking ASR in the Real World"
thumbnail: /blog/assets/ffasr-leaderboard/thumbnail.png
authors:
- user: daniel-treble
- user: whojavumusic
- user: alessia-treble
- user: georg-goetz
- user: bezzam
---

# Introducing the FFASR Leaderboard: Benchmarking ASR in the Real World

🚀 **First open far-field ASR benchmark:** community-driven evaluation across 14 simulated rooms, validated against real-world measurements

📉 **The gap is real and it is large:** across all submitted models, far-field WER at low SNR is consistently several times higher than near-field WER on the same speech content

🔬 **Methodology you can trust:** hybrid wave-based simulation, sim-to-real validation, moving-source splits in beta, held-out audio, and standardized evaluation hardware across all submissions

⚡ **Accuracy and speed together:** the Pareto front plots average WER against RTFx so you can evaluate the tradeoff that is right for your deployment

👀 **More is coming:** multi-talker scenarios, microphone array support, and echo cancellation are on the roadmap

The gap between benchmark performance and real-world deployment is one of the more persistent frustrations in ASR development. Models that score well on standard evaluations often behave differently once real room acoustics are involved: reverberation, background noise, microphone distance. The complex interactions between these factors affect performance in ways that clean-speech benchmarks do not capture. The FFASR Leaderboard is our attempt to quantify that gap.

[Treble Technologies](https://huggingface.co/treble-technologies) and Hugging Face are launching the Far-Field ASR (FFASR) Leaderboard, the first open, community-driven benchmark designed to evaluate ASR models under realistic far-field acoustic conditions. It is live now, and we are inviting the community to submit models, explore the results, and help shape what comes next.

## Why far-field evaluation matters

Voice interfaces have expanded well beyond the headset and the smartphone. AI voice agents, conference room transcription, in-car assistants, humanoid robots, smart glasses, and hands-free tools are all seeing rapid adoption. What they have in common is that they operate in acoustically complex environments: reverberation, background noise, overlapping sounds, and a microphone that may be anywhere from one to several meters from the speaker.

The dominant ASR evaluation paradigm has not caught up with this reality. Clean, close-microphone benchmarks remain the standard, and while they are useful for measuring core recognition quality, they do not predict far-field performance. A model that performs well on LibriSpeech or other near-field sets may degrade substantially once real room acoustics enter the picture. While there have been several research efforts around far-field and noisy speech evaluation — including [CHiME](https://www.chimechallenge.org/), [URGENT](https://v2.urgent-challenge.com/), and [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/) — the community has not had a standardized, open way to measure that degradation consistently across models in a continuously updated leaderboard format. That is what FFASR is built for.

A major challenge of far-field evaluation is the availability of data. Collecting far-field recordings across a representative range of room types, microphone distances, and noise conditions at scale is prohibitively expensive with physical measurements alone. Simulation makes it possible to cover that space systematically and to extend coverage over time without a corresponding increase in measurement cost.

A secondary goal is to encourage the development of models that are explicitly robust to these conditions. Leaderboards have historically been effective at directing research effort. By making far-field performance visible and comparable, we hope to raise the priority of real-world acoustic robustness across the field.

## How the benchmark is constructed

The FFASR Leaderboard evaluates models across nine conditions. The four that determine the primary ranking score are:

- Near-field (dry) — clean speech measured in an anechoic chamber (similar to Librispeech but with minimal reverberation)
- Far-field high SNR (above 14 dB)
- Far-field mid SNR (8 to 12 dB)
- Far-field low SNR (below 6 dB)

To give a sense of what these conditions actually sound like, the samples below let you hear the same speech utterance as dry anechoic audio, then convolved with a room impulse response, and finally with noise added at each SNR tier. The difference between the dry recording and the low-SNR far-field condition is a reasonable proxy for the scale of the problem the leaderboard is measuring.

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ffasr-leaderboard/dry.wav" type="audio/wav">
</audio>

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ffasr-leaderboard/High_SNR.wav" type="audio/wav">
</audio>

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ffasr-leaderboard/Mid_SNR.wav" type="audio/wav">
</audio>

<audio controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ffasr-leaderboard/Low_SNR.wav" type="audio/wav">
</audio>

Two additional columns, Lab Measured and Lab Simulated, serve as a sim-to-real validation track. The leaderboard also includes moving-source splits, currently in beta, which evaluate models against audio where the speaker is in motion rather than stationary. This condition reflects use cases such as humanoid robots, in-car speech, and mobile voice assistants where the acoustic geometry between speaker and microphone changes continuously.

The acoustic data is generated with [Treble's hybrid simulation engine](https://docs.treble.tech/intro), which combines a wave-based solver at low to mid frequencies with geometrical-acoustics modeling at higher frequencies. This approach captures physical phenomena that simpler simulation methods often miss: diffraction, scattering, interference, and modal behavior. The result is simulated data that closely matches measured acoustic conditions, which the Lab Measured and Lab Simulated columns confirm directly by running the same evaluation on both.

Fourteen fully furnished rooms are included in the benchmark, ranging from 20 to 470 m³ and covering bathrooms, living rooms with hallways, offices, classrooms, and restaurant spaces. Each acoustic scene contains one target speaker, recorded in an anechoic chamber to avoid reverberation artifacts from the recording environment, and up to three noise sources. Every scene includes both a transient noise source such as coughing and a continuous noise source such as HVAC, at three SNR levels. This coverage is designed to reflect the actual variety of spaces where deployed voice systems operate.

Alongside WER, the leaderboard reports RTFx (audio seconds per inference second) for every submission, evaluated on an NVIDIA L4 GPU under identical conditions. Accuracy and latency together are what matter in real deployments, and the Pareto front view in the Analysis tab makes that tradeoff explicit.

![Pareto front of average WER vs RTFx across submitted models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ffasr-leaderboard/pareto-screenshot.png)

This benchmark is build on simulated acoustic spaces via Treble Technologies proprietaty simulation engine. An example of the output from the enginge can be found in the [Treble10 dataset](https://huggingface.co/collections/treble-technologies/treble10) released last year, which established the simulation pipeline and made far-field RIRs available for training and research. FFASR extends that foundation into a standardized evaluation framework with a held-out test set, consistent normalization, and automated scoring.

## What the data already shows

With the leaderboard live, a consistent pattern is emerging across all submitted models: the gap between near-field and far-field performance is large, and it grows significantly as SNR decreases. Near-field WER values, on clean dry speech, look comparable to what the same models achieve on established benchmarks. Far-field WER at low SNR tells a different story, often several times higher. The benchmark makes this degradation visible and comparable in a way that was previously difficult to do outside proprietary evaluation pipelines.

The Pareto front of average WER against RTFx is also revealing. There is a genuine spectrum of approaches represented in the current submissions: models that prioritize speed at the cost of some accuracy, models that push accuracy at the cost of throughput, and a smaller number that achieve a competitive position on both axes. Visualizing these tradeoffs against far-field accuracy rather than clean-speech accuracy produces a materially different picture of where the real differences between systems lie. The Analysis tab is worth exploring beyond the main ranking table.

One observation worth highlighting for developers: the leaderboard reports both near-field (dry) and far-field WER side by side. This separation is intentional and useful. It makes it possible to distinguish between a model that is genuinely accurate and one that is accurate but brittle to acoustic conditions, which matters for deciding whether to invest in far-field fine-tuning, speech enhancement preprocessing, or a different architecture altogether.

## How to submit

Open the Submit tab on the [FFASR Leaderboard](https://huggingface.co/spaces/treble-technologies/ffasr), paste a Hugging Face model ID, and evaluation runs server-side against the held-out dataset. The pipeline supports Whisper variants, IBM Granite Speech, Cohere Transcribe, Wav2Vec2 and HuBERT CTC heads, SpeechBrain ASR, and most other ASR architectures on the Hub without any custom configuration.

For teams using more complex inference stacks, including systems that combine speech enhancement with ASR, a custom evaluator option allows you to define your own `evaluate()` function. Custom evaluators run on Hub Jobs after moderator review, and the submission notes field is a good place to document any preprocessing steps so results are interpretable by others.

![Custom evaluate method](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ffasr-leaderboard/custom_evaluate.png)

The held-out evaluation set uses 2,000 anechoic speech samples across 14 rooms at three SNR tiers, approximately 8 hours of audio per condition, with Whisper-style text normalization applied consistently. The audio is not exposed to submitters, to avoid test-set contamination.

## What is coming next

The conditions we are actively exploring for future tracks include multi-talker scenarios, where more than one speaker is active simultaneously, microphone array evaluation, covering beamforming and spatial filtering approaches, and echo cancellation, relevant for any device that plays audio while also listening.

What we build next will depend on where the community tells us the gaps are largest. If you work on a deployment environment or a use case that is not well represented in the current benchmark, we want to hear from you. The FFASR Leaderboard is designed to grow, and the direction it grows should reflect real needs.

Submit your model, explore the Analysis tab, post your ideas and suggestions on the [FFASR forum](https://huggingface.co/spaces/treble-technologies/ffasr/discussions), and help us build a benchmark that is actually useful for the problems the field is working on.
