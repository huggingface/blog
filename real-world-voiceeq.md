---
title: "Introducing Real World VoiceEQ: Measuring the human quality of voice AI"
thumbnail: /blog/assets/real-world-voiceeq/thumbnail.png
authors:
- user: dayllon
  guest: true
  org: HumeAI
- user: aliceebaird
  guest: true
  org: HumeAI
- user: jeffbrooks
  guest: true
  org: HumeAI
- user: francamps
  guest: true
  org: HumeAI
- user: jpc
  guest: true
  org: HumeAI
- user: tlebryk02
  guest: true
  org: HumeAI
- user: jens-hume-ai
  guest: true
  org: HumeAI
- user: itsolyaossi
  guest: true
  org: HumeAI
- user: sharath25
  guest: true
  org: HumeAI
- user: hoon-hume
  guest: true
  org: HumeAI
- user: tig88
  guest: true
  org: HumeAI
- user: rashisht
  guest: true
  org: HumeAI
- user: tzirakis
  guest: true
  org: HumeAI
---

# Introducing Real World VoiceEQ: Measuring the human quality of voice AI

*Existing benchmarks suggest voice AI is nearing human-level performance but real-world conversations tell a different story.*

Voice is rapidly becoming AI's primary interface. From customer support and healthcare to education, entertainment, and personal assistants, speech is increasingly replacing text as the way people interact with AI.

Over the last few years, voice models have improved dramatically. Word error rates continue to fall, latency has reached conversational speeds, and many established benchmarks are approaching saturation. Yet anyone who regularly uses voice AI knows something still feels off.

Voice models can sound like different people over the course of a conversation, miss hesitation or uncertainty, and struggle with accents, noise, or emotional speech. Those shortcomings are easy to miss in benchmarks focused on latency and word error rate. People care whether a voice system can truly listen, respond appropriately, and remain natural and reliable in real conversations.

## A broader benchmark for voice AI

To measure those qualities, we built [Real World VoiceEQ](https://www.hume.ai/rw-voice-eq)—a benchmark designed to evaluate the human quality of voice interaction. It assesses whether voice systems can recognize, produce, and respond to the acoustic information transcripts leave out, from tone and emotion to speaker identity and background context.

Real World VoiceEQ evaluates more than **40 leading proprietary and open-source voice models** across **15+ key evaluation dimensions** and more than **60 metrics** spanning Automatic Speech Recognition (ASR), Text-to-Speech (TTS), Speech-to-Speech (S2S), and Speech Understanding.

![The four components of Real World VoiceEQ — Text-to-Speech, Speech-to-Speech, Speech Understanding, and ASR Robustness — each with its evaluation dimensions.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/real-world-voiceeq/benchmark-overview.png)

Real World VoiceEQ was developed from more than 1 million individual human ratings collected across different demographics, speaking styles, and acoustic environments. The current benchmark includes 785,000 TTS ratings and 48,000 STS ratings, making it one of the largest human evaluations of voice AI conducted to date.

Every evaluation was conducted using **[Kairos](https://www.hume.ai/kairos)**, our flexible, voice-native evaluation platform. The same infrastructure enables frontier AI labs and enterprises to run custom evaluations tailored to specific use cases, identify granular failure modes in production voice systems, generate human preference data, and continuously improve models through reinforcement learning and human feedback.

## Key findings from Real World VoiceEQ

### Progress in voice AI is becoming increasingly specialized.

The race for a single "best" voice model is giving way to a collection of specialized capabilities.

Today's leading systems optimize for different strengths—including technical accuracy, emotional understanding, conversational intelligence, expressiveness, and robustness. One model that excels at repeating booking reference numbers, bank account details, or complex pharmaceutical names may struggle to produce emotionally expressive speech. Another may sound remarkably natural but be less reliable on precision-oriented tasks.

As voice AI matures, measuring progress increasingly requires evaluating these capabilities independently rather than collapsing them into a single overall score. In our TTS evaluations, no system configuration ranked among the top five across all eight capability groups—underscoring why there is no single "best" voice model.

### Voice models have become better at speaking than *actually* listening.

Speech-to-Speech models showed the widest variation of any category we evaluated. Some systems recognized emotion exceptionally well but struggled to respond naturally. We found that access to audio did not guarantee that agents used the paralinguistic information it contained. Some systems remained largely transcript-driven, relying on the words being spoken while overlooking cues such as tone, pacing, hesitation, emphasis, and volume.

Humans naturally use these cues to infer confidence, uncertainty, frustration, sarcasm, and empathy. Today's models often miss them.

Imagine a banking agent asking whether you recognize a potentially fraudulent transaction. A confident "Yes" and a hesitant "…yes…" may have completely different meanings, even though the transcript is identical. Humans recognize that difference immediately. Many of today's voice models do not.

### Traditional benchmarks increasingly overestimate real-world performance.

Many established benchmarks are nearing their limits and don't reflect real-world conditions. Models still struggle with accented speech, overlapping speakers, emotion, background noise, and longer conversations. In our evaluation, performance varies far more across leading open-source and proprietary models than traditional benchmarks suggest. In one example, transcription word error rates on noise-backed speech were roughly four times higher than on music-backed speech, showing how a single background-audio score can hide the real failure mode.

### Human evaluation remains essential.

In preliminary research, we found signs that some models may be optimized for established public benchmarks. Several reproduced known errors in reference transcripts, followed arbitrary spelling conventions, and even reconstructed masked words that were not present in the audio.

LLMs are now widely used to evaluate text-based models, but our findings suggest that speech-language models (SLMs) should be used more carefully for voice evaluation. When we compared leading SLMs with trained human raters on text-to-speech assessments, agreement was highest on tasks with clear, verifiable answers, such as pronunciation accuracy.

Agreement declined on more subjective evaluations. SLMs sometimes appeared to infer emotion from text-based contextual cues, and agreement was weakest for open-ended judgments such as whether a voice fit an acting role or maintained a consistent identity. Automated evaluators can be valuable for well-defined tasks, but they are not yet a substitute for human listeners when judgments depend on acoustic-context, perception, and social interpretation.

## Why Voice AI needs a new measurement layer

As voice becomes one of AI's defining interfaces, speed and technical accuracy alone will no longer determine which systems succeed. The models people ultimately choose will be those that can understand, express, and respond like humans—not just under ideal benchmark conditions, but across the complexity of real-world conversation.

For decades, speech AI has advanced by optimizing against quantitative metrics on standardized benchmarks; from WER for transcription accuracy to objective perceptual metrics like PESQ and DNSMOS for speech quality. We hope Real World VoiceEQ can extend this paradigm by providing a human-grounded metric for evaluating the components of synthetic voice interactions.

**Read the [full technical report](https://huggingface.co/papers/2607.14846) and explore the [public leaderboards](https://huggingface.co/spaces/HumeAI/rw-voice-eq)—or [get in touch](https://www.hume.ai/sales-form) to learn how Hume can evaluate your voice model or agent using Real World VoiceEQ, or design custom evaluations tailored to your specific use case.**
