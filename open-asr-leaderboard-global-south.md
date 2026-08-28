---
title: "The Open ASR Leaderboard Adds Its First Global South Language"
thumbnail: /blog/assets/open-asr-leaderboard-global-south/thumbnail.png
authors:
- user: bezzam
- user: Shobhitbanga
  guest: true
  org: VoiceArena
- user: manasdhir04
  guest: true
  org: VoiceArena
- user: bhaskarJT
  guest: true
  org: VoiceArena
- user: manmeet-voicearena
  guest: true
  org: VoiceArena
- user: pareek-voicearena
  guest: true
  org: VoiceArena
- user: Amritansh8675
  guest: true
  org: VoiceArena
- user: sagarjain268380
  guest: true
  org: VoiceArena
- user: hanuman44420
  guest: true
  org: VoiceArena
- user: vanshikachhabra-voicearena
  guest: true
  org: VoiceArena
---

# The Open ASR Leaderboard Adds Its First Global South Language

*Voice Arena and Hugging Face partner to launch open ASR evaluation for Hindi and Indian English*

Benchmarks decide what gets built. A model that scores well on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) gets adopted and iterated on, while capabilities the leaderboard does not measure tend not to improve. Much of the recent work on the leaderboard has gone into making the evaluation metrics more trustworthy: 
1. [Held-out private splits](https://huggingface.co/blog/open-asr-leaderboard-private-data).
2. [Benchmark-fitting analysis](https://huggingface.co/blog/asr-benchmark-optimization) to quantify how much models are reproducing reference transcripts rather than transcribing solely on the audio.
3. Closing the gaps in normalisers to ensure correct predictions/variants are not penalized.

All of that makes one number (WER) harder to game. It is still one number. A long line of work has shown that ASR error rates are not evenly distributed across the people using them. [Racial disparities in automated speech recognition](https://www.pnas.org/doi/10.1073/pnas.1915768117) found commercial systems roughly twice as bad for Black speakers as for white speakers, and [Quantifying Bias in Automatic Speech Recognition](https://huggingface.co/papers/2103.15122) found further differences by gender, age and accent. None of that is visible on a leaderboard, and not because the leaderboard is hiding it. The test sets it runs on record what was said and almost nothing about who said it.

To address this gap, we introduce two evaluation sets to the Open ASR Leaderboard: [Monsoon en-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN) and [Monsoon hi-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN). Hindi, **spoken by more than half a billion people**, is the first Indic language on a multilingual tab that currently covers only European languages. Each set is released as a public split, available for self-scoring, and a private split withheld to limit benchmark-specific optimisation. The four splits are speaker-disjoint, comprising 4,888 speakers, with 12 speaker attributes recorded for each.

## Design of the collection

A test set can only expose a failure mode it varies along. Most benchmarks are built from whatever audio was readily available. Monsoon was built to vary along nine axes: geography, age, gender, vocabulary, devices, acoustic environments, speech type, speech rate, and the existence of multiple valid transcripts for the same audio. Each is a way an aggregate WER can be right on average and wrong for a particular population.

![nine axes of variation in the Monsoon collection](https://storage.googleapis.com/research_team_data/blog_figures/nine-axes-gray.png)

The collection method follows from that.
- **Geography** comes from recruiting across hundreds of districts rather than recording longer sessions in fewer places.
- **Devices and acoustic conditions** come from contributors using their own handsets and connections, indoors and out, rather than supplied hardware in a quiet room.
- **Vocabulary, speech type and speech rate** come from the prompts: everyday topics that push contributors toward opinion, disagreement, narration and recall, which is where named entities, numbers and unrehearsed phrasing appear.
- **Age and gender** are recorded per speaker and verified.
- **Multiple valid transcripts** is a property of the reference rather than the audio, and it is the subject of a later section.

## Dataset composition

Four splits, two languages, collected through one pipeline.

| Set | Language | Duration | Speakers | Clip length (mean / median) | M/F | Districts | States/UTs | Devices | Style | Transcription |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| Monsoon en-IN public | Indian English | 5.62 h | 1,444 | 9.6s / 10.4s | 50/50 | 428 | 24/6 | 556 | Conversational, spontaneous | Normalised, disfluencies |
| Monsoon en-IN private | Indian English | 5.58 h | 1,405 | 9.6s / 10.4s | 45/55 | 420 | 24/6 | 560 | Conversational, spontaneous | Normalised, disfluencies |
| Monsoon hi public | Hindi | 1.33 h | 468 | 6.4s / 5.0s | 54/46 | 202 | 11/3 | 315 | Conversational, spontaneous | Lattice (accepted orthographic variants) |
| Monsoon hi private | Hindi | 4.47 h | 1,571 | 6.6s / 5.3s | 55/45 | 295 | 12/3 | 582 | Conversational, spontaneous | Lattice (accepted orthographic variants) |

The data is sourced from unscripted dual-channel spontaneous conversations, with clips segmented from a single channel so that each clip carries one speaker. Along with the fields reported in the table, each clip also records occupation, education, marital status, income band, handset brand, current city and years in the current district.

Five clips from the public Indian English split, with the metadata each one carries:

29-year-old woman, West Tripura, Tripura. Student, samsung SM-G781B.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/41512950_850_59_859_68.wav"></audio>

32-year-old woman, Satna, Madhya Pradesh. Unemployed, samsung SM-E146B.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/04646786_238_95_252_63.wav"></audio>

22-year-old man, Rohtas, Bihar. Student, motorola moto g54 5G.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/19508278_139_86_149_79.wav"></audio>

27-year-old woman, Warangal, Telangana. Unemployed, vivo V2247.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/41114168_788_37_795_21.wav"></audio>

57-year-old man, Puducherry. Private job, Xiaomi M2006C3LI.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/21485437_227_25_237_45.wav"></audio>

The English sets use standard string references, where the leaderboard's normaliser collapses most spelling variation. Hindi has far more of it, and no normaliser can resolve it, because the variants are not a fixed mapping between two conventions. The Hindi sets therefore ship a lattice: for each span of the transcript, a list of the spellings that are accepted as correct.

## Speaker coverage

Monsoon is small measured in hours and large measured in speakers. That is the design, and it is where most of the value sits.

**Speaker concentration and diversity beyond the fields above.**

| | [hi public](https://huggingface.co/datasets/VoiceArena/Monsoon_hi_test) | hi private | [en-IN public](https://huggingface.co/datasets/VoiceArena/Monsoon_en_IN_test) | en-IN private |
| :--- | :--- | :--- | :--- | :--- |
| Segments per speaker (mean) | 1.61 | 1.56 | 1.46 | 1.48 |
| Speakers with a single segment | 261 | 994 | 956 | 924 |
| Audio per speaker (median) | 8.34 s | 8.28 s | 12.36 s | 12.39 s |
| Share held by top 10 speakers | 6.8% | 3.1% | 2.8% | 2.9% |
| Current cities | 289 | 814 | 641 | 584 |
| Device manufacturers | 18 | 25 | 23 | 20 |

Three properties follow, and each is a claim about variance rather than volume.

- **No voice carries the score:** The ten largest contributors account for between 2.8% and 6.8% of total duration, and more than half of all speakers appear exactly once. A result on Monsoon is an average over hundreds of distinct voices, not a small number of talkers recorded at length. Test sets of comparable duration are usually constructed the other way.

- **No region or handset carries it either:** The Indian English public set draws on 428 native districts across 30 states and union territories; the Hindi sets, being a Hindi-belt language, concentrate more tightly but still span 202 and 295 districts. Recordings come from 315 to 582 distinct device models, with no single model exceeding 2.1% of segments in any subset. Corpora collected on standardised hardware overfit to one microphone response; this one cannot.

- **Indian English here is not one accent:** This is English as it is spoken across the country, not the English of one region. All six zones are represented: in the public set, 35% of segments are contributed by southern speakers, 18% from the East, 18% from Central, 16% from the North and 11% from the West. The accent variation that follows from that spread is recorded in the metadata rather than asserted.

### Metadata fields

Monsoon ships 18 columns per segment, of which 12 are metadata, where most public ASR test sets ship an identifier, a transcript and a duration. Demographic fields are complete or near-complete; contributors consented to this use.

| Group | Fields |
| :--- | :--- |
| Segment | `id`, `audio`, `audio_length_s`, `language` |
| Reference | `lattice` (Hindi) or `text` (Indian English) |
| Speaker | `speaker_id`, `gender`, `date_of_birth` |
| Background | `occupation`, `educational_background`, `marital_status`, `income` |
| Geography | `native_district`, `native_state`, `current_city`, `years_spent_in_current_district` |
| Recording | `device_manufacturer`, `device_model` |

The two languages have different geographic shapes, and the shape is informative. The Hindi sets concentrate in the Hindi belt, with Uttar Pradesh accounting for roughly 40% of speakers, which is what a Hindi corpus sampled by population should look like. The Indian English sets are much flatter: no state exceeds 13%, and a third of speakers come from outside the eight largest. Public and private halves match closely on both.

Indian state boundaries were drawn along linguistic lines, so district and state carry real accent signal, which is why these fields are released rather than summarised away. Analysis of this kind has been reported at scale for Indian ASR [2]: district-level error rates spanning roughly 4% to 44%, with underrepresented regions well behind the Hindi belt and the metros, plus disaggregation by audio quality, speaking rate, utterance duration, gender, age and device. Those runs were on a closed benchmark. Monsoon makes the same class of analysis possible on a public leaderboard test set.

## Collection and quality control

![monsoon_collection_and_annotation_pipeline](https://storage.googleapis.com/research_team_data/blog_figures/flowchart%201.png)


Broad geographic coverage requires recruitment across hundreds of districts rather than longer sessions from fewer speakers, and distributed recruitment at this scale introduces failure modes that a smaller collection does not face: contributors gaming the task, played-back audio submitted as live speech, and inattentive annotation. Each is addressed by an explicit check.

- **Recruitment and recording:** Contributors were recruited through the Voice Arena community, a global digital platform whose reach extends into the rural and semi-urban districts that speech corpora rarely cover. Pairs then recorded two-person conversations over a peer-to-peer interface, dual-channel, on assigned everyday topics. Contributors used their own handsets and their own connections. Many of those are low-end devices on unstable bandwidth, which is why that condition is present in the released audio rather than filtered out of it. Prospective contributors completed a language proficiency screening before being granted recording access, were compensated, and provided informed consent covering use in training and distribution. A per-speaker duration cap, calibrated per language to the population size and geographic distribution of its speakers, prevented a small number of prolific contributors from dominating a language or region; more than half of the speakers in these sets contribute exactly one segment.

- **Elicitation:** Eliciting spontaneous speech at scale presents its own difficulty, as contributors tend to produce short and sparse responses without structured guidance. Each conversation was therefore seeded with an open-ended narrative cue and progressively revealed follow-up questions, spanning domains including travel, healthcare, agriculture, education and digital services, guiding the exchange toward extended description without scripting it. Candidate topics were generated with large language models, then reviewed and localised by native-speaker linguists.

- **Quality control:** Every recording passed a set of gating checks prior to transcription. The spoken language was verified against the assigned language using language identification models trained on human-annotated data across more than 30 languages. Speaker gender was confirmed against the self-reported label using a dedicated classifier, applied as corroboration of the self-report rather than as a replacement for it. A further model distinguished genuine spontaneous conversation from pre-recorded or played-back audio. Signal-to-noise ratio estimation removed recordings degraded beyond intelligibility, while natural environmental background noise was deliberately preserved so that the acoustic realism of in-the-wild speech is retained. Recordings clearing these checks were segmented by voice activity detection, split at two seconds of continuous silence or at a fifteen-second soft cap closed at the next detected silence. Segmentation was applied independently per channel, so every segment is single-speaker and single-channel. Segments then passed a DNSMOS P.808 check.

- **Transcription:** Reference transcripts are human work. A first draft was generated by internal ASR models trained on in-domain data, none of which appear on any public leaderboard, so no system evaluated on these sets contributed to the references it is scored against. Every subsequent stage was performed by native-speaking linguists under a five-level protocol built on a strict separation of labour, in which each correction round is followed by an independent verification round performed by a different annotator, so no linguist audits their own output. A linguist first corrects the draft segment by segment against the acoustic signal; a second re-verifies it and flags residual disagreements. Subsequent levels iterate this cycle with fresh annotators, progressively resolving ambiguous phonetic realisations, code-switching boundaries, named entities, and orthographic consistency across spelling variants. Numerals are written as words, so that the transcript corresponds directly to what was spoken. Segments still flagged at the final level were returned for re-transcription before admission. Annotator behaviour was monitored automatically throughout, flagging submissions containing characters outside the target script, unnatural character or word repetitions, and unusually low or high edit counts.

## Regional variation

What follows is one example, run on the public Indian English split, to show the kind of evaluation the metadata makes possible. It is not the finding the sets exist to deliver; it is an illustration of what becomes answerable once every clip carries a speaker.

Eight models on the leaderboard land between 4.81 and 4.99 WER on this set. That is 0.18 points from best to worst, inside what five hours can resolve. Ranked on the corpus, they are the same model.

Grouping speakers by region tells a different story. Each speaker's native district is rolled up to its zonal council, the Ministry of Home Affairs grouping of Indian states, giving five well-sampled zones. [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) varies by 0.46 points across them. [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), fourteen hundredths of a point behind it on the corpus, varies by 1.68, running 4.38 in the Central zone against 6.06 in the East. Two systems that are indistinguishable on the leaderboard differ almost fourfold in how much their accuracy depends on where the speaker is from.

![corpus wer against zone range](https://storage.googleapis.com/research_team_data/csv_files/corpus_wer_versus_zone_range_coloured_by_zone.png)

Which zone is hardest is not fixed either. [ibm-granite/granite-speech-3.3-2b](https://huggingface.co/ibm-granite/granite-speech-3.3-2b) is worst in the North, [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) in the South, [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) in the East. If a single region were simply harder to transcribe, every model would rank the zones the same way. They do not, which points at the models rather than the audio.

Region is one of twelve recorded attributes, and the zones above are a coarse rollup of 428 districts. The same breakdown runs on age, education, occupation and handset, and the released files carry everything needed to reproduce it. None of it is available for a test set that records only what was said.

## Orthographic variation in Hindi

English orthographic variation is bounded. British against American spelling, punctuation, casing, digits against words: a normaliser can map most of it to a single form, and the leaderboard's does. Hindi is not bounded in the same way. Everyday speech is heavily code-mixed, English-origin words have no settled Devanagari spelling, and compound forms are written joined or separated according to preference. A single phrase can have ten or more valid written forms, and no fixed mapping collapses them, because there is no canonical side to map to.

![english_normaliser](https://storage.googleapis.com/research_team_data/blog_figures/flowchart%202.png)

Scored with a single reference, WER rewards a system for producing the spelling the annotator happened to choose. Two systems that recognised the audio equally well can differ by several points on orthography alone.

The Hindi sets therefore ship a lattice: for each span of the transcript, the set of written forms accepted as correct. Building it is manual work. Candidate variants are drawn from multiple ASR transcripts of the same audio and expanded with language models, then native-speaker linguists decide which are valid for that utterance and prune the rest, so only forms consistent with what was said are admitted.

![hindi_oiwer_scoring](https://storage.googleapis.com/research_team_data/blog_figures/flowchart%203a.png)

Thus, for Hindi we report the [Orthographically-Informed Word Error Rate (OIWER)](https://huggingface.co/papers/2603.00941), introduced by AI4Bharat, instead of WER. A hypothesis is aligned against the accepted set at each span, so any admitted form counts as correct and only genuine recognition errors are charged.

To quantify the effect, the same hypotheses were scored twice. Flattening each lattice to its first variant per span yields a single string reference of the kind a conventional benchmark provides; any of the admitted variants would serve equally well, and a different choice would yield a different reference. Scored against the flattened reference, error rates rise for every system, and they do not rise uniformly. Rankings change as a consequence. The figure below shows two pairs of systems that reverse order between the two references: under a single reference a system is rewarded in part for reproducing the annotator's orthography, whereas the lattice scores only recognition.

![WER against OIWER on Monsoon hi](https://storage.googleapis.com/research_team_data/csv_files/hindi_oiwer_flips.png)

We also open source our implementation, [voi-oiwer](https://pypi.org/project/voi-oiwer), so every result on these sets can be reproduced directly.

## Getting evaluated

For the private splits, get your model on the Open ASR Leaderboard and the Hugging Face team will run the evaluation. As before, the process for adding a model to the leaderboard takes place on the [Open ASR Leaderboard GitHub](https://github.com/huggingface/open_asr_leaderboard):

1. Open a pull request, a [model checklist](https://github.com/huggingface/open_asr_leaderboard/blob/main/.github/PULL_REQUEST_TEMPLATE.md#new-model-checklist) will appear. As before, you should report your results on the public datasets.
2. We will verify the results on the public sets and compute the metrics on the private ones.
3. Confirm the results we've obtained.

Indian-English joins the [main leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) as `Voice Arena Monsoon`, in the default column set rather than as an opt-in toggle, so it contributes to the headline Average WER for every model. The private split feeds the aggregated `Private (conversational)` column alongside [Appen and DataoceanAI data](https://huggingface.co/blog/open-asr-leaderboard-private-data). The public and private Hindi appears in the [Multilingual tab](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), where a model is ranked only if it supports every selected language, making that column a like-for-like comparison. Alternative, select "Hindi" from the "Language dataset breakdown" dropdown menu.

## What comes next

Hindi is a sharp case of a general problem. Any language written more than one way, spoken by people a benchmark has not sampled, carries both of the failures described here. These sets do not fix them. What they add is a way to see them: a test set on a leaderboard the field already watches, carrying enough about each speaker and each reference that a difference between two systems can be traced to who was talking and how they write it, instead of disappearing into one number.

These four sets are part of Monsoon, Voice Arena's broader dataset initiative for the Global South.
