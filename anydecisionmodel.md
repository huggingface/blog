---
title: "Introducing AnyDecisionModel and AnySpeechRecognizer: Open Models for Decisions and Speech on Apple Platforms"
thumbnail: /blog/assets/anydecisionmodel/banner.png
authors:
- user: mattt
  guest: true
---

Language models write text,
but apps often ask them to make decisions.
Is this message urgent?
Which team should handle this ticket?
How frustrated is this customer?
The usual approach is to prompt a language model
and parse the text that it generates.
That gives you one answer,
but doesn't show the probabilities it assigned to the alternatives.

In our [AnyLanguageModel 1.0 recap](https://huggingface.co/blog/anylanguagemodel-1-0),
we looked at a year of local and hosted models in Swift.
With AnyLanguageModel and swift-huggingface now on their 1.x releases,
we're extending the same approach to decisions and speech:
familiar Swift APIs with a choice of model.
Today we're announcing two packages:

- [AnyDecisionModel](https://github.com/mattt/AnyDecisionModel)
  answers yes-or-no, multiple-choice, and rating-scale questions
  with typed results and probabilities.
  It derives each answer from the model's probabilities
  instead of generating text.
- [AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer)
  implements Apple's new speech-to-text API with a choice of recognition engine,
  including NVIDIA's Parakeet and Alibaba's Qwen3-ASR running locally on Core ML,
  and adds speaker diarization to track who speaks when.

Both packages can download open models from the Hugging Face Hub
and run them on device.
For AnySpeechRecognizer, we converted the models to Core ML
and published them on the Hub.

## AnyDecisionModel

Here's what a decision looks like in Swift:

```swift
import AnyDecisionModel

enum Department: String, Choosable {
    case billing, technical, sales
}

let ticket = "I was charged twice for my order, and now the app logs me out when I try to ask for a refund."

let model = MLXDecisionModel(modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit")
let session = DecisionSession(model: model, state: .text(ticket))

let urgent = try await session.probability(of: "Does this convey urgency?")
let team = try await session.choice("Which team?", from: Department.self)
let anger = try await session.score(
    "How frustrated?",
    levels: ["Calm", "Frustrated", "Very angry"]
)
```

`urgent` is a `Double`.
`team` is a `Choice<Department>` containing the selected case,
a probability for each option, and a confidence value.
`anger` is a `Score` containing the expected level,
a probability for each level, and a confidence value.
Because `Department` conforms to `Choosable`,
`team.value` is a `Department` that you can use in an exhaustive `switch`.

If you've used Foundation Models or AnyLanguageModel,
the model-and-session pattern will look familiar.
A `DecisionSession` holds a single state, which can be text or JSON,
and every question is about that state.
Questions are independent:
an answer never becomes input to a later question.
To ask several questions at once,
pass them to `decide(_:)` as a batch.

For each question, the MLX backend builds a prompt
with a one-token label for each allowed answer,
then reads the probability of each label from one forward pass.

### Try it out: Emoji Magnet

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/anydecisionmodel/emoji-magnet.mp4" title="Emoji Magnet sorting emoji for seven phrases, from &quot;things you wear&quot; to &quot;junk food&quot;"></video>

Emoji Magnet is a macOS app with a text field above a pile of 343 emoji.
Type a phrase, such as "things you wear",
and the emoji that match rise out of the pile.
Refine it to "things you wear in winter",
and the emoji that no longer match fall back.

The app has no keyword rules.
When you pause typing, it asks the model one yes-or-no question for each emoji:

```
Statement: a dog fits the description "pet". Is this statement true?
```

The session's state holds the phrase.
The demo above uses the hosted Jev backend,
which answers all 343 questions in one request,
so the emoji settle in about half a second for each phrase.
Without a `TYPESAFE_AI_API_KEY`, the app runs Qwen3 4B on your Mac with the MLX backend.
That backend uses prefix caching to evaluate the state once,
then answers 16 questions per forward pass.
On an M4 Max, a group of 16 takes about 0.45 seconds after the model loads.
A full pass over all 343 emoji takes about 10 seconds,
and each group's results appear as soon as they arrive.

Emoji Magnet is in the [AnyDecisionModel-Examples](https://github.com/mattt/AnyDecisionModel-Examples) repository.

### Decision heads and Clef

Reading next-token probabilities works with general-purpose language models,
but those models weren't trained to make decisions.
Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) models were.
Clef turns a state and a schema of typed questions into decisions,
returning probabilities for all allowed options across the questions
in one forward pass.
[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) is the 9B variant.

AnyDecisionModel runs Clef-Flash on a Mac with MLX:

```swift
let model = MLXHeadDecisionModel(
    modelID: "mlx-community/clef-flash-4bit",
    head: .jointSchema
)
```

You use the same sessions and questions,
but `MLXHeadDecisionModel` reads each decision from Clef's trained decision head
instead of from next-token probabilities.
The [4-bit model](https://huggingface.co/mlx-community/clef-flash-4bit) is a 6.2 GB download,
and its model card recommends a Mac with at least 16 GB of memory.
Clef can read images and video,
but this backend accepts only text and JSON states.

### Calibrating and checking probabilities

Raw probabilities from language models are often overconfident.
The local backends accept a `Calibration`,
which applies a temperature to every answer and a bias to yes-or-no answers.
Use separate datasets to fit the calibration and evaluate the results.

Language models can also favor an option because of its position in a list.
On the MLX, Core AI, and llama.cpp next-token backends,
turn on `rotationDebiasing` to ask a choice question
once for each rotation of the options and average the probabilities.
That costs one prompt evaluation per rotation.

Each local choice and score also includes a confidence value
based on the entropy of its probabilities.
It measures how concentrated the probabilities are,
not whether the answer is right,
so check the error rate on your own data before you pick a threshold.
The next-token backends' diagnostics report how much probability
the model assigned to the allowed answers.
A low value means it assigned most of the probability elsewhere.

### Backends

AnyDecisionModel has six backends:

- **MLX** runs a small language model on Apple silicon
  with [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
  and downloads it from the Hub.
- **MLX decision heads** run a model with its own trained decision head,
  such as Clef.
- **Core AI** does the same as MLX with a Core AI model on macOS 27 and iOS 27.
- **llama.cpp** reads decisions from a local GGUF model
  through [llama.swift](https://github.com/mattt/llama.swift).
- **Jev** calls TypeSafe's Jev through the [System One API](https://docs.typesafe.ai/api/system-one),
  or any server that implements it, such as [Kev](https://github.com/jaredpalmer/kev).
- **Ollama** calls a System One decision model on a local Ollama server
  or in Ollama's cloud.
  It requires Ollama 0.35.0 or later, or 0.40.5 or later for cloud models.

The next-token backends label answers "yes" and "no", A to Z, or 0 to 9,
so a choice can have at most 26 options, and a score at most 10 levels.
Decision heads have no such limit.

Like AnyLanguageModel, the package uses Swift package traits.
Without traits, it has no third-party dependencies.
You get the core types and the Jev and Ollama backends,
which also build on Linux.
Enable the `MLX`, `CoreAI`, or `Llama` trait for the local backends.

### Status

AnyDecisionModel is at [0.6.0](https://github.com/mattt/AnyDecisionModel/releases/tag/0.6.0),
which adds Ollama and llama.cpp support
and works with swift-huggingface 1.x.
The example calibration values in the README come from a fit on two tasks,
so don't assume they'll work for yours.
Typed output alone doesn't protect against prompt injection.

For now, an app can't enable the `MLX` traits
of both AnyDecisionModel 0.6.0 and
[AnyLanguageModel 1.1.1](https://github.com/huggingface/AnyLanguageModel/releases/tag/1.1.1):
they require mlx-swift-lm 3.31.x and 3.32.x respectively.

## AnySpeechRecognizer

The same approach works for speech, too.

macOS 26 and iOS 26 introduced `SpeechAnalyzer`,
Apple's new speech-to-text API.
You add modules such as `SpeechTranscriber` and `SpeechDetector` to an analyzer,
give it a file or a stream of audio buffers,
and read results from async sequences as they arrive.
`AssetInventory` installs the models,
so your app decides when downloads happen.

It's a well-designed API,
but the only engine behind it is Apple's.
AnySpeechRecognizer keeps the API and lets you choose the engine:

```diff
- import Speech
+ import AnySpeechRecognizer
```

The package mirrors `SpeechAnalyzer`, `SpeechTranscriber`, `DictationTranscriber`,
`SpeechDetector`, and `AssetInventory`,
with Apple's Speech framework as the default engine.
If your code uses that shared API, changing the import is enough;
the differences and unsupported features are listed below.
To use Parakeet instead,
change only how you create the transcriber:

```swift
let transcriber = SpeechTranscriber(
    model: ParakeetSpeechModel(variant: .v3),
    transcriptionOptions: [],
    reportingOptions: [.volatileResults],
    attributeOptions: [.audioTimeRange]
)
```

The steps for installing assets, running the analyzer,
and reading results stay the same.
The [README](https://github.com/mattt/AnySpeechRecognizer#transcribe-a-file)
has a complete example.
The package defines its own analyzer and module types to match Apple's.
If a file imports both `Speech` and `AnySpeechRecognizer`,
you must qualify the names they share.
The README [lists the differences](https://github.com/mattt/AnySpeechRecognizer#differences-from-apples-speech-framework).

`.v3` is NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
which recognizes 25 European languages and detects the language automatically,
so it takes no locale.
`.ultra` is Moondream's [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra),
a post-trained version of v3 that was more accurate in most of our measurements.
`.unified` is NVIDIA's
[Parakeet Unified EN 0.6B](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
an English-only model under the NVIDIA Open Model License.
The package runs all three on Core ML with its own Parakeet runtime,
which supports file and live input, volatile results, word timing,
and word confidence.

`QwenSpeechModel` runs Alibaba's Qwen3-ASR,
in [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)
and [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) sizes.
It recognizes 30 languages and 22 Chinese dialects,
including Mandarin, Cantonese, Japanese, and Korean, which Parakeet doesn't cover.
It detects the language automatically, or you can set one:

```swift
let transcriber = SpeechTranscriber(
    model: QwenSpeechModel(variant: .large, language: Locale.Language(identifier: "yue")),
    transcriptionOptions: [],
    reportingOptions: [],
    attributeOptions: []
)
```

Qwen3-ASR transcribes audio in segments of up to 40 seconds
and gives one final result for each segment,
without volatile results, word timing, or confidence.
It runs on the GPU and is slower than Parakeet,
so it's better suited to transcribing files than live input.

To add another engine, implement `SpeechRecognitionModel` for its configuration
and `SpeechRecognitionExecutor` for one module's session.
The README includes a [small provider](https://github.com/mattt/AnySpeechRecognizer#custom-providers)
that reports placeholder text to help you get started.

### Models from the Hub

AnySpeechRecognizer runs seven Core ML models that we converted and published on the Hub:

| Model | Core ML conversion | Original checkpoint | License |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v3 | [mattt/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/mattt/parakeet-tdt-0.6b-v3-coreml) | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC BY 4.0 |
| Parakeet Ultra | [mattt/parakeet-ultra-coreml](https://huggingface.co/mattt/parakeet-ultra-coreml) | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | CC BY 4.0 |
| Parakeet Unified EN 0.6B | [mattt/parakeet-unified-en-0.6b-coreml](https://huggingface.co/mattt/parakeet-unified-en-0.6b-coreml) | [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| Qwen3-ASR 0.6B | [mattt/qwen3-asr-0.6b-coreml](https://huggingface.co/mattt/qwen3-asr-0.6b-coreml) | [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Apache 2.0 |
| Qwen3-ASR 1.7B | [mattt/qwen3-asr-1.7b-coreml](https://huggingface.co/mattt/qwen3-asr-1.7b-coreml) | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Apache 2.0 |
| Streaming Sortformer 4spk v2 | [mattt/diar-streaming-sortformer-4spk-v2-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2-coreml) | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | CC BY 4.0 |
| Streaming Sortformer 4spk v2.1 | [mattt/diar-streaming-sortformer-4spk-v2.1-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml) | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

FluidInference published earlier Core ML conversions of Parakeet v3, Ultra,
and both Sortformer versions.
The conversion workflows are in the repository;
those for Parakeet and Sortformer v2 follow FluidInference's
[mobius](https://github.com/FluidInference/mobius) scripts.
The tools document weight comparisons with FluidInference's v3 and Ultra conversions.
Here's what our model repositories include:

- Each one records its source checkpoint and revision,
  the tools that converted it,
  and the size and SHA-256 digest of every file.
  The package pins each repository at a fixed revision.
- Each one carries its base model's license and attribution,
  in both the model card and the Core ML metadata.
- The Parakeet repositories add an 8-second encoder for live input,
  which cuts live analysis time by about a third
  at the cost of a somewhat higher word error rate.
- The Parakeet Unified and Sortformer v2.1 repositories include the license agreement
  and the notice file that NVIDIA's license requires for redistribution.

The repository also compares each Parakeet and Sortformer conversion
with NVIDIA's NeMo toolkit on the same input,
and each Qwen3-ASR conversion with Hugging Face Transformers.

When you request an installation, the package downloads the pinned files
and checks each file's size and SHA-256 digest.
The base Parakeet files are about 632 MB each for v3 and Ultra,
and about 614 MB for Unified.
Live transcription can also use the 8-second encoder, which adds about 591 MB.
The Qwen3-ASR downloads are about 0.94 GB for 0.6B and 2.36 GB for 1.7B.
Each Sortformer version is about 246 MB
and installs only when you add a diarizer.
Analysis never triggers a download:
if the assets are missing, you get an error that tells you what to install.

The runtime loads models directly from a cache
with the standard Hugging Face layout,
so each file is stored once.
The package respects `HF_HUB_CACHE` and `HF_HOME`,
and a Mac app outside the sandbox shares `~/.cache/huggingface/hub`
with Python and other Hugging Face tools.

### Speaker diarization

Apple's Speech framework has no diarization module,
so AnySpeechRecognizer adds one.
`SpeakerDiarizer` identifies who speaks when.
It runs NVIDIA's
[Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
on Core ML alongside an Apple or Parakeet transcriber.
Both use the same timeline:

```swift
let diarizer = SpeakerDiarizer(model: SortformerDiarizationModel())
if let request = try await AssetInventory.assetInstallationRequest(
    supporting: [transcriber, diarizer]
) {
    try await request.downloadAndInstall()
}
let analyzer = SpeechAnalyzer(modules: [transcriber, diarizer])
```

The diarizer tracks up to four speakers,
including overlapping speech.
A `SpeakerTimeline` collects its results
and attributes each timed word to a speaker,
so you can print a transcript by speaker turn.
Like image support in AnyLanguageModel,
this goes beyond what Apple's API offers today.

`SortformerDiarizationModel` uses Sortformer v2 by default.
`SortformerDiarizationModel(version: .v2_1)` uses
[v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
which NVIDIA trained with additional meeting data
and reports as more accurate on most of its meeting and telephone benchmarks.
Like Parakeet Unified, v2.1 is under the NVIDIA Open Model License
instead of CC BY 4.0.
Use of either model must be consistent with NVIDIA's Trustworthy AI terms,
and if you distribute its files, for example in an app bundle,
you must include the agreement and the notice file from its repository.
The post-processing presets include NeMo's CALLHOME parameters (the default)
and its DIHARD III parameters.
You can also choose the compute units.
On an M4 Max, `cpuAndGPU` processed audio about three times faster
than the default, `cpuAndNeuralEngine`,
but used about 530 MB of memory instead of about 135 MB.

### Status

AnySpeechRecognizer is at [0.1.0](https://github.com/mattt/AnySpeechRecognizer/releases/tag/0.1.0).

We used the package's `hear` command-line tool to measure its performance
on an M4 Max Mac Studio with macOS 27.0.1,
using natural speech from LibriSpeech and the AMI meeting corpus.
Testing on natural speech exposed a recurring problem:
after the end of a sentence, a window of audio sometimes decoded as silence.
The runtime now starts each file-mode window at the quietest nearby moment,
and when a window goes blank, it decodes the gap again in a window that starts there.

On LibriSpeech files of 3 to 8 minutes,
we measured the following word error rates
after lowercasing and removing punctuation:

| Transcriber | test-clean | test-other |
| --- | --- | --- |
| Parakeet v3, file mode | 1.55% | 2.78% |
| Parakeet Ultra, file mode | 1.66% | 2.55% |
| Parakeet Unified, file mode | 1.43% | 2.19% |
| Apple's transcriber | 2.12% | 3.97% |

In live mode, v3 and Ultra's error rates on those files were 2.65% to 4.73%;
Unified's were 1.45% on test-clean and 2.42% on test-other.
Live mode uses the 8-second encoder when it's installed;
with the 15-second encoder, live analysis takes longer,
and v3 and Ultra's error rates were 2.03% to 3.17%.
On four AMI test meetings using headset-mix audio,
where people often talk over each other,
every tested engine had a word error rate above 23%.
Parakeet v3 and Ultra's word start times had a median error of 60 ms
when compared with forced alignments of LibriSpeech.
Unified's word start times are currently about 300 ms late.

For Parakeet transcription without a diarizer,
file mode processes an hour of audio in 18 to 31 seconds with v3 and Ultra
and in 33 to 42 seconds with Unified.
Live mode takes 93 to 112 seconds with v3 and Ultra
and 124 to 136 seconds with Unified.
Peak memory in those transcription runs is about 100 MB,
except while Core ML compiles a model, which can take about 680 MB.

Qwen3-ASR is much slower than Parakeet and needs a few gigabytes of memory,
so it's better suited to transcribing files,
especially in languages that Parakeet doesn't cover.

Parakeet sometimes decodes a window of speech as silence,
mostly in live mode and in meetings with overlapping speech.

With Parakeet v3 in the same analyzer,
Sortformer v2 had a diarization error rate of 25.73%
on the 16 AMI test meetings; v2.1 had 23.42%.
The score counts overlapping speech
and doesn't skip the moments around each change of speaker,
as some evaluations do.

These results come from one machine and a few datasets,
so measure with your own hardware and audio.
`hear evaluate` reports elapsed time and peak memory;
with `--reference` it also reports word error rate,
and with `--diarize --rttm` it reports diarization error rate.
The repository's `natural-speech-eval.py` tool downloads the test sets
and [reproduces these measurements](https://github.com/mattt/AnySpeechRecognizer#accuracy-and-performance).

So far, we've tested only on macOS 27.
We haven't yet tested Parakeet on a physical iPhone,
Qwen3-ASR on any iOS device,
the OS 26 runtime, or the minimum Swift 6.2 toolchain.
The package doesn't implement the older `SFSpeechRecognizer` API,
Apple's custom language models,
or the capture and asset input providers in OS 27.

Next, we'd like to add more engines, including Core AI on macOS 27 and iOS 27.

## Get started

Both packages require Swift 6.2 or later and macOS 26 or iOS 26 or later.
The MLX backend and the Parakeet and Sortformer runtimes require Apple silicon.
AnyDecisionModel's MLX build tasks use Xcode's Swift 6.4 or later
and the Metal toolchain; Core AI requires Xcode 27 and an OS 27 runtime.

Add the packages to your `Package.swift`:

```swift
dependencies: [
    .package(
        url: "https://github.com/mattt/AnyDecisionModel",
        from: "0.6.0",
        traits: ["MLX"]
    ),
    .package(url: "https://github.com/mattt/AnySpeechRecognizer", from: "0.1.0"),
]
```

For now, AnyDecisionModel's `CoreAI` trait can't be combined with AnySpeechRecognizer,
because coreai-kit 0.7.3 requires swift-huggingface below 1.0
and AnySpeechRecognizer requires 1.x.

Each README has more examples and details about requirements and backends.
AnySpeechRecognizer also includes a command-line tool, `hear`,
that transcribes local files, labels their speakers, and evaluates the results.
Every `hear` command requires `--backend apple`, `--backend parakeet`, or `--backend qwen`;
run `hear prepare` with the same backend and model options before analysis
to install the needed assets.

Try them in your apps and tell us what works and what doesn't.
Issues and pull requests are welcome on both repositories.

## Links

- [AnyDecisionModel on GitHub](https://github.com/mattt/AnyDecisionModel)
- [AnyDecisionModel-Examples on GitHub](https://github.com/mattt/AnyDecisionModel-Examples)
- [AnySpeechRecognizer on GitHub](https://github.com/mattt/AnySpeechRecognizer)
- [AnyLanguageModel on GitHub](https://github.com/huggingface/AnyLanguageModel)
- [swift-huggingface on GitHub](https://github.com/huggingface/swift-huggingface)

_Thanks to Cloudflare for Clef,
to NVIDIA for Parakeet, Sortformer, and NeMo, to Moondream for Parakeet Ultra,
to FluidInference, whose FluidAudio and mobius projects
documented the conventions that the Parakeet runtime and the Core ML conversions follow,
and to everyone who contributed to AnyLanguageModel,
whose patterns both packages borrow._
