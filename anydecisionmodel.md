---
title: "Introducing AnyDecisionModel and AnySpeechRecognizer: Open Models for Decisions and Speech on Apple Platforms"
thumbnail: /blog/assets/anydecisionmodel/banner.png
authors:
- user: mattt
  guest: true
---

Language models can do a lot.
They write, summarize, translate, and call tools.
But they're only a small part of what machine learning can do.

Say a customer calls your support line.
First, you want to know what they said.
Then, you want to decide where to send them.
The first job is speech recognition; the second is a decision.
You could hand both to a language model,
but there are models built for each.

In our [AnyLanguageModel 1.0 recap](https://huggingface.co/blog/anylanguagemodel-1-0),
we looked at a year of local and hosted models in Swift.
With AnyLanguageModel and swift-huggingface now on their 1.x releases,
we're extending the same approach to speech and decisions:
familiar Swift APIs with a choice of model.
Today we're announcing two packages:

- [AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer)
  implements Apple's new speech-to-text API with a choice of recognition engine,
  including NVIDIA's Parakeet and Alibaba's Qwen3-ASR running locally on Core ML,
  and adds speaker diarization to track who speaks when.
- [AnyDecisionModel](https://github.com/mattt/AnyDecisionModel)
  answers yes-or-no, multiple-choice, and rating-scale questions
  with typed results and probabilities.
  It derives each answer from the model's probabilities
  instead of generating text.

Both packages can download open models from the Hugging Face Hub
and run them on device.

## AnySpeechRecognizer

Computers have been transcribing speech for decades,
and in the past few years, they've gotten very good at it.
Apple platforms already have an API for it.
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
If your code uses that shared API, changing the import is enough.
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
has a complete example
and [lists the differences](https://github.com/mattt/AnySpeechRecognizer#differences-from-apples-speech-framework)
from Apple's framework.

### Get to know the models

**Parakeet v3** is NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
It's a polyglot:
it recognizes 25 European languages and works out which one it's hearing,
so you don't pass a locale.

**Parakeet Ultra** is Moondream's [post-trained take on v3](https://huggingface.co/moondream/parakeet-ultra).
It was more accurate in most of our measurements.

**Parakeet Unified** is NVIDIA's
[English-only Parakeet](https://huggingface.co/nvidia/parakeet-unified-en-0.6b).

All three Parakeets run on the package's own Core ML runtime,
which supports file and live input, volatile results, word timing,
and word confidence.

**Qwen3-ASR** is Alibaba's speech model,
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

Qwen3-ASR gives one final result for each segment of up to 40 seconds,
without volatile results, word timing, or confidence.
It runs on the GPU and is slower than Parakeet,
so it's better suited to transcribing files than live input.

To add another engine, implement `SpeechRecognitionModel` for its configuration
and `SpeechRecognitionExecutor` for one module's session.
The README includes a [small provider](https://github.com/mattt/AnySpeechRecognizer#custom-providers)
that reports placeholder text to help you get started.

### Core ML models on the Hub

To run these models in apps,
we converted them to Core ML and published them on the Hub:

| Model | Core ML conversion | Original checkpoint | License |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v3 | [mattt/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/mattt/parakeet-tdt-0.6b-v3-coreml) | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC BY 4.0 |
| Parakeet Ultra | [mattt/parakeet-ultra-coreml](https://huggingface.co/mattt/parakeet-ultra-coreml) | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | CC BY 4.0 |
| Parakeet Unified EN 0.6B | [mattt/parakeet-unified-en-0.6b-coreml](https://huggingface.co/mattt/parakeet-unified-en-0.6b-coreml) | [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| Qwen3-ASR 0.6B | [mattt/qwen3-asr-0.6b-coreml](https://huggingface.co/mattt/qwen3-asr-0.6b-coreml) | [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Apache 2.0 |
| Qwen3-ASR 1.7B | [mattt/qwen3-asr-1.7b-coreml](https://huggingface.co/mattt/qwen3-asr-1.7b-coreml) | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Apache 2.0 |
| Streaming Sortformer 4spk v2 | [mattt/diar-streaming-sortformer-4spk-v2-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2-coreml) | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | CC BY 4.0 |
| Streaming Sortformer 4spk v2.1 | [mattt/diar-streaming-sortformer-4spk-v2.1-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml) | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

Each repository records its source checkpoint
and a checksum for every file,
and the package pins it to a fixed revision.
The Parakeet repositories add an 8-second encoder for live input,
which cuts live analysis time by about a third
at the cost of a somewhat higher word error rate.
The repository also has the conversion workflows,
most of which follow FluidInference's [mobius](https://github.com/FluidInference/mobius) scripts,
and checks each conversion against the original model.

Downloads happen only when your app asks for them:
about 600 MB for each Parakeet, 0.9 to 2.4 GB for Qwen3-ASR,
and about 250 MB for Sortformer.
Files go in the standard Hugging Face cache,
so a Mac app outside the sandbox shares them with Python and other Hugging Face tools.

### Speaker diarization

When you transcribe a conversation,
you often want to know who said what.
That's speaker diarization,
and Apple's Speech framework doesn't have a module for it,
so AnySpeechRecognizer adds one.
`SpeakerDiarizer` runs NVIDIA's
[Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
on Core ML alongside an Apple or Parakeet transcriber,
on the same timeline:

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
even when they talk over each other.
A `SpeakerTimeline` collects its results
and attributes each timed word to a speaker,
so you can print a transcript by speaker turn.
Like image support in AnyLanguageModel,
this goes beyond what Apple's API offers today.

The diarizer uses Sortformer v2 by default.
You can also choose [v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
which NVIDIA trained with more meeting data
and which was more accurate on meetings in our measurements, too.

### How well does it work?

We measured the package with its `hear` command-line tool
on an M4 Max Mac Studio with macOS 27.0.1.
Here are the word error rates on LibriSpeech files of 3 to 8 minutes:

| Transcriber | test-clean | test-other |
| --- | --- | --- |
| Parakeet v3, file | 1.55% | 2.78% |
| Parakeet v3, live | 4.73% | 4.29% |
| Parakeet Ultra, file | 1.66% | 2.55% |
| Parakeet Ultra, live | 2.65% | 3.28% |
| Parakeet Unified, file | 1.43% | 2.19% |
| Parakeet Unified, live | 1.45% | 2.42% |
| Apple's transcriber, file | 2.12% | 3.97% |

Parakeet transcribes an hour of audio
in about 20 to 40 seconds from a file,
or about 1.5 to 2.5 minutes live,
with about 100 MB of memory once Core ML has compiled the model.
Qwen3-ASR is much slower and needs a few gigabytes of memory.
Meetings are hard for everyone:
on AMI recordings, where people talk over each other,
every engine we tested had a word error rate above 23%.

These numbers come from one machine and a few datasets,
so measure with your own hardware and audio.
`hear evaluate` works on your own files,
and the repository's `natural-speech-eval.py` tool
[reproduces these measurements](https://github.com/mattt/AnySpeechRecognizer#accuracy-and-performance).
So far, we've tested only on macOS 27, not on an iPhone.

Next, we'd like to add more engines, including Core AI on macOS 27 and iOS 27.

## AnyDecisionModel

AnyDecisionModel is a different story.
There's no Apple framework to map onto,
and decision models aren't new, exactly.
It's only a decision model if it comes from the Décision region of France;
otherwise, it's sparkling zero-shot classification.
What's new is the name, a lot of hype,
and a crop of models trained to make decisions.
We think that's where things are going,
so we wanted a Swift API that's ready for them.

Apps often ask language models to make decisions.
Is this message urgent?
Which team should handle this ticket?
How frustrated is this customer?
The usual approach is to prompt a language model
and parse the text that it generates.
That gives you one answer,
but doesn't show the probabilities it assigned to the alternatives.

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
Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) models
are an example of a newer approach:
a model with a trained decision head
that turns a state and a schema of typed questions into decisions,
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

## Get started

Both packages require Swift 6.2 or later and macOS 26 or iOS 26 or later.
The Parakeet and Sortformer runtimes and the MLX backend require Apple silicon.
AnyDecisionModel's MLX build tasks use Xcode's Swift 6.4 or later
and the Metal toolchain; Core AI requires Xcode 27 and an OS 27 runtime.

Add the packages to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnySpeechRecognizer", from: "0.1.0"),
    .package(
        url: "https://github.com/mattt/AnyDecisionModel",
        from: "0.6.0",
        traits: ["MLX"]
    ),
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

- [AnySpeechRecognizer on GitHub](https://github.com/mattt/AnySpeechRecognizer)
- [AnyDecisionModel on GitHub](https://github.com/mattt/AnyDecisionModel)
- [AnyDecisionModel-Examples on GitHub](https://github.com/mattt/AnyDecisionModel-Examples)
- [AnyLanguageModel on GitHub](https://github.com/huggingface/AnyLanguageModel)
- [swift-huggingface on GitHub](https://github.com/huggingface/swift-huggingface)

_Thanks to NVIDIA for Parakeet, Sortformer, and NeMo, to Moondream for Parakeet Ultra,
to FluidInference, whose FluidAudio and mobius projects
documented the conventions that the Parakeet runtime and the Core ML conversions follow,
to Cloudflare for Clef,
and to everyone who contributed to AnyLanguageModel,
whose patterns both packages borrow._
