---
title: "Introducing swift-huggingface: The Complete Swift Client for Hugging Face"
thumbnail: /blog/assets/swift-huggingface/banner.png
authors:
- user: mattt
  guest: true
---

Today, we're announcing [swift-huggingface](https://github.com/huggingface/swift-huggingface),
a new Swift package that provides a complete client for the Hugging Face Hub.

You can start using it today as a standalone package,
and it will soon integrate into swift-transformers as a replacement for its current `HubApi` implementation.

## The Problem

When we released [swift-transformers 1.0](https://huggingface.co/blog/swift-transformers) earlier this year,
we heard loud and clear from the community:

- **Downloads were slow and unreliable.**
  Large model files (often several gigabytes)
  would fail partway through with no way to resume.
  Developers resorted to manually downloading models and bundling them with their apps —
  defeating the purpose of dynamic model loading.
- **No shared cache with the Python ecosystem.**
  The Python `transformers` library stores models in `~/.cache/huggingface/hub`.
  Swift apps downloaded to a different location with a different structure.
  If you'd already downloaded a model using the Python CLI,
  you'd download it again for your Swift app.
- **Authentication is confusing.**
  Where should tokens come from?
  Environment variables? Files? Keychain?
  The answer is, _"It depends"_,
  and the existing implementation didn't make the options clear.

## Introducing swift-huggingface

swift-huggingface is a ground-up rewrite focused on reliability and developer experience.
It provides:

- **Complete Hub API coverage** — models, datasets, spaces, collections, discussions, and more
- **Robust file operations** — progress tracking, resume support, and proper error handling
- **Python-compatible cache** — share downloaded models between Swift and Python clients
- **Flexible authentication** — a `TokenProvider` pattern that makes credential sources explicit
- **OAuth support** — first-class support for user-facing apps that need to authenticate users
- **Xet storage backend support** _(Coming soon!)_ — chunk-based deduplication for significantly faster downloads

Let's look at some examples.

---

## Flexible Authentication with TokenProvider

One of the biggest improvements is how authentication works. The `TokenProvider` pattern makes it explicit where credentials come from:

```swift
import HuggingFace

// For development: auto-detect from environment and standard locations
// Checks HF_TOKEN, HUGGING_FACE_HUB_TOKEN, ~/.cache/huggingface/token, etc.
let client = HubClient.default

// For CI/CD: explicit token
let client = HubClient(tokenProvider: .static("hf_xxx"))

// For production apps: read from Keychain
let client = HubClient(tokenProvider: .keychain(service: "com.myapp", account: "hf_token"))
```

The auto-detection follows the same conventions as the Python `huggingface_hub` library:

1. `HF_TOKEN` environment variable
2. `HUGGING_FACE_HUB_TOKEN` environment variable
3. `HF_TOKEN_PATH` environment variable (path to token file)
4. `$HF_HOME/token` file
5. `~/.cache/huggingface/token` (standard HF CLI location)
6. `~/.huggingface/token` (fallback location)

This means if you've already logged in with `huggingface-cli login`,
swift-huggingface will automatically find and use that token.

## OAuth for User-Facing Apps

Building an app where users sign in with their Hugging Face account?
swift-huggingface includes a complete OAuth 2.0 implementation:

```swift
import HuggingFace

// Create authentication manager
let authManager = try HuggingFaceAuthenticationManager(
    clientID: "your_client_id",
    redirectURL: URL(string: "yourapp://oauth/callback")!,
    scope: [.openid, .profile, .email],
    keychainService: "com.yourapp.huggingface",
    keychainAccount: "user_token"
)

// Sign in user (presents system browser)
try await authManager.signIn()

// Use with Hub client
let client = HubClient(tokenProvider: .oauth(manager: authManager))

// Tokens are automatically refreshed when needed
let userInfo = try await client.whoami()
print("Signed in as: \(userInfo.name)")
```

The OAuth manager handles token storage in Keychain,
automatic refresh, and secure sign-out.
No more manual token management.

## Reliable Downloads

Downloading large models is now straightforward with proper progress tracking and resume support:

```swift
// Download with progress tracking
let progress = Progress(totalUnitCount: 0)

Task {
    for await _ in progress.publisher(for: \.fractionCompleted).values {
        print("Download: \(Int(progress.fractionCompleted * 100))%")
    }
}

let fileURL = try await client.downloadFile(
    at: "model.safetensors",
    from: "microsoft/phi-2",
    to: destinationURL,
    progress: progress
)
```

If a download is interrupted,
you can resume it:

```swift
// Resume from where you left off
let fileURL = try await client.resumeDownloadFile(
    resumeData: savedResumeData,
    to: destinationURL,
    progress: progress
)
```

For downloading entire model repositories,
`downloadSnapshot` handles everything:

```swift
let modelDir = try await client.downloadSnapshot(
    of: "mlx-community/Llama-3.2-1B-Instruct-4bit",
    to: cacheDirectory,
    matching: ["*.safetensors", "*.json"],  // Only download what you need
    progressHandler: { progress in
        print("Downloaded \(progress.completedUnitCount) of \(progress.totalUnitCount) files")
    }
)
```

The snapshot function tracks metadata for each file,
so subsequent calls only download files that have changed.

## Shared Cache with Python

Remember the second problem we mentioned?
_"No shared cache with the Python ecosystem."_
That's now solved.

swift-huggingface implements a Python-compatible cache structure
that allows seamless sharing between Swift and Python clients:

```
~/.cache/huggingface/hub/
├── models--deepseek-ai--DeepSeek-V3.2/
│   ├── blobs/
│   │   └── <etag>           # actual file content
│   ├── refs/
│   │   └── main             # contains commit hash
│   └── snapshots/
│       └── <commit_hash>/
│           └── config.json  # symlink → ../../blobs/<etag>
```

This means:

- **Download once, use everywhere.**
  If you've already downloaded a model with `huggingface-cli` or the Python library,
  swift-huggingface will find it automatically.
- **Content-addressed storage.**
  Files are stored by their ETag in the `blobs/` directory.
  If two revisions share the same file, it's only stored once.
- **Symlinks for efficiency.**
  Snapshot directories contain symlinks to blobs,
  minimizing disk usage while maintaining a clean file structure.

The cache location follows the same environment variable conventions as Python:

1. `HF_HUB_CACHE` environment variable
2. `HF_HOME` environment variable + `/hub`
3. `~/.cache/huggingface/hub` (default)

You can also use the cache directly:

```swift
let cache = HubCache.default

// Check if a file is already cached
if let cachedPath = cache.cachedFilePath(
    repo: "deepseek-ai/DeepSeek-V3.2",
    kind: .model,
    revision: "main",
    filename: "config.json"
) {
    let data = try Data(contentsOf: cachedPath)
    // Use cached file without any network request
}
```

To prevent race conditions when multiple processes access the same cache,
swift-huggingface uses file locking
([`flock(2)`](https://man7.org/linux/man-pages/man2/flock.2.html)).

## Before and After

Here's what downloading a model snapshot looked like with the old `HubApi`:

```swift
// Before: HubApi in swift-transformers
let hub = HubApi()
let repo = Hub.Repo(id: "mlx-community/Llama-3.2-1B-Instruct-4bit")

// No progress tracking, no resume, errors swallowed
let modelDir = try await hub.snapshot(
    from: repo,
    matching: ["*.safetensors", "*.json"]
) { progress in
    // Progress object exists but wasn't always accurate
    print(progress.fractionCompleted)
}
```

And here's the same operation with swift-huggingface:

```swift
// After: swift-huggingface
let client = HubClient.default

let modelDir = try await client.downloadSnapshot(
    of: "mlx-community/Llama-3.2-1B-Instruct-4bit",
    to: cacheDirectory,
    matching: ["*.safetensors", "*.json"],
    progressHandler: { progress in
        // Accurate progress per file
        print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
    }
)
```

The API is similar, but the implementation is completely different —
built on `URLSession` download tasks with proper
delegate handling, resume data support, and metadata tracking.

## Beyond Downloads

But wait, there's more!
swift-huggingface contains a complete Hub client:

```swift
// List trending models
let models = try await client.listModels(
    filter: "library:mlx",
    sort: "trending",
    limit: 10
)

// Get model details
let model = try await client.getModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
print("Downloads: \(model.downloads ?? 0)")
print("Likes: \(model.likes ?? 0)")

// Work with collections
let collections = try await client.listCollections(owner: "huggingface", sort: "trending")

// Manage discussions
let discussions = try await client.listDiscussions(kind: .model, "username/my-model")
```

And that's not all!
swift-huggingface has everything you need to interact with
[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index),
giving your app instant access to hundreds of machine learning models,
powered by world-class inference providers:

```swift
import HuggingFace

// Create a client (uses auto-detected credentials from environment)
let client = InferenceClient.default

// Generate images from a text prompt
let response = try await client.textToImage(
    model: "black-forest-labs/FLUX.1-schnell",
    prompt: "A serene Japanese garden with cherry blossoms",
    provider: .hfInference,
    width: 1024,
    height: 1024,
    numImages: 1,
    guidanceScale: 7.5,
    numInferenceSteps: 50,
    seed: 42
)

// Save the generated image
try response.image.write(to: URL(fileURLWithPath: "generated.png"))
```

Check the [README](https://github.com/huggingface/swift-huggingface) for a full list of everything that's supported.

## What's Next

We're actively working on two fronts:

**Integration with swift-transformers.**
We have a [pull request in progress](https://github.com/huggingface/swift-transformers) to replace `HubApi` with swift-huggingface.
This will bring reliable downloads to everyone using swift-transformers, mlx-swift-examples, and the broader ecosystem.

**Faster downloads with Xet.**
We're adding support for the [Xet storage backend](https://huggingface.co/docs/hub/storage-backends),
which enables chunk-based deduplication and significantly faster downloads for large models.
More on this soon.

## Try It Out

Add swift-huggingface to your project:

```swift
dependencies: [
    .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.3.0")
]
```

We'd love your feedback.
If you've been frustrated with model downloads in Swift, give this a try and
[let us know how it goes](https://github.com/huggingface/swift-huggingface/issues).
Your experience reports will help us prioritize what to improve next.

## Resources

- [swift-huggingface on GitHub](https://github.com/huggingface/swift-huggingface)
- [swift-transformers](https://github.com/huggingface/swift-transformers)
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples)
- [AnyLanguageModel](https://github.com/mattt/AnyLanguageModel)

---

*Thanks to the swift-transformers community for the feedback that shaped this project, and to everyone who filed issues and shared their experiences. This is for you.* ❤️