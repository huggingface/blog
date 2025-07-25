---
title: "Say hello to `hf`: a faster, friendlier Hugging Face CLI ✨"
thumbnail: /blog/assets/hf-cli-thumbnail.png
authors:
  - user: Wauplin
  - user: celinah
  - user: julien-c
---

We are glad to announce a long-awaited quality-of-life improvement: the Hugging Face CLI has been officially renamed from `huggingface-cli` to `hf`!

So... why this change?

Typing `huggingface-cli` constantly gets old fast. More importantly, the CLI’s command structure became messy as new features were added over time (upload, download, cache management, repo management, etc.). Renaming the CLI is a chance to reorganize commands into a clearer, more consistent format.

We decided not to reinvent the wheel and instead follow a well-known CLI pattern: **`hf <resource> <action>`**. This predictable grammar makes the Hugging Face CLI more ergonomic and discoverable, while also setting the stage for upcoming features.

## Getting started

To start playing with the new CLI, you’ll need to install the latest `huggingface_hub` version:

```
pip install -U huggingface_hub
```

and reload your terminal session. To test the install completed successfully, run `hf version`:

```
➜ hf version
huggingface_hub version: 0.34.0
```

Next, let’s explore the new syntax with `hf --help`:

```
➜ hf --help
usage: hf <command> [<args>]

positional arguments:
  {auth,cache,download,jobs,repo,repo-files,upload,upload-large-folder,env,version,lfs-enable-largefiles,lfs-multipart-upload}
                        hf command helpers
    auth                Manage authentication (login, logout, etc.).
    cache               Manage local cache directory.
    download            Download files from the Hub
    jobs                Run and manage Jobs on the Hub.
    repo                Manage repos on the Hub.
    repo-files          Manage files in a repo on the Hub.
    upload              Upload a file or a folder to the Hub. Recommended for single-commit uploads.
    upload-large-folder
                        Upload a large folder to the Hub. Recommended for resumable uploads.
    env                 Print information about the environment.
    version             Print information about the hf version.

options:
  -h, --help            show this help message and exit
```

As we can see, commands are grouped by "resource" (`hf auth`, `hf cache`, `hf repo`, etc.). We also surface `hf upload` and `hf download`at the root level since they’re expected to be the most-used commands.

To dive deeper into any command group, simply append `--help`:

```
➜ hf auth --help
usage: hf <command> [<args>] auth [-h] {login,logout,whoami,switch,list} ...

positional arguments:
  {login,logout,whoami,switch,list}
                        Authentication subcommands
    login               Log in using a token from huggingface.co/settings/tokens
    logout              Log out
    whoami              Find out which huggingface.co account you are logged in as.
    switch              Switch between access tokens
    list                List all stored access tokens

options:
  -h, --help            show this help message and exit
```

## 🔀 Migration

If you are used to `huggingface-cli`, most commands will look familiar. The biggest change affects authentication:


```bash
huggingface-cli login
# became
hf auth login
```

```bash
huggingface-cli whoami
# became
hf auth whoami
```

```bash
huggingface-cli logout
# became
hf auth logout
```

All `auth` commands have been grouped together with the existing `hf auth switch` (to switch between different local profiles) and `hf auth list` (to list local profiles).

The legacy `huggingface-cli` remains active and fully-functional. We’re keeping it around to ease the transition. If you use any command from the legacy CLI, you’ll see a warning that points you to the new CLI equivalent:

```bash
➜ huggingface-cli whoami
⚠️  Warning: 'huggingface-cli whoami' is deprecated. Use 'hf auth whoami' instead.
Wauplin
orgs:  huggingface,competitions,hf-internal-testing,templates,HF-test-lab,Gradio-Themes,autoevaluate,HuggingFaceM4,HuggingFaceH4,open-source-metrics,sd-concepts-library,hf-doc-build,hf-accelerate,HFSmolCluster,open-llm-leaderboard,pbdeeplinks,discord-community,llhf,sllhf,mt-metrics,DDUF,hf-inference,changelog,tiny-agents
```

## One more thing... 💥 `hf jobs`

We couldn’t resist shipping our first dedicated command: **hf jobs**.

Hugging Face Jobs is a new service that lets you run any script or Docker image on Hugging Face Infrastructure using the hardware flavor of your choice. Billing is "pay-as-you-go", meaning you pay only for the seconds you use. Here’s how to launch your first command:

```bash
# Run "nvidia-smi" on an A10G GPU
hf jobs run --flavor=a10g-small ubuntu nvidia-smi
```

The CLI is heavily inspired by Docker’s familiar commands:

```bash
➜ hf jobs --help
usage: hf <command> [<args>] jobs [-h] {inspect,logs,ps,run,cancel,uv} ...

positional arguments:
  {inspect,logs,ps,run,cancel,uv}
                        huggingface.co jobs related commands
    inspect             Display detailed information on one or more Jobs
    logs                Fetch the logs of a Job
    ps                  List Jobs
    run                 Run a Job
    cancel              Cancel a Job
    uv                  Run UV scripts (Python with inline dependencies) on HF infrastructure

options:
  -h, --help            show this help message and exit
```

Learn more about Jobs by [reading the guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/jobs).

> 💡 **Tip**  
>**Hugging Face Jobs** are available only to [Pro users](https://huggingface.co/pro) and [Team or Enterprise organizations](https://huggingface.co/enterprise). Upgrade your plan to get started!
