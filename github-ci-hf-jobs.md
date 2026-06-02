---
title: "Migrating Your GitHub CI to Hugging Face Jobs"
thumbnail: /blog/assets/github-ci-hf-jobs/thumbnail.gif
authors:
- user: abidlabs
---

# Migrating Your GitHub CI to Hugging Face Jobs

If you have a GitHub repository and you have GitHub Actions enabled, you probably use GitHub-hosted runners for CI. That is the default for many projects because it is simple: add a workflow, write `runs-on: ubuntu-latest`, and GitHub gives you a machine.

That default is convenient, but it also has limits. GitHub Actions can be slow, the hosted machines are generic, and GPU access is not something most open-source projects can just turn on. For [Trackio](https://github.com/gradio-app/trackio), those limits started to matter. We wanted both normal CPU CI for basic unit tests and frontend checks, but also GPU CI for tests that need to run on actual CUDA hardware.

So we tried an experiment: keep GitHub Actions as the CI control plane, but run selected jobs on [Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs-overview).

The result: Trackio's CI now runs on Hugging Face Jobs, **cutting our CI time for CPU jobs by 40% and enabling a whole new test suite that runs on GPU machines**!

In this article, we explain step-by-step, how to recreate the same setup for your GitHub repo. 

But first, a quick intro to Hugging Face Jobs!

## What is Hugging Face Jobs?

[Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs-overview) lets you run commands or scripts on Hugging Face's serverless infrastructure with almost any hardware flavor. A Job is essentially:

- a command to run
- a Docker image, from Docker Hub or a Hugging Face Space
- a hardware flavor, such as CPU or `t4-small` or `h200` GPU
- optional environment variables and secrets

That makes Jobs a natural fit for CI. CI jobs are already command-driven, already run in clean environments, and often benefit from choosing exactly the right hardware. For ML libraries, the GPU case is especially compelling: you can run a test suite on real GPU hardware without maintaining your own always-on runner.

The key step is connecting GitHub Actions to HF Jobs.

## The architecture

For this setup, we created [`huggingface/jobs-actions`](https://github.com/huggingface/jobs-actions), a small bridge that turns a GitHub Actions job into an ephemeral self-hosted runner running inside an HF Job.

The complete flow looks like this:

1. A pull request triggers a GitHub Actions workflow.
2. GitHub sees a job with a custom label, for example `hf-jobs-cpu-upgrade` or `hf-jobs-t4-small`.
3. A GitHub App receives the workflow event and asks Hugging Face Jobs to start a new Job.
4. The HF Job downloads and registers a GitHub Actions self-hosted runner.
5. GitHub assigns the pending workflow job to that runner.
6. The runner executes the CI job, reports status back to GitHub, and exits.

From GitHub's point of view, this is just a self-hosted runner. From Hugging Face's point of view, it is just a short-lived Job.

## Step 1: Duplicate the dispatcher Space

The first thing you need is the dispatcher. This is a small Docker Space that receives GitHub `workflow_job` webhook events and launches HF Jobs in response.

Create this first because the GitHub App needs a webhook URL, and that URL comes from the Space. This Space should be under your own namespace or under an org that has billing credits enabled, since Jobs will be [charged to this account](https://huggingface.co/docs/hub/en/jobs-pricing#pricing).

#### Web setup

Go to [`huggingface/jobs-actions-dispatcher`](https://huggingface.co/spaces/huggingface/jobs-actions-dispatcher) and click **Duplicate this Space**.

<img width="996" alt="image" src="https://github.com/user-attachments/assets/c8b450c3-b801-43dc-97ff-954d9bbaf975" />


Use:

```text
Owner: your HF user or org
Name: jobs-actions-dispatcher
Hardware: cpu-basic
```

If you see warnings saying that "Your duplicated Space may not work if you switch to a different hardware than the suggested one", that's okay! You also don't need to fill out any secrets at this point.

After it builds, open the duplicated Space. You may see some configuration errors, but that's also okay. The landing page should display the GitHub App webhook URL you need in the next step. It will look like this:

```text
https://YOUR-HF-NAMESPACE-jobs-actions-dispatcher.hf.space/webhook
```

#### CLI setup

If you'd prefer to set up dispatcher Space with  an agent or use a CLI workflow:

```bash
export HF_NAMESPACE=your-hf-user-or-org
export SPACE_ID="$HF_NAMESPACE/jobs-actions-dispatcher"

hf repo duplicate huggingface/jobs-actions-dispatcher "$SPACE_ID" \
  --type space \
  --flavor cpu-basic \
  --exist-ok
```

Then set:

```bash
export DISPATCHER_URL="https://${HF_NAMESPACE}-jobs-actions-dispatcher.hf.space"
```

## Step 2: Create and install the GitHub App

Next, create the GitHub App from the manifest included in [`huggingface/jobs-actions`](https://github.com/huggingface/jobs-actions). This App needs permission to listen for queued workflow jobs and create ephemeral self-hosted runner registration tokens.

### Web setup

Open:

```text
setup/create-app.html
```

from a local clone of [`huggingface/jobs-actions`](https://github.com/huggingface/jobs-actions). Fill in:

```text
GitHub user or org: YOUR-GITHUB-ORG
HF Space dispatcher URL: https://YOUR-HF-NAMESPACE-jobs-actions-dispatcher.hf.space
```

Submit the form. GitHub will create the App and take you to the App settings page. Save:

```text
App ID
Webhook secret
Private key .pem
```

Then install the App on the GitHub repo whose CI should run on HF Jobs. In the Trackio setup, we installed it on `gradio-app/trackio`.

### Agent-assisted setup

The GitHub App manifest flow is intentionally browser-based, but an agent can still prepare almost everything:

```bash
git clone https://github.com/huggingface/jobs-actions
cd jobs-actions
export GH_ORG=your-github-org
open setup/create-app.html
```

Fill in the GitHub org and dispatcher URL, submit, and save the generated App credentials.

After the App exists, install it on your repo from the App settings page. For a GitHub org, the installation settings are under:

```text
https://github.com/organizations/YOUR-GITHUB-ORG/settings/installations
```

## Step 3: Configure dispatcher settings

The dispatcher is a small web service. GitHub sends webhook events to it, and it launches HF Jobs with the right image, hardware flavor, labels, and one-shot runner token. The Space only needs three real secrets:

```text
GH_APP_PRIVATE_KEY
GH_WEBHOOK_SECRET
HF_TOKEN
```

It also needs one normal environment variable:

```text
GH_APP_ID
```

By default, HF Jobs are launched under the owner namespace of the dispatcher Space. Set `HF_NAMESPACE` as an optional Space variable only if you want to bill jobs to a different HF user or org.

You can add the configuration through the Space settings UI: **Settings → Variables and secrets**. Put `GH_APP_PRIVATE_KEY`, `GH_WEBHOOK_SECRET`, and `HF_TOKEN` under **Secrets**. Put `GH_APP_ID` under **Variables**.

Or from the CLI:

```bash
export GH_APP_ID=123456
export GH_WEBHOOK_SECRET=your-webhook-secret
export GH_APP_PRIVATE_KEY_PATH=/path/to/private-key.pem
export HF_TOKEN=hf_xxx
export HF_NAMESPACE=your-hf-user-or-org
export SPACE_ID="$HF_NAMESPACE/jobs-actions-dispatcher"

python - <<'PY' > /tmp/jobs-actions-secrets.env
import os
from pathlib import Path

print(f"GH_WEBHOOK_SECRET={os.environ['GH_WEBHOOK_SECRET']}")
print(f"HF_TOKEN={os.environ['HF_TOKEN']}")
private_key = Path(os.environ["GH_APP_PRIVATE_KEY_PATH"]).read_text()
print("GH_APP_PRIVATE_KEY=" + private_key.replace("\n", "\\n"))
PY

hf spaces secrets add "$SPACE_ID" --secrets-file /tmp/jobs-actions-secrets.env
hf spaces variables add "$SPACE_ID" -e GH_APP_ID="$GH_APP_ID"
hf spaces restart "$SPACE_ID"
```

At this point, GitHub can notify the dispatcher whenever a workflow job is queued.

## Step 4: Change `runs-on`

The actual workflow change is small. Instead of:

```yaml
runs-on: ubuntu-latest
```

use one of the labels handled by the dispatcher:

```yaml
runs-on: hf-jobs-cpu-upgrade
```

For GPU tests, use a GPU label:

```yaml
runs-on: hf-jobs-t4-small
```

To add a minimal smoke-test workflow from the CLI:

```bash
mkdir -p .github/workflows
cat > .github/workflows/hf-jobs-test.yml <<'EOF'
name: HF Jobs Test

on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: hf-jobs-cpu-upgrade
    steps:
      - uses: actions/checkout@v4
      - run: echo "Hello from Hugging Face Jobs"
EOF

git add .github/workflows/hf-jobs-test.yml
git commit -m "Run CI on Hugging Face Jobs"
git push
```

To verify from the CLI:

```bash
gh run list --repo YOUR-GITHUB-ORG/YOUR-REPO --limit 5
hf jobs ps --namespace "$HF_NAMESPACE"
hf spaces logs "$SPACE_ID"
```

Here is a simplified version of the CPU test job we used in [Trackio PR #565](https://github.com/gradio-app/trackio/pull/565):

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [hf-jobs-cpu-upgrade, windows-latest]
    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        id: setup-python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install Python dependencies
        run: |
          uv pip install --system -e '.[dev,tensorboard,spaces]' --prerelease=allow --python "${{ steps.setup-python.outputs.python-path }}"
          uv pip install --system pytest --prerelease=allow --python "${{ steps.setup-python.outputs.python-path }}"

      - name: Build frontend
        if: matrix.os == 'hf-jobs-cpu-upgrade'
        run: |
          cd trackio/frontend
          npm ci
          npm run build

      - name: Run backend unit tests
        run: |
          "${{ steps.setup-python.outputs.python-path }}" -m pytest --deselect=tests/ui --deselect=tests/e2e-spaces --deselect=tests/unit/test_gpu_hardware.py

      - name: Run ui/ux interaction tests
        if: matrix.os == 'hf-jobs-cpu-upgrade'
        env:
          PLAYWRIGHT_BROWSERS_PATH: /ms-playwright
        run: |
          "${{ steps.setup-python.outputs.python-path }}" -m pytest tests/ui
```

And here is the GPU workflow pattern:

```yaml
jobs:
  gpu-test:
    runs-on: hf-jobs-t4-small

    steps:
      - uses: actions/checkout@v3

      - name: Show GPU
        run: nvidia-smi

      - name: Run GPU hardware tests
        run: |
          python -m pytest tests/unit/test_gpu_hardware.py -v
```

That separation matters. We made `tests/unit/test_gpu_hardware.py` less guarded so that it fails if CUDA or NVML is missing. Then we made sure those tests only run on the HF GPU runner. If the GPU job passes, we know the test really ran on GPU hardware.

## Choosing the right Docker image

Our first CPU setup used `ubuntu:22.04` and installed missing system packages during every run. That worked, but it was slower than it needed to be. GitHub's `ubuntu-latest` image includes a lot of developer tooling by default; a bare Ubuntu image does not.

For Trackio, the UI tests need Playwright browsers, Node, ffmpeg, sqlite, git, and normal Linux build dependencies. We tested the Microsoft Playwright image:

```text
mcr.microsoft.com/playwright:v1.60.0-jammy
```

That image already has the browser stack we needed. The one extra detail was setting:

```text
PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
```

Without that, Python Playwright looked in the default cache directory instead of the browser directory baked into the image.

The GPU job used:

```text
nvidia/cuda:12.4.0-runtime-ubuntu22.04
```

## Results

Here are the numbers from the Trackio PR experiment:

| Runner setup | Runtime | Compared to GitHub average |
| --- | ---: | ---: |
| GitHub `ubuntu-latest` baseline | `1m40s` average | baseline |
| HF Jobs CPU, `ubuntu:22.04` plus installs | `1m57s` average | `+17s`, about `17%` slower |
| HF Jobs CPU, Playwright image | `1m10s` | `-30s`, about `30%` faster |
| HF Jobs CPU, Playwright image, trigger-to-finish wall time | `2m45s` | `+65s`, about `65%` slower wall-clock |
| HF Jobs GPU, `t4-small` label | `45s` | no GitHub-hosted GPU baseline |

There are two honest conclusions from these numbers.

First, once the HF runner was attached, the CI job itself was fast. Moving from a bare Ubuntu image to the Playwright image reduced the HF CPU job from about `1m57s` to `1m10s`, roughly a `40%` speedup over the first HF setup and about `30%` faster than the GitHub-hosted Ubuntu baseline.

Second, end-to-end wall-clock time still had extra latency. The workflow run was created at `19:25:52`, the HF Job was created at `19:25:54`, but the runner did not start listening for GitHub jobs until `19:27:23`. GitHub assigned the job about five seconds later. That means the remaining bottleneck was not GitHub assigning work; it was the time to schedule the HF Job, start the container, install bootstrap dependencies, download the GitHub Actions runner, register it, and begin listening.

This is fixable. A more optimized runner image could preinstall the GitHub runner and bootstrap dependencies, reducing the attach time.

## What worked well

The biggest win was GPU CI. The Trackio GPU check ran on HF Jobs and passed in `45s`, costing less than a cent at the `t4-small` rate for that duration. More importantly, the tests now fail if the machine does not actually have a GPU, so the check is meaningful.

The CPU result was also encouraging. With the right image, the Linux test job was faster than the GitHub-hosted baseline. That suggests HF Jobs can be a practical CI backend, especially for ML projects that need custom images or accelerators.

Logs were another pleasant surprise. GitHub Actions logs are useful, but the web UI can be heavy for large logs. HF Jobs logs are easy to fetch from the CLI:

```bash
hf jobs logs <job_id> > logs.txt
```

That makes them easy to inspect with local tools or coding agents. In our bridge, we also mirrored the GitHub Actions job log into the HF Job log, so either system had enough information to debug a run.

## Tradeoffs

This setup is still more work than using GitHub-hosted runners. We had to configure a GitHub App, a dispatcher Space, HF tokens, webhook secrets, runner labels, custom images, and branch protection checks. That is a lot more moving pieces than `runs-on: ubuntu-latest`.

There are also real operational considerations:

- Startup latency matters. The job runtime was fast, but the runner took around `1m29s` to become available.
- Caching is different. `actions/cache` can still work because this is a GitHub Actions runner, but the local HF Job filesystem is ephemeral. Docker layer caches, apt caches, and local dependency caches do not persist unless you explicitly use a remote cache.
- Security needs care. Self-hosted runners for public repositories can be risky if untrusted pull requests are allowed to run with secrets or privileged access.
- Docker-in-Docker workflows may not work out of the box, because the runner itself is already inside a container.
- HF webhook Jobs currently fit Hugging Face repo events best. For GitHub pull request and workflow-job events, we still needed the GitHub App plus dispatcher bridge.

None of these are blockers, but they are the difference between a good prototype and a polished product experience.

## What would make this better?

The ideal version would feel like this:

```yaml
runs-on: hf-jobs/t4-small
image: mcr.microsoft.com/playwright:v1.60.0-jammy
```

To get there, a few things would help:

- A documented "GitHub Actions on HF Jobs" recipe.
- A maintained runner image with GitHub Actions runner dependencies preinstalled.
- First-class log mirroring between GitHub and HF Jobs.
- Better cancellation handling, so canceling a GitHub workflow cancels the HF Job immediately.
- A simpler way to connect GitHub workflow events directly to HF Jobs without running a separate dispatcher Space.
- Clear examples for CPU, GPU, custom Docker images, caching, secrets, and public repository security.

Trackio can be a useful example because it has the common pieces: Python tests, frontend tests, Playwright, branch protection, and GPU-specific tests.

## That's it

The final workflow is simple for users:

1. Duplicate the dispatcher Space.
2. Create and install the `jobs-actions` GitHub App.
3. Add the GitHub App and HF secrets to the dispatcher Space.
4. Change `runs-on` from `ubuntu-latest` to an HF Jobs label.
5. Pick the right Docker image for your test suite.

For Trackio, that was enough to run CPU and GPU CI on HF Jobs while still reporting normal GitHub pull request checks.

The experiment convinced me that HF Jobs is a strong fit for ML-oriented CI, especially when GPU tests matter. The main thing to improve is the setup experience: fewer moving parts, faster runner attachment, and a documented path from "I have a GitHub workflow" to "this job runs on Hugging Face hardware."
