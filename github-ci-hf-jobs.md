---
title: "Migrating Your GitHub CI to Hugging Face Jobs"
thumbnail: /blog/assets/github-ci-hf-jobs/thumbnail.gif
authors:
- user: abidlabs
---

# Migrating Your GitHub CI to Hugging Face Jobs

If you have a GitHub repository and you have GitHub Actions enabled, you probably use GitHub-hosted runners for CI. That is the default for many projects because it is simple: add a workflow, write `runs-on: ubuntu-latest`, and GitHub gives you a machine.

That default is convenient, but it also has limits. GitHub Actions can be slow or down for maintenance, the hosted machines are generic, and GPU access is not something most open-source projects can just turn on. For [Trackio](https://github.com/gradio-app/trackio), those limits started to matter. We wanted both reliable CPU CI for basic unit tests and frontend checks, but also GPU CI for tests that need to run on actual CUDA hardware.

So we tried an experiment: keep GitHub Actions as the CI control plane, but run selected jobs on [Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs-overview).


The result: Trackio's CI now runs on Hugging Face Jobs and streams back real-time logs, **cutting our CI time for CPU jobs by about 30% and enabling a whole new test suite that runs on GPU machines**!

In this article, we explain step-by-step how to recreate the same setup for your GitHub repo. If you are using an agent, you can point it to this article, since we provide CLI instructions alongside browser-based instructions for us humans.

Let's start with a quick intro to Hugging Face Jobs!

## What is Hugging Face Jobs?

[Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs-overview) lets you run commands or scripts on Hugging Face's serverless infrastructure with almost any hardware flavor. A Job is essentially:

- a command to run
- a Docker image, from Docker Hub or a Hugging Face Space
- a hardware flavor, such as CPU or `t4-small` or `h200` GPU
- optional environment variables and secrets

That makes Jobs a natural fit for CI. CI jobs are already command-driven, already run in clean environments, and often benefit from choosing exactly the right hardware. For ML libraries, the GPU case is especially compelling: you can run a test suite on real GPU hardware without maintaining your own always-on runner.

The key step is connecting GitHub Actions to HF Jobs, which we describe below.

## The architecture

For this setup, we created [`huggingface/jobs-actions`](https://github.com/huggingface/jobs-actions), a small bridge that turns a GitHub Actions job into an ephemeral self-hosted runner running inside an HF Job.

The complete flow looks like this:

1. A pull request triggers a GitHub Actions workflow.
1. GitHub queues any job whose `runs-on` label is not available, for example `hf-jobs-cpu-upgrade` or `hf-jobs-t4-small`, and sends a signed `workflow_job.queued` webhook to the dispatcher through the GitHub App.
1. The dispatcher Space verifies the webhook, checks for an `hf-jobs-*` label, mints a short-lived GitHub runner registration token, and starts an HF Job on the matching hardware.
1. The HF Job boots an ephemeral GitHub Actions runner and registers it with the repo using that one-shot token.
1. GitHub assigns the pending workflow job to that runner; the runner executes the CI job, reports status back to GitHub, and exits.

From GitHub's point of view, this is just a self-hosted runner. From Hugging Face's point of view, it is just a Job that launches a container to run the workflow steps from the repo’s GitHub Actions.

## Step 1: Duplicate the dispatcher Space

The first thing you need is the dispatcher. This is a small Docker Space that receives GitHub `workflow_job` webhook events and launches HF Jobs in response.

Create this first because the GitHub App needs a webhook URL, and that URL comes from the Space. This Space should be under your own namespace or under a Hugging Face org that you have write access to.

#### Web setup

Go to [`huggingface/jobs-actions-dispatcher`](https://huggingface.co/spaces/huggingface/jobs-actions-dispatcher) and click **Duplicate this Space**.

<img width="996" alt="image" src="https://github.com/user-attachments/assets/c8b450c3-b801-43dc-97ff-954d9bbaf975" />


Use:

```text
Owner: your HF user or org
Name: jobs-actions-dispatcher
Hardware: cpu-upgrade
```

Use `cpu-upgrade` for real CI so the dispatcher stays available for GitHub webhooks. `cpu-basic` is fine for testing and will probably work, but it can sleep after inactivity; if GitHub's webhook arrives while it is waking up, the workflow may stay queued until you rerun it or redeliver the webhook.

After it builds, open the duplicated Space. You will see a section that says "Required Space secrets," which you can ignore for now. The landing page should display the GitHub App webhook URL you need in the next step. It will look like this:

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
  --flavor cpu-upgrade \
  --exist-ok
```

Then set:

```bash
export DISPATCHER_URL="https://${HF_NAMESPACE}-jobs-actions-dispatcher.hf.space"
```

## Step 2: Create and install the GitHub App

Next, create and install the GitHub App from the dispatcher Space itself. This App needs permission to listen for queued workflow jobs and create ephemeral self-hosted runner registration tokens.

### Web setup

Open your duplicated dispatcher Space:

```text
https://YOUR-HF-NAMESPACE-jobs-actions-dispatcher.hf.space
```

In the setup form, enter the GitHub repo whose CI should run on HF Jobs:

```text
YOUR-GITHUB-ORG/YOUR-REPO
```

Then click the button to create the GitHub App. GitHub will ask you to choose a name for the App; the name can be anything, as long as it is available in your GitHub account or org. After you submit, the final screen tells you exactly how to upload the App credentials to the dispatcher Space with the `hf` CLI.

**Important note**: you will need to provide an [Hugging Face token](https://huggingface.co/settings/tokens) that has permissions to launch Jobs, corresponding to your personal account or an org under which Jobs should be charged. This token should be saved as the `HF_TOKEN` secret in your dispatcher Space.

Finally, you will install the App on the same GitHub repo you entered in the Space. In the Trackio setup, we installed it on `gradio-app/trackio`.

### Agent-assisted setup

The GitHub App manifest flow is still browser-based, but an agent can follow the same Space-driven path:

```bash
export HF_NAMESPACE=your-hf-user-or-org
export GITHUB_REPO=YOUR-GITHUB-ORG/YOUR-REPO
open "https://${HF_NAMESPACE}-jobs-actions-dispatcher.hf.space"
```

Paste `$GITHUB_REPO` into the Space, click the GitHub App creation button, choose any available App name, and follow the generated GitHub instructions.

After the App exists, install it on your repo from the App settings page. For a GitHub org, the installation settings are under:

```text
https://github.com/organizations/YOUR-GITHUB-ORG/settings/installations
```

## Step 3: Final dispatcher settings

At this point, the dispatcher Space should be configured. The GitHub App setup flow generated the commands that upload the App credentials, webhook secret, and Hugging Face token to the Space.

<img width="1317" alt="image" src="https://github.com/user-attachments/assets/0fc8ac73-f93a-419b-bd80-70da2756f50c" />


By default, HF Jobs are launched under the same namespace as the dispatcher Space. Optionally, set `HF_NAMESPACE` as a Space variable if you want to bill jobs to a different Hugging Face user or org:

```bash
export SPACE_ID=YOUR-HF-NAMESPACE/jobs-actions-dispatcher
hf spaces variables add "$SPACE_ID" -e HF_NAMESPACE=your-billing-namespace
hf spaces restart "$SPACE_ID"
```

The token you set in Step 2 should correspond to this namespace.

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

For any GitHub Action you'd like to run on on HF Jobs, this 1-line change is all you need!

## Step 5: Test it out

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

You should be able see logs just like a regular GitHub Action, e.g. like in this [Trackio PR #565](https://github.com/gradio-app/trackio/pull/565).


And that's it!

*Note on choosing the right Docker image*

Our first CPU setup used `ubuntu:22.04` and installed missing system packages during every run. That worked, but it was slower than it needed to be. GitHub's `ubuntu-latest` image includes a lot of developer tooling by default; a bare Ubuntu image does not.

For Trackio, the UI tests need Playwright browsers, Node, ffmpeg, sqlite, git, and normal Linux build dependencies. We tested the Microsoft Playwright image:

```text
mcr.microsoft.com/playwright:v1.60.0-jammy
```

For GPU jobs, we use:

```text
nvidia/cuda:12.4.0-runtime-ubuntu22.04
```

## Results

Here are the numbers from the Trackio PR experiment:

| Runner setup | Runtime | Compared to GitHub average |
| --- | ---: | ---: |
| GitHub `ubuntu-latest` baseline | `1m40s`  | baseline |
| HF Jobs CPU, Playwright image | `1m10s` | `-30s`, about `30%` faster |
| HF Jobs GPU, `t4-small` label | `45s` | no GitHub-hosted GPU baseline |


The biggest win was GPU CI. The Trackio GPU check ran on HF Jobs and passed in `45s`, costing less than a cent at the `t4-small` rate for that duration.

The CPU result was also encouraging. With the right image, the Linux test job was faster than the GitHub-hosted baseline. That suggests HF Jobs can be a practical CI backend, especially for ML projects that need custom images or accelerators.

Logs were another pleasant surprise. GitHub Actions logs are useful, but the web UI can be heavy for large logs. HF Jobs logs are easy to fetch from the CLI:

```bash
hf jobs logs <job_id> > logs.txt
```

That makes them easy to inspect with local tools or coding agents. In our bridge, we also mirrored the GitHub Actions job log into the HF Job log, so either system had enough information to debug a run.

Hopefully, this convinces you to give HF Jobs a shot for running your GitHub Actions!
