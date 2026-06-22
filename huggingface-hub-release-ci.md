---
title: "Shipping `huggingface_hub` every week with AI, open tools, and a human in the loop"
thumbnail: /blog/assets/huggingface-hub-release-ci/thumbnail.png
authors:
- user: Wauplin
- user: celinah
---

# Shipping `huggingface_hub` every week with AI, open tools, and a human in the loop

`huggingface_hub` is the Python client at the base of the Hugging Face ecosystem. `transformers`, `datasets`, `diffusers`, `sentence-transformers` and dozens of other libraries depend on it to talk to the Hub, so every week we *don't* ship is a week of fixes and features stuck in `main`.

For a long time we released every 4 to 6 weeks. We now release **every week**, from a single GitHub Actions workflow. The interesting part isn't the cadence. It's *how* we got there: we built the whole thing out of open-source tools and open-weights models, and we kept a human firmly in the loop at the one place where judgment matters. Nothing in this post requires a vendor contract, a closed model, or infrastructure you can't run yourself. That was a deliberate design goal, because we wanted a workflow other maintainers could pick up and adapt. So can you, by the end of this post.

## Where we started

The old process was **half automated, half manual**.

Already in CI:

- Publishing to PyPI once a tag was pushed.
- Opening test branches in downstream libraries with the release candidate pinned.

Still manual, every single time:

- Creating the release branch, bumping the version in `__init__.py`, committing, tagging, pushing.
- Watching the downstream CI runs and triaging failures.
- Reading through every PR merged since the last release and writing release notes by hand: grouped by theme, with context, in a voice that didn't read like a `git log` dump.
- Cutting the stable release after the RC period.
- Drafting an internal Slack announcement and social posts.
- Opening the post-release PR to bump `main` to the next `dev0`.

Writing good notes for a minor version was the heavy part. Thirty PRs, a handful of themes, some user-facing and some internal. Not technically hard, but a few hours of focused attention. Add the announcements on top and a minor release was easily a half-day of work spread over several days. Predictably, the cadence drifted.

## Two kinds of work

So we decided to streamline the whole thing. Looking at that list, the work split cleanly in two.

Some steps are purely mechanical and automate in a few lines of shell: bumping the version, committing, tagging, pushing, opening downstream test branches, opening the post-release PR. Nobody needs to think about those. They just have to happen in the right order, every time, which is precisely what a CI workflow is good at.

The rest is different. Writing release notes, deciding what to highlight, phrasing an announcement for a human audience: that's *brain work*. It's the kind of judgment that kept the release manual for years, because it genuinely needed a person. This is exactly where AI earns its place, turning a blank page into a solid first draft in seconds. It's also exactly where we have to be careful, because a draft that looks confident and is subtly wrong is worse than no draft at all.

That split shaped everything that follows.

## The design principle: open parts, reappropriable by anyone

When we decided to fix this, we set one constraint up front: **every moving part had to be something any maintainer could run themselves.** No closed model behind an API we couldn't swap, no proprietary release platform, no secret sauce.

Here's the entire stack:

| Part                                                                                              | What it does                                    |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **GitHub Actions**                                                                                | Orchestrates the whole release                  |
| **[OpenCode](https://opencode.ai/)**                                                              | Agent runtime that drives the model             |
| **An open-weights model** (currently [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) from Z.ai) | Drafts the release notes and Slack announcement |
| **[HF Inference Providers](https://huggingface.co/docs/inference-providers/index)**               | Serves the model                                |
| **[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)**                          | Publishes the package                           |

The second principle, just as important: **the model drafts, a human decides.** Language models are genuinely good at turning thirty terse PR titles into readable, grouped release notes. They are not good at being trusted blindly. So the workflow is built as *semi-supervised*: the model does the tedious first pass, deterministic code checks its work, and a human reviews and edits at exactly one well-defined checkpoint before anything ships. More on that below. It's the heart of the whole thing.

## A tour of the pipeline

The full workflow is a single file, [`.github/workflows/release.yml`](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml), triggered by hand from the Actions UI. It takes exactly one input:

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        options:
          - minor-prerelease   # cut an RC from main
          - minor-release      # promote the RC to final
          - patch-release      # bugfix on an existing release branch
```

From there, jobs fan out (roughly in order):

- **Prepare.** Compute the next version, create or reuse the release branch, bump `__version__`, commit, tag, push.
- **Publish to PyPI.** Build and upload `huggingface_hub`. In parallel, build and upload the `hf` CLI as its own PyPI package.
- **Release notes.** Diff the commit range since the last tag, pull PR metadata from the GitHub API, and have the model draft a structured changelog ([here's a recent one](https://github.com/huggingface/huggingface_hub/releases/tag/v1.20.0)). Saved as a *draft* GitHub release.
- **Downstream test branches.** For RCs, open a branch in `transformers`, `datasets`, `diffusers`, `sentence-transformers` with the RC pinned, so their CI tells us fast if we broke something.
- **Slack announcement.** Read the notes and produce an internal announcement in our team voice.
- **Archive notes.** Upload both the raw AI draft and the human-edited version to a Hugging Face Bucket, side by side.
- **Post-release bump.** After a stable release, open a PR on `main` bumping to the next `dev0`.
- **Comment on shipped PRs.** Leave a "this shipped in vX.Y.Z" comment on every PR in the release.
- **Sync CLI docs.** Push the latest `hf` CLI skill docs to our `skills` repo.
- **Report to Slack.** Every step posts its status as a thread reply; a final job updates the root message with ✅ or ❌.

From a human's chair, triggering a release is four clicks. The remaining manual steps are reviewing and publishing the draft release notes, and reviewing and posting an internal Slack message. These two steps are not an oversight. They are the design.

## Trust, but verify: the semi-supervised core

Here's the failure mode everyone worries about with AI-generated release notes: the model quietly drops a PR, or invents one that isn't in this release. A changelog that's *almost* right is worse than no changelog, because nobody re-checks it.

So we don't trust the model to be complete. We *verify* it, deterministically. Before the model runs, plain Python walks the commit range and writes down the ground truth: every PR number that actually belongs in this release.

```python
# Deterministic: extract PR numbers from squash-merge commits in the range.
PR_NUMBER_PATTERN = re.compile(r"\(#(\d+)\)$")

pr_numbers = [
    int(m.group(1))
    for commit in commits_since_last_tag
    if (m := PR_NUMBER_PATTERN.search(commit.title))
]
save_manifest(pr_numbers)  # the source of truth
```

The model drafts the notes. Then we check its output against that manifest, not with another model, just with set arithmetic:

```python
expected = set(load_manifest())          # what should be there
found    = extract_pr_refs(notes_md)     # what the model wrote (#1234 -> 1234)

missing = expected - found               # silently dropped
extra   = found - expected               # belongs to a different release
```

If anything is missing or extra, we don't fail and we don't ship a wrong file. We hand the discrepancy *back to the agent* and ask it to fix exactly those PRs, looping up to three times:

```python
for _ in range(MAX_ITERATIONS):
    missing, extra = validate(notes)
    if not missing and not extra:
        break  # matches the manifest exactly
    run_agent_fix(missing_prs=missing, extra_prs=extra)
```

This is the pattern that makes the whole thing trustworthy: **a non-deterministic model wrapped in deterministic guardrails.** The model is great at writing prose; it's unreliable at being exhaustive. So we let it write, and let code enforce the one property that actually matters: every PR accounted for, exactly once.

And when the model genuinely can't do its job (auth failure, empty output, or an unknown model name, since OpenCode exits `0` even on a typo'd model), the job fails *loudly* instead of publishing an empty release:

```python
if not output_file.exists() or output_file.stat().st_size == 0:
    sys.exit("OpenCode produced no notes; refusing to publish an empty release")
```

## Grounding the model so it doesn't make things up

Completeness is one half. Accuracy is the other. A model summarizing a PR from its title alone will cheerfully invent a code example that doesn't match the real API.

To prevent that, when we fetch PR metadata we also pull the **actual documentation diffs** from each PR: the unified diff of any `.md` file under `docs/` that the PR touched.

```python
def fetch_doc_diffs(pr):
    return [
        {"filename": f.filename, "status": f.status, "patch": f.patch}
        for f in pr.get_files()
        if f.filename.startswith("docs/") and f.filename.endswith(".md") and f.patch
    ]
```

That diff goes into the model's context. So when it writes "here's the new CLI command," it's quoting the example the PR author actually wrote in the docs, not guessing. Same idea, applied everywhere: give the model real source material and a narrow job, and it stays honest.

The prompts themselves live as Skills: small Markdown files (`SKILL.md` plus reference templates) checked into the repo. The release-notes skill spells out how to pick highlights, how to structure sections, when to add a doc link (and never to fabricate one). It reads like documentation for a careful junior teammate, which is exactly the right mental model.

## The human checkpoint

After the RC is published, the draft GitHub release sits there with the AI's first pass in it. This is where the human comes in, and the workflow is deliberately shaped around this pause:

1. A reviewer reads the draft, edits for tone and emphasis, fixes anything the model over- or under-weighted.
2. Only then do they trigger the `minor-release` run, which promotes the RC to final.

The reviewer's time goes into *polishing*, never into staring at a blank page. That's the whole productivity win: the half-day of writing becomes fifteen minutes of editing.

We also keep a paper trail to improve over time. We archive **two files** side by side to a Hugging Face Bucket: the raw AI draft, uploaded at RC time *before* anyone touches it, and the human-edited version, uploaded when the final release is cut.

```bash
# at RC time: straight from the model, untouched
hf cp release_notes_raw.txt    "hf://buckets/huggingface/releases/huggingface_hub/${V}/release_notes_raw.txt"

# at release time: after the human review
hf cp release_notes_edited.txt "hf://buckets/huggingface/releases/huggingface_hub/${V}/release_notes_edited.txt"
```

Collecting both every week gives us a growing dataset of "what the model wrote" versus "what we wished it wrote", which is the exact signal we use to tighten the prompt.

## Open and secure plumbing

Two details that matter if you care about supply-chain hygiene, and both come for free from open standards.

**No PyPI token, anywhere.** Publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/): PyPI verifies a short-lived OIDC token minted by GitHub for this exact workflow, and issues [PEP 740](https://peps.python.org/pep-0740/) attestations / Sigstore provenance for every artifact. There's no long-lived secret to leak or rotate.

```yaml
permissions:
  id-token: write       # mint the OIDC token for PyPI
  attestations: write   # generate Sigstore provenance
# ...
- uses: pypa/gh-action-pypi-publish@v1.14.0
  with:
    attestations: true  # no password, no API token, just OIDC
```

**The agent runtime is pinned and verified.** We don't `curl | bash` the latest OpenCode and hope. We pin a version and check its SHA256 before running it:

```bash
curl -fsSL https://opencode.ai/install | bash -s -- --version "${OPENCODE_VERSION}"
echo "${OPENCODE_SHA256}  $(which opencode)" | sha256sum -c -
```

Open tooling doesn't mean careless tooling.

## The boring-on-purpose parts

Plenty of the workflow is just mechanical glue, and that's the point, because it's the glue that used to eat the afternoon:

- **Slack threading.** A root message is posted when the release starts (tagging whoever triggered it); every job replies in-thread with its own ✅/❌, and a final job flips the root message and adds a reaction. The whole release is one readable thread.
- **Two AI jobs.** The release notes and the Slack announcement are separate jobs, each a short Python script that shells out to OpenCode with a focused skill. Small, single-purpose prompts beat one giant do-everything prompt.
- **PR comments.** Every PR in the release gets a "shipped in vX.Y.Z" comment, and it's idempotent, so re-runs don't double-post.
- **Downstream branches.** Opened automatically on every RC, so integration breakage shows up during the RC window, not after release.

## So, what did it cost?

Almost nothing, which is really a consequence of the open-weights choice, not the headline. A full release (notes plus Slack announcement, across 20 to 40 PRs and several rounds of prompting) runs to **well under a dollar** on Inference Providers (~0.25$). Open weights served pay-as-you-go means the marginal cost of a release rounds to zero, so the only real question becomes "is there something worth shipping?", and weekly, there always is.

## What changed in practice

The cadence went from one release every 4 to 6 weeks to one a week, consistently. The secondary effects were the interesting ones:

- **Notes got *better*, not worse.** A first draft always exists, so review time goes to polishing. Grouping is more consistent and we omit fewer things.
- **Breakages surface earlier.** Downstream test branches on every RC catch integration issues during the candidate window.
- **Contributor loops shortened.** The automatic "shipped in vX.Y.Z" comment turned out to be the sleeper feature. When someone reports an issue on a closed PR, everyone can immediately see which release the fix is in. That used to be a manual tag hunt.

## Make it yours

This is the part we care about most. The workflow is shaped around `huggingface_hub`, but the structure is generic, and the boundary between "generic" and "ours" is easy to see.

**Reusable almost as-is:**

- The trigger and version-bump logic (`minor-prerelease` then `minor-release` then `patch-release`).
- The trust-but-verify loop: deterministic manifest, model draft, validate, re-prompt. This is the transferable idea, independent of *what* you're generating.
- OIDC Trusted Publishing, pinned and checksum-verified runtime, Slack threading.
- The skill-based prompts: swap the templates, keep the structure.

**Specific to us (and obvious where):**

- The downstream repo list and their dependency-pin formats.
- The exact section taxonomy and tone in the skills.
- The Slack and bucket destinations.

To adapt it: fork the file, point it at your package, rewrite the skill Markdown for your project's voice, set two repo variables (the model ID and your OpenCode version), wire up Trusted Publishing on PyPI, and delete the downstream-testing job if you don't have downstreams. The trust-but-verify loop is the part worth stealing wholesale. It's what makes a generated artifact safe to ship.

## What's next

- **Auto-triaging downstream failures.** Today the workflow opens test branches and a human reads the CI. An obvious next step: a job that reads the failing logs and summarizes whether the break is ours, theirs, or a flake.
- **Two-pass notes for big releases.** When a release spans 40 PRs, even a strong model flattens the narrative. We want to cluster PRs first, then write each section against its cluster.
- **Extending the pattern.** Most of this is generic. We expect to reuse large parts across other Python libraries in our ecosystem.

## Takeaway

The parts of a release that used to need a half-day of focused human work (writing notes, drafting announcements, coordinating downstream checks) are exactly the parts a model is good at *drafting*. Everything else is mechanical and fits in a YAML file. The trick was never "let the AI do it"; it was **let the model draft, let deterministic code verify, and let a human decide.** It's built entirely from open tools and open weights, so the cost rounds to zero and anyone can run it.

The full workflow file is public. If you maintain a Python library and recognize the pattern (half-automated, half-manual, release day eats your afternoon), [fork it](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml), adapt it, and let us know how it goes.