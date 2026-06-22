---
title: "Shipping huggingface_hub every week with AI, open tools, and a human in the loop"
thumbnail: /blog/assets/huggingface-hub-release-ci/thumbnail.png
authors:
- user: Wauplin
- user: celinah
---

# Shipping `huggingface_hub` every week with AI, open tools, and a human in the loop

`huggingface_hub` is the Python client at the base of the Hugging Face ecosystem. `transformers`, `datasets`, `diffusers`, `sentence-transformers` and dozens of other libraries depend on it to talk to the Hub. Every week we don't ship is a week of fixes and features stuck on `main`.

For a long time we released every 4 to 6 weeks. We now release every week from a single GitHub Actions workflow. We built it using open-source tools and open-weights models and kept a human in the loop at the one place where judgment matters. Nothing in this post requires a vendor contract, a closed model, or infrastructure you can't run yourself. That was a design goal from the start since we wanted a workflow other maintainers could pick up and adapt. 

By the end of this post, you'll have everything you need to build your own.

## Where we started

The old process was half-automated half-manual.

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

Writing good notes for a new version was the heavy part, aggregating tens of PRs on different topics. Nothing technically hard but a few hours of focused attention. Add the announcements on top and a minor release was easily a half-day of work spread over several days.

## Two kinds of work

So we decided to streamline the whole thing. Looking at that list, the work splits in two.

Some steps are purely mechanical and can be automated: bumping the version, committing, tagging, pushing, opening downstream test branches, opening the post-release PR. Nobody needs to think about those. They just have to happen in the right order, every time, which is what a CI workflow is good at.

The rest is different. Writing release notes, deciding what to highlight, phrasing an announcement for a human audience: that's brain work. It's the kind of judgment that kept the release manual for years. This is where AI takes place, turning a blank page into a solid first draft in seconds. It's also where we have to be careful because a draft that looks confident and is subtly wrong is worse than no draft at all.

## The design principle: open parts, reusable by anyone

When we decided to fix this, we set one constraint up front: every moving part had to be something any maintainer could run themselves. No closed model behind an API we couldn't swap, no proprietary release platform, no secret sauce.

Here's the entire stack:

| Part                                                                                              | What it does                                    |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **GitHub Actions**                                                                                | Orchestrates the whole release                  |
| **[OpenCode](https://opencode.ai/)**                                                              | Agent runtime that drives the model             |
| **An open-weights model** (currently [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) from Z.ai) | Drafts the release notes and Slack announcement |
| **[HF Inference Providers](https://huggingface.co/docs/inference-providers/index)**               | Serves the model                                |
| **[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)**                          | Publishes the package                           |

The second principle: the model drafts, a human decides. Language models are good at turning thirty terse PR titles into readable release notes. They are not good at being trusted blindly. So the workflow is human-supervised: the model does the first pass, a deterministic script checks its work, and a human reviews and edits before anything ships (more on that below).

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

From there, the jobs run roughly in this order:

- **Prepare.** Compute the next version, create or reuse the release branch, bump `__version__`, commit, tag, push.
- **Publish to PyPI.** Build and upload `huggingface_hub`. In parallel, build and upload the `hf` CLI as its own PyPI package.
- **Release notes.** Diff the commit range since the last tag, pull PR metadata from the GitHub API, and have the model draft a structured changelog ([here's a recent one](https://github.com/huggingface/huggingface_hub/releases/tag/v1.20.0)). Saved as a *draft* GitHub release.
- **Downstream test branches.** For RCs, open a branch in `transformers`, `datasets`, `diffusers`, `sentence-transformers` with the RC pinned, so their CI tells us fast if we broke something.
- **Slack announcement.** Read the notes and produce an internal announcement in our team voice.
- **Archive notes.** Upload both the raw AI draft and the human-edited version to a Hugging Face Bucket, side by side.
- **Post-release bump.** After a stable release, open a PR on `main` bumping to the next `dev0`.
- **Comment on shipped PRs.** Leave a "this shipped in vX.Y.Z" comment on every PR in the release.
- **Sync CLI docs.** Open a PR to our [skills](https://github.com/huggingface/skills) repo with the regenerated `hf` CLI skill docs.
- **Report to Slack.** Every step posts its status as a thread reply; a final job updates the root message with ✅ or ❌.

The remaining manual steps are reviewing and publishing the draft release notes, and reviewing and posting an internal Slack message. Those two steps are where we want a human in the loop.

## Trust but verify: the human-in-the-loop core

Here's the failure mode everyone worries about with AI-generated release notes: the model quietly drops a PR or invents one that isn't in this release. A changelog that's almost right is worse than no changelog because nobody re-checks it.

We don't trust the generated release notes to be complete on first-shot, we verify it deterministically. Before the model runs, a Python script retrieves all PRs that belong to the release and store them as ground truth.

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

Then model drafts the notes from them. Once done, we check its output against the initial list of PRs:

```python
expected = set(load_manifest())          # what should be there
found    = extract_pr_refs(notes_md)     # what the model wrote (#1234 -> 1234)

missing = expected - found               # silently dropped
extra   = found - expected               # belongs to a different release
```

If anything is missing or extra, we don't fail and we don't ship a wrong file. We hand the discrepancy back to the agent and ask it to fix exactly those PRs:

```python
for _ in range(MAX_ITERATIONS):
    missing, extra = validate(notes)
    if not missing and not extra:
        break  # matches the manifest exactly
    run_agent_fix(missing_prs=missing, extra_prs=extra)
```

This is the pattern that makes the whole thing trustworthy: a non-deterministic model wrapped in deterministic guardrails. The model is great at writing prose and unreliable at being exhaustive. So we let it write and let code enforce the consistency.

## Grounding the model so it doesn't make things up

Completeness is one half. Accuracy is the other. A model summarizing a PR from its title alone will cheerfully invent a code example that doesn't match the real API.

To prevent that, when we fetch PR metadata we also pull the actual documentation diffs from each PR i.e. the unified diff of any `.md` file under `docs/` that the PR touched.

```python
def fetch_doc_diffs(pr):
    return [
        {"filename": f.filename, "status": f.status, "patch": f.patch}
        for f in pr.get_files()
        if f.filename.startswith("docs/") and f.filename.endswith(".md") and f.patch
    ]
```

That diff goes into the model's context so when it writes "here's the new CLI command," it's quoting the example the PR author actually wrote in the docs. That's the same logic as before: give the model real source material and a narrow job.

The prompts themselves live as [Skills](https://github.com/huggingface/huggingface_hub/tree/main/.opencode/skills/hf-release-notes): small Markdown files (`SKILL.md` plus reference templates) checked into the repo. The release-notes skill spells out how to pick highlights, how to structure sections, when to add a doc link, etc. It reads like onboarding instructions, which is exactly the right mental model.

## The human checkpoint

After the RC is published, the draft GitHub release sits there with the AI's first pass in it. This is where the human comes in:

1. A reviewer reads the draft, edits for tone and emphasis, fixes anything the model over- or under-weighted.
2. Only then do they trigger the `minor-release` run, which promotes the RC to final.

The reviewer's time goes into polishing, turning a half-day writing process into a fifteen minutes editing session.

We also keep a paper trail to improve over time. We archive two files side by side to a Hugging Face Bucket: the raw AI draft, uploaded at RC time before anyone touches it, and the human-edited version, uploaded when the final release is cut.

```bash
# at RC time: straight from the model, untouched
hf cp release_notes_raw.txt    "hf://buckets/huggingface/releases/huggingface_hub/${V}/release_notes_raw.txt"

# at release time: after the human review
hf cp release_notes_edited.txt "hf://buckets/huggingface/releases/huggingface_hub/${V}/release_notes_edited.txt"
```

Collecting both every week gives us a growing dataset of "what the model wrote" versus "what we wished it wrote". Dataset that we can then reuse to update the agent's skill.

## Open and secure plumbing

Revamping the release process was a good opportunity to tighten security, especially against supply-chain attacks.

**No PyPI token.** Publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/): PyPI verifies a short-lived OIDC token minted by GitHub for this exact workflow, and issues [PEP 740](https://peps.python.org/pep-0740/) attestations / Sigstore provenance for every artifact. There's no long-lived secret to leak or rotate.

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

## So, what did it cost?

Almost nothing. A full release (notes plus the Slack announcement, across 20-40 PRs and a few rounds of prompting) costs about **0.25$** on Inference Providers. With open weights billed pay-as-you-go, the only real question each week is "is there something worth shipping?", and there always is.

## What changed in practice

The cadence went from one release every 4 to 6 weeks to once a week. The secondary effects were the interesting ones:

- **Notes got better, not worse.** A first draft always exists, so review time goes to polishing. Grouping is more consistent and we omit fewer things.
- **Breakages surface earlier.** Downstream test branches on every RC catch integration issues during the candidate window.
- **Contributor loops shortened.** The automatic "shipped in vX.Y.Z" comment turned out to matter more than we expected. When someone reports an issue on a closed PR, everyone can immediately see which release the fix is in. That used to be a manual tag hunt.

## Make it yours

This is the part we cared about most. The workflow is shaped around `huggingface_hub` but the structure is generic.

**Reusable almost as-is:**

- The trigger and version-bump logic (`minor-prerelease` then `minor-release` then `patch-release`).
- The trust-but-verify loop: deterministic manifest, model draft, validate, re-prompt. This is the transferable idea, independent of what you're generating.
- OIDC Trusted Publishing, pinned and checksum-verified runtime, Slack threading.
- The skill-based prompts: swap the templates, keep the structure.

**Specific to us:**

- The downstream repo list and their dependency-pin formats.
- The exact section taxonomy and tone in the skills.
- The Slack and bucket destinations.

To adapt it: fork the [workflow file](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml) and [scripts](https://github.com/huggingface/huggingface_hub/tree/main/utils/release_notes), point it at your package, rewrite the [skill Markdown](https://github.com/huggingface/huggingface_hub/blob/main/.opencode/skills/hf-release-notes/SKILL.md) for your project's voice, set two repo variables (the model ID and your OpenCode version), setup Trusted Publishing on PyPI, and delete the downstream-testing job if you don't have downstreams. The trust-but-verify loop is the part worth reusing as-is. It's what makes a generated artifact safe to ship.

## What's next

- **Auto-triaging downstream failures.** Today the workflow opens test branches and a human reads the CI. An obvious next step is to check the failing logs to report them in the internal slack message.
- **Extending the pattern.** Most of this is generic. We expect to reuse large parts across other Python libraries in our ecosystem.

## Takeaway

The parts of a release that used to need a half-day of focused human work (writing notes, drafting announcements, coordinating downstream checks) are the parts a model is good at drafting. Everything else is mechanical and fits in a YAML file. The trick was never just "let the AI do it" — it's to let the model draft, let deterministic code verify, and let a human decide. It's built entirely from open tools and open weights so the cost rounds to zero and anyone can run it.

The full workflow file is public. If you maintain a Python library, [fork it](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml), adapt it, and let us know how it goes!