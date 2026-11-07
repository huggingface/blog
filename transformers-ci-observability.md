---
title: "Building CI Observability for Transformers with pytest, OpenTelemetry, and Grafana"
thumbnail: /blog/assets/96_hf_bitsandbytes_integration/Thumbnail_blue.png
authors:
- user: tarekziade
- user: ydshieh
---

# Building CI Observability for Transformers with pytest, OpenTelemetry, and Grafana


## The CircleCI Era

When the `transformers` library first set up CI, the choice was CircleCI. Honestly, no one on the current team remembers exactly why — and that's probably the most honest thing we can say about it. It was there when we arrived, it worked, and nobody had a strong reason to question it. Classic engineering: if it ain't broke, don't fix it.

And for a while, it really wasn't broke. Back in 2021, the test suite covered around 155 models. CircleCI handled PR CI on CPU runners, tests ran, PRs merged, life was good.

Then the library kept growing. And growing.

| Year | Number of model test folders |
|------|------------------------------|
| 2021 | ~155 |
| 2022 | 171 |
| 2023 | 233 |
| 2024 | 281 |
| 2025 | 406 |
| 2026 | 537 |

That's a 3.5x increase in five years. The number of CI jobs stayed roughly the same — what changed was the sheer volume of tests inside each job. At some point, a system that was "good enough" starts to show its age.


## Moving all runners to our infrastructure

The decision to migrate wasn't a single dramatic moment — more like a slow accumulation of frustrations that eventually tipped the scales.

The biggest one was **rate limits**. CircleCI runners were fetching model files and cached artifacts from Hugging Face Hub over the public internet, which meant they were subject to the same rate limits as any regular user. During busy periods, those limits would kick in, jobs would fail for reasons completely unrelated to the code being tested, and whoever was watching the PR dashboard would have to figure out whether the red status meant "real problem" or "Hub said no again." Spoiler: it was often the latter, and it was exhausting.

The second frustration was **trigger control** — or rather, the lack of it. We wanted fine-grained control over when CI runs: for instance, only triggering when a PR is explicitly marked as "ready for review," rather than on every push to a draft. CircleCI's GitHub App pipelines turned out not to support this for PRs opened from forks — which is, of course, where most external contributions come from. We spent a good few weeks trying workarounds: custom webhooks, triggering CircleCI from GitHub Actions via API, an official CircleCI action. None of it worked reliably. We eventually gave up, which felt bad, because it meant community contributors were sometimes confused about why their CI wasn't running. Not a great experience for anyone.

Beyond the pain points, there was also a simpler, quieter reason: **having one system instead of two just makes life easier**. With self-hosted AWS runners co-located on the same infrastructure as our cache, the rate limit problem disappears entirely — the runners talk to the cache over a shared internal filesystem, no public internet involved. And GitHub Actions gives us full control over trigger logic, runner configuration, and workflow behavior, without having to coordinate between two platforms.

The migration itself was gradual. For a period, both CircleCI and GitHub Actions ran in parallel on every PR — the GitHub Actions jobs were non-blocking, so a red status there wouldn't stop a merge. The idea was to build confidence before cutting over completely. It worked. By the time we flipped the switch fully, we'd already ironed out most of the surprises.

One happy surprise: after some runner size tuning, the new setup ended up *faster* than CircleCI. Merge queue times dropped from around 8-9 minutes to about 5. Not the main goal, but we'll take it.

## The Gap : Observability

XXX GH Action alone dont give us the Observability we need.
XXX we needed a circle CI replacement and build our custom UI


## Pytest and OpenTelemetry

XXX explain how it was simple to instrument our pytest suite with OpenTelemetry

XXX show the plugin and a small example

## Grafana, Tempo and Prometheus

XXX Describe the Grafana and Tempo stack
XXX describe the plumbing between grafana and github events to make grafana as live as possible

## Grafana as a wrapper

XXX Explain that more and more poanels are served by a custom Py web server
XXX in iframes. Grafana is just becoming a a grid
XXX an the py app is more efficient using tempo as its DB

## What we plan next


XXX

