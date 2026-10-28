---
title: "Announcing Project ProtoStar: Making Publicly Funded Science AI-Ready"
thumbnail: /blog/assets/project-protostar/thumbnail.png
authors:
- user: evijit
  org: hfpolicy
- user: sayeedmd
  guest: true
  org: CMU-OSPO
- user: gtananbaum
  guest: true
  org: ORCA-Open
- user: EmmaScharfmann
  org: hugging-science
- user: cgeorgiaw
  org: hugging-science
- user: EuniceML
  guest: true
  org: ORCA-Open
- user: irenesolaiman
  org: hfpolicy
---

# Announcing Project ProtoStar: Making Publicly Funded Science AI-Ready

*An initiative bringing together universities, research organizations, industry, and the open science community to make publicly funded research easier to discover, reuse, and build upon.*

![Project ProtoStar: Public research, put to work](/blog/assets/project-protostar/thumbnail.png)

Today, we're announcing **[Project ProtoStar](https://huggingface.co/project-protostar)**, an initiative to unlock the potential of publicly funded scientific research by making datasets, models, and code more accessible and reusable, particularly for AI-driven research. ProtoStar brings together research institutions, industry, and the open science community to support the release of scientific artifacts in AI-ready, machine-actionable formats, and to demonstrate their value through a series of open challenges. The first challenge, hosted by [Carnegie Mellon University](https://www.cmu.edu/), opens on November 2. See [How to Participate](#how-to-participate) below.

The initiative's founding partners include [Carnegie Mellon University](https://www.cmu.edu/), [Purdue University](https://www.purdue.edu/), [Georgia Institute of Technology](https://www.gatech.edu/), [Arizona State University](https://www.asu.edu/), and [The George Washington University](https://www.gwu.edu/), all working through the [Open Forum for AI (OFAI)](https://www.openforumai.org/) as the university coalition partner; the [Astera Institute](https://astera.org/); [Hugging Face](https://huggingface.co/) as the infrastructure partner; and the [Open Research Community Accelerator (ORCA)](https://orcaopen.org/), through its [Fostering Innovation Network](https://orcaopen.org/work/fostering-innovation-through-open-science-fin), as the community partner.

Participating organizations formalize their involvement through a memorandum of understanding supporting the release of suitable scientific datasets, models, and code in AI-ready formats on open infrastructure. Organizations interested in joining Project ProtoStar can reach out to [protostar@orcaopen.org](mailto:protostar@orcaopen.org).

## Making publicly funded science more usable

The U.S. federal government invests approximately [$190 billion annually in research and development](https://www.congress.gov/crs-product/R48694), and federal policy increasingly expects the results to be shared publicly. Agencies such as the [National Institutes of Health (NIH)](https://www.nih.gov/) and [National Science Foundation (NSF)](https://www.nsf.gov/) have strengthened public-access requirements for funded research publications and require researchers to plan for sharing the underlying data. Hugging Face is among the [ten generalist repositories listed by NIH](https://www.nlm.nih.gov/NIHbmic/generalist_repositories.html) for sharing research data.

**But making research publicly available does not automatically make it usable.** Scientific datasets, models, and code are often scattered across incompatible systems, stored in static or legacy formats, or released without the documentation, metadata, and provenance needed to understand and reuse them. These limitations make it harder for researchers to build on existing work, and harder still for AI systems to discover, interpret, and combine scientific resources across disciplines.

Project ProtoStar aims to help close this gap by encouraging systematic sharing of publicly funded scientific artifacts in AI-ready formats, supported by open infrastructure and practical publishing guidance. It builds on efforts such as the [National Cancer Institute's AI Data-Readiness Challenge](https://datacommons.cancer.gov/news/nci-crdc-artificial-intelligence-data-readiness-aidr-challenge) and [Hugging Face's guide to preparing public-sector data for AI](https://huggingface.co/blog/evijit/public-org-data-ai). The goal is to make it easier for research organizations to meet existing public-access expectations while increasing the scientific value of the work they release.

The demand for reusable scientific resources is already evident. **[Hugging Face](https://huggingface.co/) hosts more than 17,000 public datasets across core scientific domains**, including genomics, protein design, chemistry, weather and climate, Earth observation, and astronomy. That figure approaches 29,000 when medicine and mathematics are included, alongside a comparable number of scientific models.

The pace of sharing is also accelerating. **Researchers shared more than three times as many new science datasets on Hugging Face in the first nine months of 2026 as in the same period of 2025**, with growth remaining substantial even after excluding the largest individual uploader.

<iframe src="https://evijit-hf-science-growth.static.hf.space/embed.html?chart=quarterly-datasets&theme=dark" title="New science datasets per quarter (Hugging Face Hub)" width="100%" height="640" style="border:0" loading="lazy"></iframe>

*Figure 1. New public science datasets on Hugging Face per quarter, by domain (Q1 2023–Q3 2026). Counts datasets still public, by creation date. Source: Hugging Face Hub data, October 2026. [Explore the data](https://huggingface.co/spaces/evijit/hf-science-growth).*

When organizations share research in reusable formats, the scientific community can build on it at scale. The 18 public repositories maintained by NIH's [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/) on Hugging Face have collectively received more than 11 million downloads, while [NASA IMPACT's](https://huggingface.co/nasa-impact) 27 repositories have received approximately 770,000 downloads, as of October 26, 2026. Documentation is also an important part of making these resources useful: [90% of dataset downloads on Hugging Face are from fully documented datasets](https://arxiv.org/abs/2401.13822).

Publicly available genomic data has proved important in contributing to research that further advances fields such as biological research; researchers at Hugging Face have been building on openly available scientific resources, including recent releases from [HuggingFaceBio](https://huggingface.co/HuggingFaceBio) such as [Carbon-A](https://huggingface.co/blog/HuggingFaceBio/carbon-annotator-genbank-genome-annotation) and [Carbon-X](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release). By releasing the resulting models, datasets, and tools openly, these efforts demonstrate how existing scientific resources can support new research that others can build upon.

This is the kind of activity that we are further catalyzing via Project ProtoStar through institutional policies adopted by participating orgs. As an early example of institutional commitment, CMU's Open Source Programs Office is encouraging its researchers to share suitable datasets, AI models, and other research outputs on Hugging Face, while recognizing the continued importance of domain-specific repositories.

## Open Science Challenges

[Making scientific artifacts AI-ready](https://huggingface.co/blog/evijit/public-org-data-ai) is an important first step. We also want to demonstrate what researchers, developers, and the broader scientific community can accomplish when those resources are made available for reuse.

A central question for ProtoStar is: **Can publicly funded scientific datasets, models, and code, released in AI-ready formats, be readily discovered, reused, combined, and extended to accelerate scientific discovery and public benefit?**

To explore this, ProtoStar will organize a series of open challenges, each hosted by a participating research institution and built around scientific resources that the institution has released openly. The challenges have two complementary goals:

- **Demonstrate scientific reuse.** Produce concrete examples of researchers and developers building on publicly funded scientific artifacts, including applications that combine resources across institutions or disciplines.
- **Learn what makes reuse possible.** Identify technical barriers, documentation gaps, interoperability challenges, and successful practices that can inform future research infrastructure, funding decisions, and public policy.

The resulting projects and lessons learned will help build a practical evidence base for making publicly funded science more accessible and useful.

### The first challenge: Climate Risk Intelligence

![CMU Climate Risk Intelligence Challenge, November 2–13, 2026](/blog/assets/project-protostar/climate-challenge.jpg)

**Hosted by [Carnegie Mellon University](https://www.cmu.edu/) | November 2–13, 2026**

The inaugural ProtoStar challenge invites participants to test whether AI can improve predictions of urban climate conditions using historical weather data from 210 U.S. cities. Building on [research by Carnegie Mellon University](https://doi.org/10.1184/R1/7890488.v6), teams will use climate records through 2018 to predict actual conditions from 2019 to 2026, allowing their models to be evaluated against observed weather rather than hypothetical future scenarios.

Participants will forecast four indicators relevant to urban planning: average annual temperature, total annual precipitation, the hottest day of the year, and cooling degree days, a measure of demand for air conditioning. Their predictions will be compared against simple statistical baselines, including historical averages and linear trends, to assess whether AI approaches offer meaningful improvements in accuracy.

Beyond forecasting, participants will also propose a short climate adaptation policy idea informed by what they learned from the data. This component encourages teams to connect scientific analysis with practical public benefits, such as preparing cities for extreme heat, improving energy efficiency, or managing changing precipitation patterns.

Teams will submit a Hugging Face dataset with these outputs. Forecasting performance will be assessed through a public leaderboard measuring prediction errors against established baselines, while policy ideas will receive separate consideration. Final forecasting results will be published in early 2027, once the complete 2026 weather records are available.

**[Explore the CMU Climate Risk Intelligence Challenge](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence)**

The CMU challenge is the first in this series. Other participating institutions will host their own challenges sequentially, each highlighting different publicly funded scientific resources and opportunities for reuse. Future challenges will be announced through the [Project ProtoStar page on Hugging Face](https://huggingface.co/project-protostar).

## How to Participate

Whether you're an academic researcher, student, independent developer, or company building on open scientific infrastructure, ProtoStar offers opportunities to put publicly funded research to work and help us learn what is needed to make AI-ready open science more widely useful.

We also welcome universities, federal research agencies, national laboratories, nonprofits, and other research organizations interested in making their scientific resources more accessible and reusable. **To explore participation or collaboration, reach out to [protostar@orcaopen.org](mailto:protostar@orcaopen.org).**

Our hope is that ProtoStar will make it easier for publicly funded scientific work to reach the people and tools that can build on it, while generating practical evidence about the infrastructure and practices needed to support open science at scale.
