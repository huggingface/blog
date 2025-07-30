---
title: "TimeScope: How Long Can Your Video Large Multimodal Model Go?"
thumbnail: /blog/assets/timescope/thumbnail.png
authors:
- user: orrzohar
  guest: true
  org: Stanford
- user: ruili0
  guest: true
  org: Stanford
- user: andito
  guest: false
  org: huggingface
- user: nicholswang
  guest: true
  org: Stanford
---

# TimeScope: How Long Can Your Video Large Multimodal Model Go?

## TL;DR
_TimeScope_ is an open-source benchmark designed to measure how well vision-language models understand long videos. By adding short “needle” clips into videos ranging from 1 minute to 8 hours, it evaluates three skills:

- localized retrieval, 
- information synthesis, 
- fine-grained temporal perception. Timescope reveals that many state-of-the-art models still struggle with true temporal comprehension.

## Table of Contents
- [Why TimeScope?](#why-timescope-motivating-a-better-benchmark-for-video)
- [Benchmark Design](#benchmark-design)
- [Baseline Evaluation Results](#baseline-evaluation-results)
- [Open-Sourcing](#open-sourcing)

Recent advances in multimodal AI have produced models claiming to understand hour-long videos. This trend mirrors progress in long-context language models, which excel at reasoning over lengthy text. Following this, vision-language systems now advertise context windows that can handle thousands of frames. But these claims require a closer look: do these models truly demonstrate understanding of the sequence of events? Are they limited to surface-level retrieval \ recognition? It's crucial to ask if their capabilities are being overstated.


Text benchmarks such as **HELM** and **RULER** have exposed the fragility of long-context claims, showing that models often struggle when tasks demand more than simple retrieval, like reasoning or aggregation at long context lengths. In the video domain, however, we're still playing catch-up. The most common test, **Video Needle in a Haystack (VideoNIAH)**, injects static *images* as "needles" into videos, effectively measuring visual search rather than true temporal dynamics. As a result, even top-tier models advertising massive frame capacities are rarely trained beyond ~256 frames and see sharp drops on benchmarks like **Video-MME** when pushed further.

This measurement gap leaves us wondering: What does it really mean for a model to "understand" long videos? To address this, we're excited to introduce **TimeScope**, a new open-source benchmark hosted on Hugging Face. TimeScope probes the limits of long-video capabilities by inserting several short (~5-10 second) *video clips*—our "needles"—into base videos ranging from 1 minute to 8 hours. With three distinct task types, it evaluates not just retrieval but synthesis, localization, and fine-grained motion analysis, providing a more holistic view of temporal comprehension.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"></script>
<gradio-app theme_mode="dark" space="Apollo-LMMs/TimeScope"></gradio-app>

## Why TimeScope? Motivating a Better Benchmark for Video

The promise of long-video AI is transformative — enabling agents to summarize hours of footage, detect subtle anomalies, and answer complex questions about extended narratives. Integrated into robotics, these models could analyze prolonged operations, adapt in real time, and push autonomous decision-making. Just as powerful is the vision of a personal assistant that understands daily life and offers continuous, actionable feedback.



In practice, this leads to overstated capabilities. Models might claim to process 10,000+ frames, but training data often caps at 256 frames per clip, leading to degraded performance on longer inputs. We've seen this in evaluations where increasing frame sampling rates tanks accuracy on tasks requiring temporal insight.

TimeScope flips the script by emphasizing three pillars of long-video understanding:
1. **Localized Retrieval**: Can the model spot and answer questions about a specific short segment within a vast video?
2. **Information Synthesis**: Can it gather and order details from multiple points across the timeline?
3. **Fine-Grained Temporal Perception**: Can it analyze motion and events in needles that demand dense, multi-frame sampling?


## Benchmark Design

TimeScope’s key idea is using short video clips as “needles,” and instead of just spotting the needle, it pushes models to deeply understand the whole video. We start with a long base video (e.g., a documentary, lecture, or ambient footage) and insert one or more hand-curated short video needles (5-10 seconds each) at random positions. These needles contain the key information needed to solve the task, forcing models to process the entire input without shortcuts like sparse sampling.


  <img src="https://huggingface.co/spaces/Apollo-LMMs/TimeScope/resolve/main/overview.png" alt="Benchmark Design Diagram" style="width: 90%; height: auto;">


*Figure 1: Overview of TimeScope's needle insertion process. A long base video (1 min to 8 hours) serves as the haystack, into which we splice short video needles (~5-10 seconds). Tasks require detecting, synthesizing, or analyzing content from these needles, embedded at varying depths.*

We evaluate across three needle types, each targeting a different aspect of long-video comprehension:

### 1. Localized Retrieval
This tests basic retrieval and understanding of a localized event. Questions are put so that sampling a relevant frame from the needle should suffice—like asking about a shorter part in a longer video.

Example:  
What mode of transportation is shown in the video?  

<video controls>
  <source src="https://huggingface.co/spaces/Apollo-LMMs/TimeScope/resolve/main/train.mp4" type="video/mp4">
</video>

### 2. Information Synthesis
Here, we embed multiple text-based needles (e.g., 2-4 short clips displaying "secret words" via on-screen text) at different points in the video. The model must identify all words and report them in chronological order, simulating tasks like extracting timestamps or key facts from dispersed scenes. This requires scanning the full timeline and understanding relative positioning.

### 3. Fine-Grained Temporal Perception
For questions focusing on motion or sequences within a short clip, single-frame sampling won't cut it—the model needs to perceive dynamics across frames. This probes whether long-context handling preserves temporal fidelity.

Example:  
How many times did the man swing his axe? (a) one (b) two (c) three (d) four (e) five (f) six  

<video controls>
  <source src="https://huggingface.co/spaces/Apollo-LMMs/TimeScope/resolve/main/temporal_wood_cutting.mp4" type="video/mp4">
</video>

With different video lengths and varying needle placements, TimeScope measures how much video a model can really handle—and shows that performance drops as the video gets longer.

## Evaluations & Leaderboard


To kick things off, we ran TimeScope on a suite of leading vision-language models, from open-source favorites to the juggernauts like Gemini 2.5-Pro. The results underscore the benchmark’s value: even models that claim to handle long videos well still struggle with real long-video tasks. These findings reveal clear patterns—performance cliffs around certain durations, strengths in static retrieval versus weaknesses in motion analysis—and pave the way for targeted improvements in model training. For detailed results and visualizations, check out our Hugging Face Space embedded above.

### What did we learn?

**Model size isn’t everything.** Qwen 2.5-VL 3B and 7B, as well as InternVL 2.5 models at 2B, 4B, and 8B parameters, exhibit nearly indistinguishable long-video curves to their smaller counterparts. All of them plateau at roughly the same context length, showing that simply scaling parameters does not automatically grant a longer temporal horizon.

**Gemini 2.5-Pro is in a league of its own.** It is the only model that maintains strong accuracy on videos longer than one hour.

**Trade-offs across tasks matter.** Qwen 2.5-VL shines in the Information-Synthesis (OCR) task—identifying and ordering dispersed text snippets—yet it falls behind on Fine-Grained Temporal Perception, where precise motion counting is required. 




## Conclusion – Let’s Raise the Bar for Long-Video AI

TimeScope demonstrates that “hour-long video understanding” is still more slogan than reality. By revealing where even state-of-the-art models stumble on temporal reasoning, information synthesis, and motion perception, the benchmark invites us to rethink how we train and evaluate multimodal systems.

1. **Run the Demo** – Explore the public Space: <https://huggingface.co/spaces/Apollo-LMMs/TimeScope>
2. **Benchmark Locally** – Evaluate any model with two quick commands:
   ```bash
   pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
   python -m lmms_eval --model-path <your-model> --benchmark timescope
   ```
3. **Join the Leaderboard** – Submit your scores and see how your model compares.

We hope this benchmark helps the community make steady, measurable progress toward models that better understand video over time.



We are open-sourcing all components of TimeScope:

- **Dataset**: [Apollo-LMMs/TimeScope](https://huggingface.co/datasets/Apollo-LMMs/TimeScope)
- **Leaderboard**: [Apollo-LMMs/TimeScope](https://huggingface.co/spaces/Apollo-LMMs/TimeScope)
- **Evaluation Framework**: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
