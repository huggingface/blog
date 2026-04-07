---
title: "Our Approach To Speeding Up The Gradio Server" 
thumbnail: /blog/assets/gradio-performance-benchmarking/thumbnail.png
authors:
- user: freddyaboulton
---

Gradio now processes millions of AI inference requests a day. Each of those requests is handled by an AI-native queueing system and backend server—both of which have been tuned over the years to yield the best results for our developers. 

That being said, one of the Gradio team's biggest priorities in 2026 is making the Gradio backend the most performant system for AI workloads. We wanted to share our technical approach for reaching this target to gather your feedback and because we believe these principles can be applied to almost any software project.

We've already shipped several server improvements over the last few weeks using this approach, and we’re just getting started!

---

## Step 1: Determine what matters

There's an old adage: *"You cannot improve what you cannot measure."* The corollary is that you will only improve what you **decide** to measure. Defining your metrics is the most critical step in the process; it determines the entire space of possible optimizations.

For the Gradio team, we chose two "North Star" metrics:
1. **Client Latency:** The total time from when a client sends a request to when they receive a complete result. This is the top-level number we want to move.
2. **Server Processing Time:** The time the server spends actively handling the request. Tracking this helps us isolate inefficiencies in our own implementation versus network overhead.

To find the biggest wins, we broke server processing time into six independent buckets: 
* `upload`: Receiving the files needed for the request.
* `queueing`: Time spent waiting for a worker slot.
* `preprocess`: Preparing raw inputs for the function.
* `fn_call`: The core logic (typically AI model inference).
* `postprocess`: Formatting outputs for the client.
* `streaming_diff`: Computing the delta between conversation states (crucial for LLM chat performance).

---

## Step 2: Trace Your Code

After identifying our metrics, we added **tracing** to the server to capture how long each phase takes in a production-like environment. We implemented a `trace_phase` context manager to wrap specific blocks of logic and store the duration for later analysis. 

It looks like this:

```python
async with trace_phase("fn_call"):
    result = await self.call_function(
        block_fn,
        inputs,
        # ... other args
    )

async with trace_phase("postprocess"):
    data = await self.postprocess_data(
        block_fn, result["prediction"], state
    )
```

## Step 3: The Test Harness

With measurement and tracing in place, we built the **test harness**. This software allows us to run A/B tests between different server implementations under controlled conditions. We designed our harness around four key principles:

* **Reproducibility:** Experiments must be repeatable. We use [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) for on-demand, consistent compute.
* **Transparency:** Data must be verifiable. Results are uploaded to [Hugging Face Buckets](https://huggingface.co/docs/hub/storage-buckets) (S3-like storage powered by Xet).
* **Experimental Control:** Configuration should be effortless.
* **Fidelity:** Tests must match real-world conditions.

This culminated in a command-line utility in the Gradio repo:

```bash
 python scripts/benchmark/remote_runner.py ab \
    --apps scripts/benchmark/apps/image_to_image.py \
    --base main \
    --branch multiprocess-gradio-test \
    --num-workers 2 \
    --tiers 100 \
    --concurrency-limit 36 \
    --mixed-traffic \
    --hardware cpu-upgrade
```

How this maps to our principles:

- Experimental Control: The `--apps` flag lets us test different modalities (Image, Audio, Video) to ensure a fix for LLMs doesn't accidentally regress performance for Computer Vision.
- Fidelity: By setting `--mixed-traffic` and `--tiers`, we simulate the "noisy" environment of a trending Space, rather than testing a single request in a vacuum.
- Reproducibility: The `--hardware` flag ensures that if we run the test tomorrow, any performance delta is due to our code—not a change in the underlying CPU.

## Step 4 - Analyzing the data

Once the runs complete, the data is uploaded to a public [bucket](https://huggingface.co/buckets/gradio/backend-benchmarks/tree/static-workers-run-2). We then use a suite of scripts to automatically analyze the data and generate comparison plots.

We’ve even shared our data schema with coding agents to generate custom visualizations, like the one below:

![](https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/client_threading_image_to_image.png)

Here we compare the median time the client spends waiting for each phase of the request to complete. Our scripts also analyze the p90 and p99 wait times though. In AI applications, the tails of the process time distribution can be fat, and a fast median doesn't matter much if some of your users are waiting for minutes for the request to complete because the server is getting hugged to death.

## In Conclusion

By adopting this scientific approach to performance optimization, we've been able to identify and ship two fixes to the backend already:

* [Run Pre/Post processing for components in a separate thread](https://github.com/gradio-app/gradio/pull/13168)
* [Avoid Polling in SSE Route To Reduce Overhead](https://github.com/gradio-app/gradio/pull/13046)

But we aren't stopping there. Our goal is for any Gradio-related overhead to approach zero. Many exciting changes to come!


