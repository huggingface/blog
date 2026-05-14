---
title: "Unlocking asynchronicity in continuous batching" 
thumbnail: /blog/assets/continuous_async/thumbnail.png
authors:
- user: ror
- user: pcuenq
- user: ariG23498
---

# Unlocking asynchronicity in continuous batching

![Title card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/banner.png)

*TL;DR: we explain how to separate CPU and GPU workloads to get a massive performance boost for inference.*

*This is the second post in a series on efficient LLM inference. The [first post](https://huggingface.co/blog/continuous_batching) covered continuous batching from first principles. It introduces some concepts we build upon: KV cache, FlashAttention, attention masks, etc.*

An H200 costs around $5 an hour on [Inference Endpoints](https://endpoints.huggingface.co/). That's cheap for an hour, but use it for a day and you are already paying $120. If this is the case, you want your GPU to be used to its fullest.  
We have seen that Continuous Batching improves GPU utilization by scheduling tightly packed batches, so no compute is wasted on padding. But there is a second source of waste that continuous batching does not address: by default, it is synchronous. This means the CPU and GPU take turns: while the GPU computes, the CPU waits. And while the CPU prepares the next batch, the GPU waits. In a loop running hundreds of steps per second, those idle gaps add up, and as we will show, they can account for nearly a quarter of total runtime. To ensure the GPU is busy computing 100% of the time, we need to get rid of those gaps.  

To achieve this, we can use **asynchronous batching**: we are going to disentangle CPU batch preparation from GPU batch compute, so both can run in parallel and we always have a productive GPU 🔥

## Synchronous batching

This is how naive synchronous batching works:

![Synchronous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/synchronous_batching.png)

When the CPU prepares a new batch, it selects which requests to include, updates the KV cache table, evicts requests that finished in the previous runs, and admits new ones to fill the freed space. Once that is done, it transfers the prepared inputs to the GPU. The GPU runs its forward pass and samples (i.e. chooses) a new token for each request. The results come back to the CPU, so it knows what token each request just produced, then the whole cycle repeats again.

Notice the red annotation on the right: after the GPU finishes computing, it goes idle. The next batch cannot start until the CPU has gone through its update step: sampling the output tokens, updating request states, re-scheduling the batch.

This is the core inefficiency of synchronous batching: the CPU and GPU take turns. While the GPU is computing, the CPU is idle. While the CPU is updating, the GPU is idle. In no circumstances are they both doing useful work at the same time. For a single forward pass this might seem like a small price to pay, but in a continuous batching loop running hundreds of steps per second, these idle gaps accumulate into real throughput loss.

To showcase this, we profile the time spent on CPU and GPU when generating 8K tokens with a batch size of 32 using an 8B model:

![CPU and GPU activity timeline](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/cpu_gpu_phases_sync.png)

_If you want to produce the same kind of graph, you can instrument the continuous batching code to dump CPU and GPU activity spans and use [this script](https://gist.github.com/remi-or/8de44738629c4d3c72451aa01df1a2ab)._

The timeline alternates between green (GPU active, CPU idle) and red (CPU active, GPU idle): the two never overlap. Total generation time is 300.6 seconds, with 24.0% of that spent with an idle GPU waiting for the CPU to finish. Nearly a quarter of all generation time is wasted, from the point of view of the GPU. This is the pessimistic way of viewing things.

The optimistic way is that generation time would drop from 300 to 228 seconds (a free 24% speedup!), if we could eliminate CPU overhead entirely. This requires zero new kernel or model changes, just careful coordination of hardware.

Fundamentally, the idea is simple: we need to figure out how to run batch preparation for batch N+1 while batch N is computing. But this simple idea hides a few technical difficulties:
- How can we launch something on the GPU and get back control to the CPU?
- How can we make sure data is ready, for either CPU or GPU tasks, by the time each task is launched?
- How can we prepare batch N+1 if it is based on the predictions of batch N?

By answering those questions, we are going to build asynchronous batching from scratch. We followed the same steps to implement it as part of continuous batching in the [transformers](https://github.com/huggingface/transformers) library. Feel free to check the code and compare!

## Creating concurrency

Our end goal is to have **concurrent** execution of CPU and GPU operations. We need a way to categorize our operations, so we can let the machine know which operations can run concurrently. We can achieve this using CUDA streams.

### What is a CUDA stream?

To understand how CUDA orders its operations, we need to talk about **CUDA streams**. A stream is an ordered queue of GPU operations (kernel launches, memory copies, synchronization barriers) that executes in the order they were submitted. Every GPU operation is always scheduled inside a stream. Operations within the same stream are sequential: the GPU will not start the next one until the previous has completed. Operations in *different* streams are independent of each other and can run concurrently. To illustrate, if you launch 3 operations across 3 different streams, execution looks like this:

![CUDA streams concurrency](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/stream_concurrency.png)

All three operations start at the same time. This is a slight simplification: every GPU operation is ultimately initiated by the CPU, and that initiation takes a small amount of time: finding the right kernel, issuing the call, transferring the command from CPU to GPU, etc. This is called **CPU launch overhead**, and a more realistic diagram looks like this:

![Realistic CUDA streams concurrency](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/realistic_concurrency.png)

The operations are still concurrent, but their start times are staggered by the cost of each CPU launch. We will keep showing these CPU launch events throughout because they take real time, and they will help us track "what is launched when" as we move to asynchronous workflows. For instance, we will often check if a stream is **flushed**: that means that all operations in a stream have been executed.

### Default and non-default streams

If you have never explicitly used CUDA streams in PyTorch, you might be surprised they exist at all. A typical PyTorch script never mentions them, and it does not *feel* like GPU operations are asynchronous: the CPU seems to wait for the GPU to finish before moving on. That feeling is accurate, and it comes from the **default stream**.

When you call a PyTorch operation without specifying a stream, it lands on the default stream. The default stream has one special property: it is **synchronizing**. If an operation is scheduled on the default stream, it waits for **all other streams to be flushed**, i.e. all work on the GPU has to be over before a single operation on the default stream can start. The reverse is also true: any operation, regardless of its stream, waits for the default stream to be flushed before it launches.

So if you transfer to the CPU the result of a default stream operation, even with a transfer that is supposed to be non-blocking for the CPU, your CPU will still block until all GPU operations have finished because the operations were scheduled on the default stream. This effectively destroys any effort to build concurrency.

That's why we need to use non-default streams. Enqueuing a kernel launch or a non-blocking memory copy returns control to the CPU immediately. The GPU will run the operation in the background, but the CPU does not wait. This answers our first question: to get back CPU control after launching GPU work, we use a non-default stream.

![Blocking vs non-blocking transfer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/block_or_not.png)

For the rest of this post, we will assume all memory transfers from one device to the other are non-blocking. We will therefore have to synchronize them ourselves.

### Back to Continuous Batching

We established that no GPU operation should land on the default stream. But the question remains: if we are not using the default stream, what streams should we use? Let us go back to the synchronous batching figure:

![Synchronous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/synchronous_batching.png)

We can identify three distinct GPU operations:
1. Transfer of inputs from CPU to GPU
2. Compute on the GPU
3. Transfer of outputs from the GPU to the CPU

This means we need three streams: one for compute, one for CPU-to-GPU transfers, and one for GPU-to-CPU transfers. The transfers are independent, so there is no reason to serialize them, and each one gets its own stream.

> [!NOTE]
> A note on nomenclature: when talking about CPUs and GPUs, the convention used throughout the CUDA documentation is to call the CPU the **host** and the GPU the **device**. We will use that convention from now on. CPU-to-GPU transfers are called **host-to-device** (H2D) transfers, and GPU-to-CPU transfers are called **device-to-host** (D2H) transfers. Hence, the three streams are the H2D stream, the compute stream, and the D2H stream.

Let us now try to use streams to asynchronously launch a batch on the GPU and get back CPU control. From the CPU, we do the following:

1. Prepare the batch input data on the CPU (no stream, CPU-only operations)
2. Transfer it to the GPU (using the H2D stream)
3. Run compute on the GPU (using the compute stream)
4. Retrieve the batch outputs (using the D2H stream)
5. Take a look at the results (no stream)

If we do this using only CUDA streams, the results are available almost instantly and they are incorrect. To understand why, let us look at what happened:

![Failed asynchronous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/failed_async.png)

Because streams are independent of each other, all three GPU operations launched at nearly the same time. The compute stream did not wait for the H2D transfer to complete, so the forward pass ran on whatever was already sitting in GPU memory. The D2H stream did not wait for compute to finish, so it transferred results that had not been computed yet. Step 5 returned instantly because nothing was blocking the CPU: there was no default stream to synchronize against.

The operations are all running correctly in isolation. The problem is that we never told the streams to wait for each other. We know that compute must start after H2D completes, and that D2H must start after compute completes, but we did not enforce that ordering. We need a mechanism to say "do not start this operation until that one is done" across stream boundaries.

## Enforcing synchronization

To enforce synchronization between the streams, we are going to use **CUDA events**.

### What is a CUDA event?

A CUDA event is a marker that can be recorded into a stream. When the GPU reaches that marker during execution, it sets the event as completed. Any other stream can then be told to wait for that event before starting its next operation. Concretely, there are two operations: `stream.record(event)`, which inserts the marker into a stream at the current position, and `stream.wait(event)`, which blocks a stream from proceeding until the event is marked complete. Importantly, `wait` blocks the *stream*, not the CPU or other streams running in parallel: the CPU call returns immediately, and only the waiting stream is held back.

![CUDA events](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/events.png)

The figure above shows a single event synchronizing two streams. The CPU issues three operations in rapid succession (the three small blocks): launch input preparation on stream 1, record the event on stream 1, then tell stream 2 to wait for it. Then the CPU continues immediately. Stream 1 runs its operation, and when it completes, the event is set. Stream 2 is held at the wait marker the whole time, and only starts compute once the event is marked complete. The CPU was not involved in any of this: the ordering was enforced entirely on the GPU side.

### Using events in Continuous Batching

Applied to our case, the fix is straightforward. After enqueueing the H2D transfer, we call `h2d_stream.record(h2d_done)`: the event will be marked as completed only when the transfer finishes. Before enqueueing the forward pass, we call `compute_stream.wait(h2d_done)`, so the compute stream will not start until `h2d_done` is set. We do the same between compute and D2H: after launching the forward pass with `model.forward`, we call `compute_stream.record(compute_done)`, then `d2h_stream.wait(compute_done)` before enqueueing the output transfer. The result is a pipeline with explicit ordering:

1. H2D transfer runs on `h2d_stream`
2. `compute_stream` waits for `h2d_done`, then runs the forward pass
3. `d2h_stream` waits for `compute_done`, then transfers the outputs back

The CPU enqueues all of this in sequence, then moves on. At no point does it block. The GPU enforces the ordering through the events, and all three streams are active as soon as their dependency is satisfied.

![Successful asynchronous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/success_async.png)

The figure above shows how this unfolds. The CPU prepares the batch, then quickly enqueues all the GPU work: the H2D transfer, the forward pass, the D2H transfer, with `record` and `wait` calls inserted between each stage. After that, the CPU is free. The GPU takes over, executing each stream in order as its dependency event is set. Notice the green annotation on the right: once the D2H transfer completes, the CPU comes back and reads the results. This final synchronization is the only point where the CPU blocks in the whole step. To implement it, we record a third event on the D2H stream after the output transfer, then call `d2h_done_event.synchronize()` on the CPU side. `synchronize` blocks the CPU until the D2H stream reaches that marker.

This is the key difference from synchronous batching: before, the CPU blocked after every operation. Now, it is free to do "something" while the GPU works.  
We need to figure out what that "something" is, because right now nothing changed from a GPU-utilization standpoint. 

## Filling the vacuum

The window where the CPU is available sits between dispatching batch N and dispatching batch N+1 to the GPU. Its natural use would be to prepare batch N+1's inputs, so we can dispatch them to the GPU and have them be ready once batch N compute is over. Let us see how we can do this.

To prepare batch N+1, we can reuse the same CPU-side objects that prepared batch N: the list of current requests, the state of the cache, the host-side tensor buffers, etc. However, we need to pay attention to two things:
- data corruption: the device-side input buffers for batch N+1 cannot be the same as batch N's: we would corrupt data the GPU is still reading
- data transmission: if a request is in both batch N and N+1, and it produces a new token in the outputs of batch N, that token is needed in the inputs of batch N+1

We address these issues, data corruption and data transmission, in the next two sections.

### Race conditions

First, we are going to tackle the potential data corruption issue.  
Imagine batch N and batch N+1 share the same device-side input buffers, and that the H2D transfer of batch N+1 inputs starts while batch N is still computing. The CPU may write batch N+1's inputs while the GPU is still reading batch N's from the same memory. So the GPU may pick up partially overwritten data, and the result is corrupted. This is a **race condition**. The same risk exists on the host side: reusing the same source for the copy while the H2D copy for batch N is still in flight corrupts the transfer.

The fix is to use two sets of tensors and alternate between them. While the GPU processes batch N from slot A, the CPU updates the requests' state with the results of batch N-1. The CPU next prepares batch N+1 in input slot B. Next step, they swap. This is illustrated in the diagram below:

![Input / Output slots](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/slots.png)

Of course, this comes with a cost: it doubles the amount of RAM and VRAM used to store the input and output tensors. This is an acceptable tradeoff, especially when using [FlashAttention](https://github.com/Dao-AILab/flash-attention), because it does not require an attention mask, which is by far the largest input tensor.

But having two slots creates another problem. In inference, we usually use **CUDA graphs** to reduce latency. In a nutshell, a [CUDA graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) is a pre-recorded sequence of CUDA operations. It is recorded against specific memory addresses: a graph captured for slot A cannot be replayed against slot B's buffers. So we need two graphs. And if each graph has its own memory buffer, that is double the VRAM again.

The solution is a **memory pool**: a shared memory buffer that both graphs allocate from. The only constraint is that two graphs in the same pool must never execute concurrently. Since batch N must finish before batch N+1 starts, that is always the case. In practice, both graphs together use nearly the same amount of VRAM as one. We only pay for two captures at initialization time.  
We can create any number of CUDA graphs in the same pool and the total memory usage is still capped at the maximum across graphs. This is showcased below.

![CUDA graph memory pool](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/memory_pool.png)

Now that we know how to prevent data corruption, we can address the second issue: getting the output tokens of batch N into the inputs of batch N+1.  

### Carry-over

Consider a request that appears in both batch N and batch N+1. In batch N, it produces a new token. That token is its input for batch N+1. The problem is that when we are preparing batch N+1's input buffer, we do not have that token yet: batch N is still running.
To address this, we use a placeholder token when building batch N+1. We will use 0 as a placeholder, for reasons that will become apparent later. We replace that placeholder after batch N is done computing and before batch N+1 starts the forward pass. We call that step the **carry-over**, because we are carrying over the new tokens from batch N to batch N+1. The idea behind carry-over is illustrated below:

![Carry over principle](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/carry_over_idea.png)

To perform carry-over, we only need three things: the output token ids of batch N, the input token ids of batch N+1, and a tensor with instructions on how to perform carry-over. We will call this tensor the **carry-over mask**. It contains the target destination for the tokens that need to be carried over, and -1 for the ones that do not. We represent one below:

![Carry over mask](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/carry_over_mask.png)

The carry-over itself consists of four operations:
- we select the tokens to carry over from batch N's output into a new tensor T
- we zero out the tokens we do not want to carry over in T
- we truncate T to match batch N+1's input length
- we add T to the input ids of batch N+1 (that's why placeholder input ids have a value of zero)

Since those four operations are very cheap, we perform them at the start of each new batch and capture the carry-over in the CUDA graph. If the carry-over mask contains only -1 (a value of -1 means: do not carry over this position) then the last step is an addition with a zero tensor. This does not happen often because decoding requests that span more than one batch are typically scheduled in consecutive batches.

## The full async loop

Let us put everything together and trace through the first two steps.

Step 0 is a cold start: there is no previous batch running, so the CPU prepares batch 0 in slot A and dispatches it as it would with synchronous batching. No overlap yet.

![Asynchronous recap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/async_recap_1.png)

Step 1 is where the async loop begins. The GPU is now running batch 0 on slot A, and the CPU is free. It immediately starts preparing batch 1 in slot B: evicting finished requests, admitting new requests, updating the KV cache routing table, building the carry-over mask. All of this runs in full overlap with the GPU. Once batch 1's inputs are ready, the CPU enqueues the work in sequence: it launches the H2D transfer for slot B, records and waits events for the compute and D2H streams, then moves on.

![Asynchronous recap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/async_recap_2.png)

Now two things happen in parallel on the GPU. On slot A, the GPU finishes compute and sets `compute_done`, which releases the D2H transfer of batch 0's outputs. On slot B, the H2D transfer of batch 1's inputs is running. Once it completes, the `h2d_done` event is set and compute for batch 1 begins. The carry-over from batch 0 to batch 1 is part of that compute: it happens before the regular forward pass. Since slot A and slot B are independent, all of this overlaps freely.

![Asynchronous recap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/async_recap_3.png)

The CPU, meanwhile, blocks on `d2h_done_event.synchronize()` until batch 0's outputs land. Then it processes the outputs, updates the state of all requests that were in batch 0, and starts scheduling batch 2. The loop is now running, and every subsequent step follows exactly the same pattern.  
We illustrate the full workload below. Each slot has a dedicated color for CPU and GPU operations and for events (which are also slot-specific). For readability's sake, we do not show the CPU's launch of GPU operations (like compute or data movement), but they still take place. This is justified because launching a GPU operation has negligible latency compared to the operations shown.

![Asynchronous recap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/async_recap_4.png)

As long as batch N+1's inputs are ready on the GPU when batch N finishes, the GPU never idles between batches. The only question is whether the CPU finishes its work before the GPU finishes compute. That is usually the case: models continue to grow while batch scheduling stays relatively cheap, so GPU compute is the bottleneck, not the CPU.


## Does it actually work?

To find out, we run the same experiment as before: 8K tokens, batch size 32, 8B model.

![CPU and GPU activity timeline](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_async/cpu_gpu_phases_async.png)

The timeline is almost entirely dark green: CPU and GPU running at the same time. The occasional light green slivers are moments where the GPU is active but the CPU has already finished its prep and is waiting. The near-invisible red marks are the sync points between batches, where the CPU blocks to sample batch N's outputs. The GPU is active for 99.4% of total runtime, up from 76.0%. Total generation time drops from 300.6s to 234.5s, a 22% speedup. We predicted 24% if CPU overhead were fully eliminated. The small remaining gap is that unavoidable sync point. No new kernels, no model changes: letting the CPU and GPU work at the same time.

## Conclusion

We started with a synchronous workload where the CPU and GPU worked one after the other, leaving both underused. By moving from schedule-based dependencies to data-based dependencies and refining synchronization points, we managed to disentangle the CPU and GPU workloads, making parallel execution of both hardwares possible. Hence, we were able to saturate the GPU work queue and ensure it is always running. This finally resulted in a large increase of generation speed while maintaining the accuracy of the model. Pretty much a slam dunk.

The full implementation is in the [transformers](https://github.com/huggingface/transformers) library. If you want to see how this translates to actual code, the general entry point for continuous batching is [continuous_batching.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py). The more asynchronous-centric code is located in the [ContinuousBatchingAsyncIOs](https://github.com/huggingface/transformers/blob/5042bb7eb64b69efd351482a05b3803c48955cb4/src/transformers/generation/continuous_batching/input_outputs.py#L609) class.

Asynchronous batching gets us one step closer to unlocking SOTA throughput for long generation, for generation lengths of 16K+ like in reinforcement learning. But there are still some other, smaller things that are also needed to reach that goal. In the next article, we will go through those: offloading requests, decode-specific kernels or fine-grained compile, among others. Stay tuned!

*Acknowledgements: Many thanks to Pedro Cuenca and Aritra Roy Gosthipaty for their help and insightful reviews.*
