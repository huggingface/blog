---
title: "Asynchronous Robot Inference: Decoupling Action Prediction and Execution"
thumbnail: /blog/assets/async_inference/thumbnail_async_blog.png
authors:
- user: fracapuano
date: 2025-06-30
slug: async-inference
---

## TL;DR
Robotic policies are increasingly bulky, and predict chunks of future actions rather than a single next action. This results in the robot being idle while awaiting new actions to perform, introducing noticeable lags at execution, and lacking of responsiveness. Asynchronous inference tightens the control loop, removing lags at runtime and resulting in more adaptive control by decoupling action prediction from action execution.


# Async inference

## Getting started
Install LeRobot following our installation guide (make sure to instal the `async` dependancies by running `pip install -e ".[async]"`). Then, spawn a `PolicyServer` in one terminal tab by running:

Once your server is live, spawn a `RobotClient` connected to the robot you wish to control and the `PolicyServer` you just spawned by running:


For more information, and an in-detail API example follow [our tutorial](NOTE:LINKTODOCUMENTATION) to get started with async inference.

## Async inference: a deep dive

With async inference, we decouple action execution from action prediction. This is particularly relevant considering the tendency of currently popular models like [[1, ACT](https://arxiv.org/abs/2304.13705)], [[2, OpenVLA](https://arxiv.org/pdf/2406.09246)], [[3, $\pi_0$](https://www.pi.website/download/pi0.pdf)], and [[4, SmolVLA](https://arxiv.org/abs/2506.01844)] to be outputting chunks of actions $a_{t:t+H}$ rather than single actions $a_t$ given an observation $o_t$. 
Convince yourself of this by running all these models using [LeRobot](https://huggingface.co/lerobot).

Using chunks sequentially results in (1) lags at runtime, impacting task execution time and (2) lack of responsiveness, due to acting widely open-loop.
Asynchronous inference avoids mitigates both these limitations by **decoupling action prediction from action execution**. 
We introduced asynchronous inference in [SmolVLA](https://arxiv.org/abs/2506.01844), and found it to result in a ~2x speedup in task completion time with comparable task success rate.

In particular, we design a 2-component system where policy inference and action execution are performed in two different processes, possibly on two different machines connected through the network:
* A **`PolicyServer`**, hosted on acceleratedhardware and capable of running inference using more computational resources than the ones allocated on a real-world robot.
* A **`RobotClient`** enqueues the received actions and executes them while the next chunk is being computed.

Communication between `PolicyServer` and `RobotClient` relies on **gRPC**, which guarantees ~5× faster performance than a comparable REST API. The result of all of this is a robot that *never* waits for inference.

<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/async_scheme.png" alt="Async inference scheme"/>
</p>

<p align="center"><i>Asynchronous inference</i>, highlighting: (1) The client sending the first observation for inference, receiving the first chunk shortly after; (2) The client sending another observation for processing while it has not yet exhausted the current chunk; (3) The client receiving an updated action chunk, which it aggregates with the remaineders of the one it was previously executing.
</p>

---

## 1. Why sequential inference falls short

Suppose a policy $ \pi $ maps the current observation $ o_t $ to a sequence of $ n $ future actions
```math
\pi : \mathcal{O} \;\longrightarrow\; \tilde{\mathcal A},
\begin{pmatrix} a_{t} \\ a_{t+1} \\ \vdots \\ a_{t+n} \end{pmatrix} = \pi(o_t)
```

A traditional control loop would therefore:

1. Capture $ o_t $.
2. Run $ \pi(o_t) $ to obtain $ \mathbf{A}_t = \pi(o_t) $.
3. Enqueue $\mathbf{A_t} $ and start acting popping actions from the queue.
4. If the queue is empty, wait for $ \mathbf{A}_{t+H} $, otherwise repeat step 3.

During step 2 the robot is **idle**. The latency grows with the model size (and models tend to be increasingly bulky over time), and can quickly dominate interaction time (which is typically around 1/`fps`), as shown in the video below (coming from our [Discord community](NOTE:LINKTODISCORD) 🤗):

<p align="center">
  <video width="600" height="400" controls>
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/lags_videos.mp4" type="video/mp4">
  </video>
</p>

This directly results in (1) reduced performance in terms of task completion time---the robot needs to be waiting for the next action chunk to be computed---and (2) reduced responsiveness, due to (2.1) acting widely open-loop while actions are available and (2.2) complete idleness while waiting for the next action chunk.

<div style="display:flex; align-items:center; justify-content:center; gap:12px;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/sync.png" alt="Sequential inference – idle periods highlighted" style="width:66%;"/>
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/time_to_select_action.png" alt="Time to select action – spikes indicate inference" style="width:33%;"/>
</div>

<p align="center">(Left)<i>Sequential inference</i> with highlighted idle periods. (Right)<i>Time to select an action</i> showing spikes when inference is triggered due to local queue exhaustion (inference latency is around ~100ms---~3 frames at 30fps---using an ACT model on a 2021 MacBook Pro).</p>

---

## 2. Asynchronous inference, in a nutshell
Our system removes the idle period by overlapping computation and execution:

1. `RobotClient` streams the latest observation to `PolicyServer`.
2. While the server performs inference, the client executes the **current queue** of actions.
3. New actions arrive, are merged into the queue, and the loop continues.

The key idea is that the robot already knows what to do for the next few timesteps, so it can keep moving while fresh actions are being computed on the server.

<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/6f323660-52b4-4537-8bde-f9b70b7f1bc0" alt="Async inference diagram"/>
</p>
<p align="center"><i>Asynchronous inference</i> overlaps in time the execution of the current action chunk with the computation of the next one, by decoupling these two processes, possibly running them on entirely distinct machines connected through the network.
</p>

This results in a tighter control loop, and a robot that never waits for inference. In turn, this results in ~2x speedup in task completion time with comparable task success rate, and more adaptive control coming from a tighther loop (see video below).

<p align="center">
  <video width="600" height="400" controls>
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/async.mp4" type="video/mp4">
  </video>
</p>

---

## 3. System Architecture
| Component | Role | Technology |
|-----------|------|-------------|
| **RobotClient** | Runs on-board, streams observations, maintains an **action queue**, executes actions | Python, gRPC |
| **PolicyServer** | Hosts the policy, performs batched inference, sends action chunks back | Python, gRPC, possibly accelerated hardware (GPU/TPU) |

Because gRPC is HTTP/2-based and uses protocol buffers, it achieves low-latency binary messaging and bidirectional streams out of the box, which in turn helps us maintain a tighter control loop and sub-100ms round-trip latency (on our local network, and hosting SmolVLA on a NVIDIA RTX 4090).

The `RobotClient` runs on-board, and streams observations to the `PolicyServer` through gRPC. The `PolicyServer` prepares the observations received for inference, and sends back to the `RobotClient` an action chunk.


### `RobotClient`

<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/from_client_perspective.png" alt="From client perspective"/>
</p>
<p align="center"><i>From the client's perspective</i>, observations are streamed to the server according to the local queue status. Incoming chunks are aggregated on overlapping portions with the currently available action queue.
</p>

The `RobotClient` maintains a local action queue and follows a simple yet effective strategy: **send a new observation when the queue length drops below a configurable threshold** ($g$ in the SmolVLA paper, `chunk_size_threshold` in the code). 
This threshold value, expressed as a fraction of the maximum chunk size, acts as a trigger condition that balances computational load with responsiveness.

<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/client_to_server.png" alt="Client to server"/>
</p>
<p align="center"><i>The client streams observations to the server</i>, according to the local queue status.
</p>

From the client's perspective, the process unfolds as follows:

1. **Queue monitoring**: The client continuously monitors its action queue length against a **chunk size threshold** parameter. When the queue drops below this threshold, it signals that a new observation should be sent for processing.

2. **Observation streaming**: Once the threshold condition is met, the client captures the current observation and streams it to the `PolicyServer` via gRPC. Crucially, **observations are streamed rather than being sent via a unary RPC** because they typically exceed the maximum message size of 4MB (multiple camera captures at high resolution result in this).

3. **Action chunk aggregation**: When a new action chunk arrives from the server, the client merges it with any remaining actions in the current queue over the overlapping portion. This is where **custom aggregators** come into play, handling overlapping sections between the current and incoming chunks differently. As of now, we support flexibly aggregation between the chunks via the specification of a custom `aggregate_fn(chunk1: torch.Tensor, chunk2: torch.Tensor) -> torch.Tensor` function, which is called for each overlapping timestep and can be user-provided.
The overlapping portions (shown in light blue in the diagram) require careful handling. We can design different aggregation strategies:
   - **Replace**: Simply replace overlapping actions with the newer predictions
   - **Weighted blend**: Combine overlapping actions using temporal weights (closer actions get higher weight)


This system is highly configurable, as the chunk size threshold can be tuned based on network latency, model inference time, and desired responsiveness. 
A lower threshold means more frequent updates (and higher computational cost), while a higher threshold reduces communication overhead at the expense of potential queue starvation.
Lastly, we typically receive actions from `PolicyServer` in a thread, and perform them in another one. This keeps the client listening for incoming chunks in a separate thread, without blocking execution and always consuming the current chunk until a new one becomes fully available.

### `PolicyServer`

Upon receiving observations from the `RobotClient`, the `PolicyServer` receives observations from the `RobotClient`, and performs the necessary observation cleaning to make received observations ready for inference. This process is illustrated in the image below:
<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/server_pipeline.png" alt="Server pipeline"/>
</p>
<p align="center"><i>The observation cleaning pipeline</i> running on the server, highlighting the three main steps related to (1) Keys matching (2) Preprocessing and (3) Preparation for inference.
</p>

Once the observation has been prepared, it is compared with the last observation used for inference. 
This avoids collapsing into a loop whereby very similar observations are processed, thus triggering unnecessary inference and similar actions being executed (which in turn, result in very similar observations being processed again). 
We compare observations in terms of their joint-space similarity, which provides us an approximate and quick way of measuring changes in the robot. Clearly, this metric is not adaptive to dynamic changes in the environment (an object changing its position, or disturbances being applied), but we found it to be a good trade-off for the majority of the cases, and to be very effective in avoiding unnecessary inference and state collapse.
Critically, the `RobotClient` retains control over whether a given observation must be processed, to avoid deadlocks. 
Observations sent by the client and tagged with `must_go=True` are processed regardless of the similarity metric.

<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/policy_workflow.png" alt="Policy workflow"/>
</p>
<p align="center"><i>The policy workflow</i>, in which incoming observations are compared to the last one used for inference, and processed only if different enough, or `must_go`.
</p>

Lastly, to ensure the `PolicyServer` always processes the latest available observation, we block incoming observations until the previous one has been successfully processed. In this, we leverage queues on the `PolicyServer` to ensure incoming observations are not enqueued until the server is ready to process them (see below).

<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/client_pings_server.png" alt="Client pings server"/>
</p>
<p align="center"><i>The client pings the server every 1/fps seconds</i>, but observations are not enqueued for processing until the previous one has been successfully processed.
</p>

---

## 4. Analyzing async inference

For all practical purposes, in async inference there are two time-scales that matter:

* **Environment step** $\texttt{environment\_dt} = 1/\texttt{fps}$, depicting how fast the robot can perform an action.
* **Inference latency** $\texttt{inference\_time}$: forward-pass + network round-trip. We can assume the network round-trip to be negligible with respect to the policy inference time, though this might not be the case for every setup.

Importantly, the ratio
$$
 c = \frac{\texttt{environment\_dt}}{\texttt{inference\_time}}
$$
results in different behaviours:

* $c \ll 1$: environment evolves faster than inference. In this scenario, the queue empties quickly and we degenerate to sequential control.
* $c \ge 1$: server keeps up. The queue is always (nearly) full.

Critically, $c$ influences the number of available actions in the queue at any given time. To avoid the aforementioned sequential limit control, one can:
1. **Use more compute for the policy server**, hosting the server on a GPU, reducing $\texttt{inference\_time}$ as a consequence of allocating more computational resources.
2. **Sending observations to the server more often**, send a new observation when the queue length $k$ drops below a **fraction** $g = k/H$ of its maximum size.
   * $g=0$ reproduces sequential inference (empty queue, wait).
   * $g=1$ sends an observation every timestep (max compute, minimal lag).

Experiments (see plots below) show that $g\approx0.7$ offers a good trade-off when observations sent are not filtered out (they are all must-go). We recommend setting $g=0.5$ and following [our documentation](NOTE:LINKTODOCUMENTATION) to tune this parameter to your needs.

<p align="center">
  <img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/async-inference/queues.png" alt="Queues"/>
</p>
<p align="center"><i>The number of available actions in the queue at any given time</i>, as a function of g. Larger values of g result in more frequent updates, and more computational cost. Values of g closer to 0 reproduce sequential inference (empty queue, wait). We found g~0.7 to be a good trade-off in our experiments.
</p>

# Conclusions

We have introduced async inference, a simple yet effective way to improve the performance of robotic policies. In our experiments using SmolVLA, async inference results in a ~2x speedup in task completion time with comparable task success rate, and more adaptive control coming from a tighter loop.

We are excited to share this work with the community, and to see how it can be used to improve the performance of robotic policies. We welcome PRs to improve and extend the async inference framework at `huggingface/lerobot`, and are available to discuss this further in our [Discord community](NOTE:LINKTODISCORD), 🤗.

