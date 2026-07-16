# Grabette: an open system to record robot-manipulation data. And build a shared dataset together

*Record your own manipulation tasks in minutes with a handheld gripper, turn them into robot-ready datasets automatically, and help grow an open, collaborative dataset for robot learning.*

---

## The bottleneck isn’t the model. It’s the data.

Robot learning has a supply problem. We have capable policy architectures (transformer-based VLAs, diffusion and flow-matching policies, and even world models) and the GPUs to train them. What we lack is large, diverse, real-world manipulation data.

Teleoperating a robot to collect it is slow and expensive: every hour of data costs an hour of human time and a robot. That doesn't scale to the millions of demonstrations modern policies need.

But **you don't need a robot to collect robot data.** Just a human hand, a gripper, a camera, and a way to recover the 6-DoF trajectory of what the hand did. Capture the demonstration and you have data a robot can learn from.

That's what we're releasing today: Grabette, an open, low-cost system for recording manipulation data. Pick it up, record a task with your own hand, and get back a clean, robot-ready dataset. No robot, no lab, no teleop rig.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/8qxJy2ykydVPPsdHGDeDV.gif" alt="put_on_grabettes" width="600">
</div>

And that's the bigger goal: if recording a demonstration is as easy as shooting a video, anyone can contribute. We want Grabette to seed a **large, open, collaborative manipulation dataset**. One no single lab could ever build alone.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/9Pn2TjbnL5uHgpxKO3f6s.gif" alt="open_cabinet" width="600">
</div>

---

## Standing on the shoulders of UMI

Grabette is directly inspired by the **Universal Manipulation Interface** (UMI) from Stanford: a handheld gripper with a fisheye camera that records demonstrations "in the wild", recovers camera trajectories with SLAM, and trains visuomotor policies from them.

UMI proved the recipe works. Our goal was to make it effortless to use, to get the barrier from "I have a task" to "I have a trained model" as low as possible. Grabette is built into the modern open ecosystem: LeRobot for datasets, the Hugging Face Hub for sharing, and a processing pipeline you run **from your browser** with nothing to install. Grabette is something anyone can build on a workbench, use in the field, and contribute data from.

---

## Meet Grabette

For months we have been developing Grabette. We are now feeling that it had become usable enough to share!

Grabette is a **handheld gripper** instrumented with everything needed to reconstruct a manipulation demonstration.

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/1vzyFTESXC0csZEAGtmpX.png" alt="grabette_label" width="600">
</div>

It carries **two cameras, each with a distinct job.** Splitting the two roles is deliberate: the cheap wide fisheye gives the policy the context-rich, wrist-camera-style view it needs, while the RGBD camera does the heavy lifting of robust 6-DoF tracking.

And **Grabette** works with **Gripette**, its motorized counterpart.

The family shares the same hardware DNA:
- **Grabette,** the handheld demonstration device (camera + IMU + gripper)
- **Gripette,** the motorized gripper (Feetech servos) that closes the loop on a real or simulated arm

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/jRrDblwGORB1qUA8CUsGE.png" alt="Grabette and Gripette" width="600">
  <p><em>Grabette hand held recording device (left) and Gripette robot gripper (right)</em></p>
</div>

---

## Built for everyone

**Everything is open source**

[Go check the repository!](https://github.com/pollen-robotics/grabette)

- **Hardware**: CAD and production files for Grabette and Gripette
- **Capture service**: the on-device Raspberry Pi software
- **Processing pipeline**: run locally, or online via our Hugging Face Space
- **Example downstream stack**: stock LeRobot training + the OpenArm evaluation, as a reference

**Components.** Standards sensors you can buy, no closed pipeline, no fork lock-in . A Raspberry Pi, a standard Pi camera, an off-the-shelf OAK-D depth camera, magnetic encoders. The whole point is that anyone can build one from parts you can just order.
**Robot-agnostic by design.** Nothing in the capture or the data format assumes a particular arm. Demonstrations are stored as camera-local 6-DoF deltas plus gripper state, the output is a standard LeRobot dataset on the Hugging Face Hub, so the same data can drive different robots and different learning methods. You will still need the matching Gripette gripper on your arm though.

---

## From your hand to a dataset, in two steps

The heart of this release is a recording system built so that going from “I want to demonstrate a task” to “I have a training-ready dataset” is fast and requires no expertise.

### 1. Record

<p align="center">
  <img alt="web_processing" src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/Yyl704g99z0Ivmg1gkb_V.gif" align="left" width="48%">
  <img alt="record_coffee_med" src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/xb6IesUqLiWfZ0eagIT_M.gif" align="right" width="50%">
</p>

<br clear="both"/>


### 2. Process, directly in your browser

![browser-post-process](https://cdn-uploads.huggingface.co/production/uploads/67f8d7925e7217d4ae752fcf/fr-l_fJQdafK1YJXNaFS-.gif)

---

## What can you do with the data? Here’s one example.

A dataset is only interesting if it trains something. So, to show the loop end-to-end, we ship a **complete example.** But to be clear, this is *an* example of what is enables, not the product: the release is the recording system. Your Grabette data works with any method that consumes LeRobot datasets.

Our example takes 200 recorded demonstrations such as :


<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/67f8d7925e7217d4ae752fcf/a2CYRnafEevGNXSA8_Rk1.gif" alt="Grasp a cup episodes" width="40%">
  <p><em>Dataset available on this <a href="https://huggingface.co/datasets/chouziel/dataset-cup-0107" target="_blank" rel="noopener noreferrer">HF Dataset</a></em></p>
</div>

and:

- **Trains a policy** with stack **LeRobot** — a Diffusion Policy (ResNet18 + SpatialSoftmax encoder, DDIM scheduler, 6-D rotation actions) that fits on a single consumer GPU.
- **Evaluates it** on a OpenArm 7-DoF arm with the Gripette gripper, driven over a gRPC API, with this result :


<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/67f8d7925e7217d4ae752fcf/FfLWm-RfgN65XA9Se-Luh.gif" alt="Grasp a cup episodes" width="40%">
  <p><em>Policy available on this <a href="https://huggingface.co/SteveNguyen/pick_cup_0107_smooth-best" target="_blank" rel="noopener noreferrer">HF Model</a></em></p>
</div>


---

## Now it's your turn

The data bottleneck doesn’t get solved by one lab, but by a community recording demonstrations everywhere. You can now help build the dataset: build a Grabette, record tasks you are interested in, and share them on the Hub. Every episode makes the open dataset bigger and more diverse, and every contributor makes robot learning a little less gated behind expensive hardware.

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 1.5rem 0;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/7JFBk7creoqyPyammZCcO.gif" alt="" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/poP5z_D6llgnVi27riI4u.gif" alt="" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/wnWbQP3xI_YPllnnhn0nm.gif" alt="" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/76YUwZ2tTUtvVGJpa1HzP.gif" alt="" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/eUBpuNl-GqDbdaKROx9sF.gif" alt="" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/62f62a29d278a8f3e781e28a/MCnQo_4dVljzE3N_B4BUp.gif" alt="" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">
</div>

---
### What's next

This release is only the start of the project! Grabette will keep evolving. We already have more coming, including a bimanual version with **Casquette**, a head-mounted POV variant for egocentric capture. But the most important next step isn’t only ours, it’s also yours: start recording, and let’s build the dataset together.


👉 **[Build a Grabette (GitHub)](https://github.com/pollen-robotics/grabette)** · [Process your data (HF Space)](https://huggingface.co/spaces/pollen-robotics/grabette-slam) · Contribute to the dataset

*Built with ❤️ by Pollen Robotics.*
