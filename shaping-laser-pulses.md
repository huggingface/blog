---
title: Shaping Laser Pulses with Reinforcement Learning
thumbnail: https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/Figure1.png
authors:
- user: fracapuano
---

# Table of Contents
- [TL;DR](#tl-dr)
- [Shaping Laser Pulses](#shaping-laser-pulses)
- [Automated approaches](#automated-approaches)
- [BO's limitations](#bos-limitations)
- [RL to the rescue](#rl-to-the-rescue)


## TL; DR: 
We train a Reinforcement Learning agent to **optimally shape laser pulses** from readily-available diagnostics images, across a range of dynamics parameters for intensity maximization.
Our method **(1) completely bypasses imprecise reconstructions** of ultra-fast laser pulses, **(2) can learn to be robust to varying dynamics** and **(3) prevents erratic behavior** at test-time by training in coarse simulation only.

<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/Figure1_and_CPA.png" alt="Phase changes animation">
    <p> (A) Schematic representation of the RL pipeline for pulse shaping in HPL systems. (B) Illustration of the process of linear and non-linear phase accumulation taking place along the pump-chain of laser systems.</p>
</div>

By opportunely controlling the phase imposed at the stretcher, one can benefit from both energy and duration gains, for maximal peak intensity.

---

## Shaping Laser Pulses

Ultra-fast light-matter interactions, such as laser-plasma physics and nonlinear optics, require precise shaping of the temporal pulse profile.
Optimizing such profiles is one of the most critical tasks to establish control over these interactions. 
Typically, the highest intensities conveyed by laser pulses can usually be achieved by compressing a pulse to its transform-limited (TL) pulse shape, while some interactions may require arbitrary temporal shapes different from the TL profile (mainly to protect the system from potential damage).


<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/phase.gif" alt="Phase changes animation">
    <p>Changes in the spectral phase applied on the input spectrum (left) have a direct impact on the temporal profile (right).</p>
</div>

In this work, we shape laser pulses by varying the GDD, TOD and FOD coefficients, effectively tuning the spectral phase applied to minimize temporal pulse duration.

<!-- add link to space demo -->

## Automated approaches

The most common automated laser pulse shape optimization approaches mainly employ black-box algorithms, such as Bayesian Optimization (BO) and Evolutionary Strategies (ES). These algorithms are typically used in a closed feedback loop between the pulse shaper and various measurement devices.

For pulse duration minimization, numerical methods including BO and ES require precise temporal shape reconstruction, to measure the loss against a target temporal profile, or obtain derived metrics such as duration at full-width half-max, or peak intensity value.

Recently, approaches based on BO have gained popularity because of their broad applicability and sample efficiency over ES, often requiring a fraction of the function evaluations to obtain comparable performance.
Indeed, in automated pulse shaping, each function evaluation requires one (or more) real-world laser bursts. Therefore, methods that directly optimize real-world operational hardware are evaluated based on their efficiency in terms of number of the required interactions.

### BO's limitations 

While effective, BO suffers from limitations related to (1) the need to perform precise pulse reconstruction (2) machine-safety and (3) transferability. To a large extent, these limitations are only more significant for other methods such as ES.

#### 1. Imprecise pulse reconstruction
BO requires accurate measurements of the current pulse shape to guide optimization. However, real-world pulse reconstruction techniques can be **noisy or imprecise**, leading to poor state estimation, and increasingly high risk of applying suboptimal controls.

<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/reconstructing_frog.png" alt="Phase changes animation" width="70%">
    <p>Temporal profiles with temporal-domain reconstructed phase (top) versus diagnostic measures of the burst status (bottom), in the form of FROG traces. Image source: Zahavy et al., 2018.</p>
</div>

#### 2. Dependancy on the dynamics
BO typically optimizes for specific system parameters and **doesn't generalize well when laser dynamics change**. Each new experimental setup or parameter regime may require re-optimizing the process from scratch!

This follows from standard BO optimizing a typically-scalar loss function under stationarity assumptions, which can prove rather problematic in the context of pulse-shaping. This follows from the fact day-to-day changes in the experimental setup can quite reasonably result in non-stationarity: **the same control, when applied in different experimental conditions, can yield significantly different results**.

<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/B_integral.png" alt="Phase changes animation" width="70%">
    <p>Impact of experimental conditions only, in this case a non-linearity parameter known as "B-integral", on the end-result of applying the same control.</p>
</div>

#### 3. Erratic exploration

BO can endanger the system by applying **abrupt controls at initialization**. Controls are applied as temperature gradients applied on a gated-optical fiber, and as such successive controls cannot typically vary significantly because the one-step difference in temperature difference cannot vary arbitrarily.

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
    <div>
        <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/pulses_anim.gif" alt="BO temporal profile">
    </div>
    <div>
        <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/control_anim.gif" alt="BO exploration">
    </div>
</div>
<p>BO, (left) temporal profile obtained probing points from the parameters space and (right) BO, evolution of the probed points as the parameters space is explored.</p>

## RL to the rescue

In this work, we address all these limitations by **(1) learning policies directly from readily-available images**, capable of **(2) working across varying dynamics**, and **(3) trained in coarse simulation to prevent erratic-behavior** at test time.

First, (1) we train our RL agent directly from readily available diagnostic measurements in the form of 64x64 images. This means we can **entirely bypass the reconstruction noise** arising from numerical methods for temporal pulse-shape reconstruction, learning straight from single-channel images.

<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/Figure1.png" width="50%">
    <p>Control is applied directly from images, thus learning to adjust to unmodeled changes in the environment. </p>
</div>

Further, (2) by training on diverse scenarios, RL can develop both **safe and general control strategies** adaptive to a range of different dynamics. In turn, this allows to run and lively update control policies across experimental conditions.
<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/udr_vs_doraemon_average.png" width="50%">
    <p>We can retain high level of performance (>70%) even for larger---above 5, fictional---levels of non-linearity in the systems. This shows we can retain performance by applying a proper randomization technique.</p>
</div>

Lastly, (3) by learning in a corse simulation, we can **drastically limit the number of interactions at test time**, preventing erratic behavior which would endanger system's safety.

<div align="center">
    <img src="https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/machinesafety.png" width="50%">
    <p> Controls applied (BO vs RL). As it samples from an iteratively-refined surrogate model of the objective function, BO explores much more erratically than RL.</p>
</div>

In conclusion, we demonstrate that deep reinforcement learning can master laser pulse shaping by learning **robust policies from raw diagnostics**, paving the way towards **autonomous control of complex physical systems**.

If you're interested in learning more, check out [our latest paper](https://huggingface.co/papers/2503.00499), our [simulator's code](https://github.com/fracapuano/gym-laser), and try out the [live demo](https://huggingface.co/spaces/fracapuano/RLaser).