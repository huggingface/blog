---
title: "TRL v1.0: Post-Training Library That Holds When the Field Invalidates Its Own Assumptions"
thumbnail: /blog/assets/trl-v1/thumbnail.png
authors:
- user: qgallouedec
- user: stevhliu
- user: sergiopaniego
---

# TRL v1.0: Post-Training Library That Holds When the Field Invalidates Its Own Assumptions

We’re releasing TRL v1.0, and it marks a real shift in what TRL is. What started as a research codebase has become a dependable library people build on, with clearer expectations around stability.
This isn't just a version bump. It reflects the reality that TRL now powers production systems, and embraces that responsibility.

TRL now implements [more than 75 post-training methods](https://huggingface.co/docs/trl/en/paper_index). But coverage isn’t the goal by itself. What matters is making these methods easy to try, compare, and actually use in practice.
The design of the library wasn’t decided upfront. It is the result of years of iteration — the first commit goes back more than six years — and it has been shaped by everything the field threw at it: new algorithms, new models, shifting paradigms. Over time, this pressure forced the codebase toward a very specific design. Parts of it might look unusual at first, but like in many evolutionary codebases, they exist for a reason.

TRL is built for a field that doesn’t sit still. So the question is not how to design the perfect abstraction. It is how to make stable software in a domain that keeps invalidating its own assumptions. This is what we tried to solve in TRL v1.0, and this post explains how.

## 1. A moving target: post-training as a shifting field

Post-training has not evolved as a smooth refinement of one recipe. It has moved through successive centers of gravity, each changing not just the objective, but the shape of the stack.

PPO [[Schulman et al., (2017)](https://huggingface.co/papers/1707.06347); [Ziegler et al., (2019)](https://huggingface.co/papers/1909.08593)] made one architecture look canonical: a policy, a reference model, a learned reward model, sampled rollouts, and an RL loop.

Then DPO-style methods such as the original DPO [[Rafailov et al., (2023)](https://huggingface.co/papers/2305.18290)], ORPO [[Hong et al., (2024)](https://huggingface.co/papers/2403.07691)], and KTO [[Ethayarajh et al., (2024)](https://huggingface.co/papers/2402.01306)] cut through that stack: preference optimization could work without a separate reward model, value model, or any online RL. Components that had looked fundamental suddenly looked optional.

RLVR-style methods such as GRPO [[Shao et al., (2024)](https://arxiv.org/abs/2402.03300)] shifted the center again. On tasks like math, code, and tool use, rewards often come from verifiers or deterministic checks rather than learned reward models. Sampling and rollouts matter again, but the objects in the loop are no longer quite the ones PPO libraries were designed around.

The lesson is not just that methods change. The definition of the core keeps changing with them. Strong assumptions here have a short half-life. This is probably why no post-training library is really stable yet.

## 2. From project to library: TRL has a chaos-adaptive design

So what does it mean to build a library for a field that won't sit still? The answer is counterintuitive: don't try to capture the essence of what's stable today. Instead, design around what could change. *Reward models* illustrate why: they looked essential in PPO, became optional in DPO, and came back as *verifiers* in RLVR methods — structures that could be deterministic functions rather than learned models. Any abstraction built around their original form would have been obsolete twice over by now. The library survives by recognizing that strong assumptions have a short life, and by making that changeability central to how the codebase is organized.

This is the environment in which TRL is downloaded 3 million times a month, and in which major downstream projects treat it as stable infrastructure. The field keeps shifting the ground, and at the same time, those users need things not to break.

<div style="margin: 24px 0;">
  <div id="pypi-downloads-chart" style="width: 100%; height: 460px;"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<script>
  const data = [
    {"week_start":"2023-03-20","downloads":"144"},
    {"week_start":"2023-03-27","downloads":"749"},
    {"week_start":"2023-04-03","downloads":"1058"},
    {"week_start":"2023-04-10","downloads":"1030"},
    {"week_start":"2023-04-17","downloads":"1483"},
    {"week_start":"2023-04-24","downloads":"2584"},
    {"week_start":"2023-05-01","downloads":"9082"},
    {"week_start":"2023-05-08","downloads":"829"},
    {"week_start":"2023-05-15","downloads":"1812"},
    {"week_start":"2023-05-22","downloads":"2448"},
    {"week_start":"2023-05-29","downloads":"1703"},
    {"week_start":"2023-06-05","downloads":"2679"},
    {"week_start":"2023-06-12","downloads":"4253"},
    {"week_start":"2023-06-19","downloads":"5344"},
    {"week_start":"2023-06-26","downloads":"6553"},
    {"week_start":"2023-07-03","downloads":"5489"},
    {"week_start":"2023-07-10","downloads":"6488"},
    {"week_start":"2023-07-17","downloads":"11225"},
    {"week_start":"2023-07-24","downloads":"17495"},
    {"week_start":"2023-07-31","downloads":"20384"},
    {"week_start":"2023-08-07","downloads":"22473"},
    {"week_start":"2023-08-14","downloads":"23342"},
    {"week_start":"2023-08-21","downloads":"24067"},
    {"week_start":"2023-08-28","downloads":"29932"},
    {"week_start":"2023-09-04","downloads":"29580"},
    {"week_start":"2023-09-11","downloads":"27647"},
    {"week_start":"2023-09-18","downloads":"30062"},
    {"week_start":"2023-09-25","downloads":"28867"},
    {"week_start":"2023-10-02","downloads":"27925"},
    {"week_start":"2023-10-09","downloads":"30383"},
    {"week_start":"2023-10-16","downloads":"31000"},
    {"week_start":"2023-10-23","downloads":"28933"},
    {"week_start":"2023-10-30","downloads":"30883"},
    {"week_start":"2023-11-06","downloads":"32874"},
    {"week_start":"2023-11-13","downloads":"33528"},
    {"week_start":"2023-11-20","downloads":"35333"},
    {"week_start":"2023-11-27","downloads":"40174"},
    {"week_start":"2023-12-04","downloads":"41673"},
    {"week_start":"2023-12-11","downloads":"42249"},
    {"week_start":"2023-12-18","downloads":"39943"},
    {"week_start":"2023-12-25","downloads":"30761"},
    {"week_start":"2024-01-01","downloads":"38673"},
    {"week_start":"2024-01-08","downloads":"46397"},
    {"week_start":"2024-01-15","downloads":"50178"},
    {"week_start":"2024-01-22","downloads":"52749"},
    {"week_start":"2024-01-29","downloads":"53913"},
    {"week_start":"2024-02-05","downloads":"55359"},
    {"week_start":"2024-02-12","downloads":"67405"},
    {"week_start":"2024-02-19","downloads":"75481"},
    {"week_start":"2024-02-26","downloads":"89834"},
    {"week_start":"2024-03-04","downloads":"82543"},
    {"week_start":"2024-03-11","downloads":"79258"},
    {"week_start":"2024-03-18","downloads":"79006"},
    {"week_start":"2024-03-25","downloads":"87604"},
    {"week_start":"2024-04-01","downloads":"86807"},
    {"week_start":"2024-04-08","downloads":"101484"},
    {"week_start":"2024-04-15","downloads":"104178"},
    {"week_start":"2024-04-22","downloads":"119964"},
    {"week_start":"2024-04-29","downloads":"115426"},
    {"week_start":"2024-05-06","downloads":"118448"},
    {"week_start":"2024-05-13","downloads":"132693"},
    {"week_start":"2024-05-20","downloads":"119712"},
    {"week_start":"2024-05-27","downloads":"107031"},
    {"week_start":"2024-06-03","downloads":"114108"},
    {"week_start":"2024-06-10","downloads":"113461"},
    {"week_start":"2024-06-17","downloads":"122258"},
    {"week_start":"2024-06-24","downloads":"122710"},
    {"week_start":"2024-07-01","downloads":"113872"},
    {"week_start":"2024-07-08","downloads":"130778"},
    {"week_start":"2024-07-15","downloads":"127730"},
    {"week_start":"2024-07-22","downloads":"150557"},
    {"week_start":"2024-07-29","downloads":"211624"},
    {"week_start":"2024-08-05","downloads":"139646"},
    {"week_start":"2024-08-12","downloads":"128588"},
    {"week_start":"2024-08-19","downloads":"132786"},
    {"week_start":"2024-08-26","downloads":"135041"},
    {"week_start":"2024-09-02","downloads":"133137"},
    {"week_start":"2024-09-09","downloads":"156608"},
    {"week_start":"2024-09-16","downloads":"141208"},
    {"week_start":"2024-09-23","downloads":"161715"},
    {"week_start":"2024-09-30","downloads":"144585"},
    {"week_start":"2024-10-07","downloads":"165890"},
    {"week_start":"2024-10-14","downloads":"184607"},
    {"week_start":"2024-10-21","downloads":"178977"},
    {"week_start":"2024-10-28","downloads":"197661"},
    {"week_start":"2024-11-04","downloads":"195731"},
    {"week_start":"2024-11-11","downloads":"209216"},
    {"week_start":"2024-11-18","downloads":"213495"},
    {"week_start":"2024-11-25","downloads":"189209"},
    {"week_start":"2024-12-02","downloads":"194899"},
    {"week_start":"2024-12-09","downloads":"203872"},
    {"week_start":"2024-12-16","downloads":"188361"},
    {"week_start":"2024-12-23","downloads":"158983"},
    {"week_start":"2024-12-30","downloads":"163910"},
    {"week_start":"2025-01-06","downloads":"218041"},
    {"week_start":"2025-01-13","downloads":"234402"},
    {"week_start":"2025-01-20","downloads":"211189"},
    {"week_start":"2025-01-27","downloads":"212962"},
    {"week_start":"2025-02-03","downloads":"245128"},
    {"week_start":"2025-02-10","downloads":"259116"},
    {"week_start":"2025-02-17","downloads":"288098"},
    {"week_start":"2025-02-24","downloads":"272073"},
    {"week_start":"2025-03-03","downloads":"290511"},
    {"week_start":"2025-03-10","downloads":"303941"},
    {"week_start":"2025-03-17","downloads":"298822"},
    {"week_start":"2025-03-24","downloads":"318681"},
    {"week_start":"2025-03-31","downloads":"300384"},
    {"week_start":"2025-04-07","downloads":"322093"},
    {"week_start":"2025-04-14","downloads":"310482"},
    {"week_start":"2025-04-21","downloads":"325206"},
    {"week_start":"2025-04-28","downloads":"342154"},
    {"week_start":"2025-05-05","downloads":"292296"},
    {"week_start":"2025-05-12","downloads":"315810"},
    {"week_start":"2025-05-19","downloads":"310062"},
    {"week_start":"2025-05-26","downloads":"287947"},
    {"week_start":"2025-06-02","downloads":"329981"},
    {"week_start":"2025-06-09","downloads":"289606"},
    {"week_start":"2025-06-16","downloads":"294622"},
    {"week_start":"2025-06-23","downloads":"320598"},
    {"week_start":"2025-06-30","downloads":"337337"},
    {"week_start":"2025-07-07","downloads":"441745"},
    {"week_start":"2025-07-14","downloads":"386394"},
    {"week_start":"2025-07-21","downloads":"464137"},
    {"week_start":"2025-07-28","downloads":"426581"},
    {"week_start":"2025-08-04","downloads":"425303"},
    {"week_start":"2025-08-11","downloads":"395650"},
    {"week_start":"2025-08-18","downloads":"390997"},
    {"week_start":"2025-08-25","downloads":"382834"},
    {"week_start":"2025-09-01","downloads":"372036"},
    {"week_start":"2025-09-08","downloads":"402363"},
    {"week_start":"2025-09-15","downloads":"526905"},
    {"week_start":"2025-09-22","downloads":"465445"},
    {"week_start":"2025-09-29","downloads":"462505"},
    {"week_start":"2025-10-06","downloads":"382975"},
    {"week_start":"2025-10-13","downloads":"414858"},
    {"week_start":"2025-10-20","downloads":"426006"},
    {"week_start":"2025-10-27","downloads":"598674"},
    {"week_start":"2025-11-03","downloads":"618934"},
    {"week_start":"2025-11-10","downloads":"688214"},
    {"week_start":"2025-11-17","downloads":"590929"},
    {"week_start":"2025-11-24","downloads":"478255"},
    {"week_start":"2025-12-01","downloads":"614661"},
    {"week_start":"2025-12-08","downloads":"576386"},
    {"week_start":"2025-12-15","downloads":"556062"},
    {"week_start":"2025-12-22","downloads":"420309"},
    {"week_start":"2025-12-29","downloads":"360290"},
    {"week_start":"2026-01-05","downloads":"492024"},
    {"week_start":"2026-01-12","downloads":"492062"},
    {"week_start":"2026-01-19","downloads":"498126"},
    {"week_start":"2026-01-26","downloads":"481844"},
    {"week_start":"2026-02-02","downloads":"537734"},
    {"week_start":"2026-02-09","downloads":"559373"},
    {"week_start":"2026-02-16","downloads":"545426"},
    {"week_start":"2026-02-23","downloads":"612639"},
    {"week_start":"2026-03-02","downloads":"678756"},
    {"week_start":"2026-03-09","downloads":"674270"},
    {"week_start":"2026-03-16","downloads":"775957"},
  ];

  const sorted = data.slice().sort((a, b) => new Date(a.week_start) - new Date(b.week_start));

  const trace = {
    x: sorted.map(d => d.week_start),
    y: sorted.map(d => parseInt(d.downloads)),
    mode: "lines",
    name: "TRL",
    line: { width: 2.5 }
  };

  Plotly.newPlot("pypi-downloads-chart", [trace], {
    title: "Weekly PyPI downloads",
    hovermode: "x unified",
    paper_bgcolor: "white",
    plot_bgcolor: "white"
  }, { responsive: true, displayModeBar: false });
</script>

### A shift in nature: from code to contract

TRL didn’t make a deliberate decision to become a library. It found out it already was one. Projects like [Unsloth](https://github.com/unslothai/unsloth) and [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) — with thousands of users between them — had built directly on top of TRL’s trainers and APIs. A breaking change in TRL propagated instantly into their stacks. A renamed argument, a shifted default, a restructured output — any of these became someone else’s incident. The shift had already happened. v1.0 is the moment TRL acknowledged it explicitly.

### Stable and experimental, under the same roof

The unusual thing about TRL’s stability model is not what it guarantees, it is what it tolerates alongside those guarantees. Stable and experimental coexist within the same package, with explicitly different contracts. The stable core follows semantic versioning. The experimental layer makes no such promises — it is where new methods land while they are still being evaluated, and where the API can move fast to keep up with the field.

This isn’t a compromise. It’s a response to a specific constraint: the field produces new methods faster than any of them can earn stability. Refusing to add immature methods would make TRL irrelevant within months. Adding them all to stable would break every downstream project every time an algorithm turned out not to work as expected.

```python
from trl import SFTTrainer  # ⚖️ stable
from trl.experimental.orpo import ORPOTrainer  # 🧪 experimental
```

Promotion from experimental to stable isn’t automatic. What matters is the ratio between maintenance cost and actual usage. Some methods earn their place because the community uses them heavily. Others become viable because we can make them cheap enough to maintain — and the design of the codebase is what makes that possible.

In practice, the **stable** surface includes trainers for SFT, DPO, Reward modeling, RLOO, and GRPO, along with their close variants. The **experimental** surface is broader and moves faster; for an up-to-date view, the best reference is the [TRL documentation](https://huggingface.co/docs/trl).

The breaking changes needed to reach v1.0 were distributed deliberately across the 0.x releases. Migration from the last 0.x version is minimal — see the [migration guide](https://github.com/huggingface/trl/blob/main/MIGRATION.md).

### Deliberately limiting abstractions

In a domain where patterns keep changing, the temptation is to build flexible abstractions that can accommodate anything. Our answer was the opposite: **limit abstractions to the strict minimum — while recognizing that this “minimum” is almost always overestimated**.

In practice, this translates into a very local approach to code:

- avoid generic class hierarchies
- favor explicit implementations
- accept, and even encourage, duplication

The goal is not to eliminate structure altogether — shared utilities still exist — but to avoid imposing abstractions where the domain itself is not yet stable. For instance, rather than defining a common base class for offline trainers, we prefer independent implementations when their future evolution is uncertain.

```python
# ❌ No
class OfflineTrainer(Trainer):
    def some_common_method(self): ...

class DPOTrainer(OfflineTrainer): ...

class KTOTrainer(OfflineTrainer): ...

# ✅ Better
class DPOTrainer(Trainer):
    def some_common_method(self): ...

class KTOTrainer(Trainer):
    def some_common_method(self): ...
```

another example:

```python
# ❌ No
# collator.py
class TRLCollator: ...

# dpo_trainer.py
class DPOTrainer:
    def __init__(self, ...):
        self.collator = TRLCollator(...)

# kto_trainer.py
class KTOTrainer:
    def __init__(self, ...):
        self.collator = TRLCollator(...)

# ✅ Better
# dpo_trainer.py
class DataCollatorForPreference: ...

class DPOTrainer:
    def __init__(self, ...):
        self.collator = DataCollatorForPreference(...)

# kto_trainer.py
class DataCollatorForUnpairedPreference: ...

class KTOTrainer:
    def __init__(self, ...):
        self.collator = DataCollatorForUnpairedPreference(...)
```

[Judges](https://github.com/huggingface/trl/blob/main/trl/experimental/judges/judges.py) are a good example of what happens when we don’t follow this principle. Early on, we introduced a `Judge` abstraction to unify the various ways of evaluating model outputs. It looked reasonable at the time. In practice, it was never really used — the abstraction didn’t match how people actually approached evaluation, and it added indirection without adding value. It still lives in the repo, mostly as legacy. In hindsight, shipping the concrete implementations without the unifying abstraction would have served users better.

### More explicit, but more adaptable

This approach favors explicit and modifiable usage over rigid frameworks: less magic, but more control. It comes with an obvious cost: code duplication. While often seen as an anti-pattern, in this context it has proven not only acceptable, but effective. Contrary to intuition, it remains manageable in practice with a small but consistent discipline: keeping deltas between implementations minimal and avoiding unnecessary divergence. Like in the [Transformers design philosophy](https://huggingface.co/blog/transformers-design-philosophy#3-machine-learning-is-evolving-at-a-neck-breaking-speed), we accept duplication and local explicitness by design. The motivations largely coincide, with some nuance in focus.

This is easier to see than to describe. Compare RLOO and GRPO: large parts of their implementation are duplicated almost line for line. That is not accidental, and it is not dead weight. These methods are close enough that keeping their code paths aligned makes them easier to read, easier to evolve, and cheaper to maintain.

<div style="padding-bottom: 20px;">
  <div id="trl_diff" style="width: 100%; height: 400px; border: 1px solid grey;"></div>
</div>

<script src="https://unpkg.com/monaco-editor@latest/min/vs/loader.js"></script>
<script>
  const proxy = URL.createObjectURL(
    new Blob(
      [`
          self.MonacoEnvironment = {
            baseUrl: 'https://unpkg.com/monaco-editor@latest/min/'
          };
          importScripts('https://unpkg.com/monaco-editor@latest/min/vs/base/worker/workerMain.js');
        `],
      { type: "text/javascript" }
    )
  );

  window.MonacoEnvironment = { getWorkerUrl: () => proxy };

  require.config({
    paths: { vs: "https://unpkg.com/monaco-editor@latest/min/vs" }
  });

  require(["vs/editor/editor.main"], function () {
    const diffEditor = monaco.editor.createDiffEditor(
      document.getElementById("trl_diff"),
      {
        readOnly: true,
        automaticLayout: true
      }
    );

    Promise.all([
      fetch("https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_trainer.py").then(r => r.text()),
      fetch("https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/rloo_trainer.py").then(r => r.text())
    ]).then(([originalTxt, modifiedTxt]) => {
      diffEditor.setModel({
        original: monaco.editor.createModel(originalTxt, "python"),
        modified: monaco.editor.createModel(modifiedTxt, "python")
      });
    });
  });
</script>

## 3. Where TRL fits

The goal of this comparison is not to argue that TRL should be judged as best on every axis. It should not. Some systems are built for maximum throughput (like [PipelineRL](https://github.com/ServiceNow/PipelineRL)), some are optimized for a narrower slice of the problem (like [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)), and some offer a more opinionated development experience in a specific environment (like [Tinker](https://github.com/thinking-machines-lab/tinker)). TRL occupies a different place in the ecosystem: it is a general-purpose post-training library that tries to keep the API and the code as simple as the domain allows, while combining broad method coverage, deep Hugging Face integration, a relatively low infrastructure burden, and an explicit stability contract.

Libraries like [Unsloth](https://github.com/unslothai/unsloth) and [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) are not included here because they build on top of TRL rather than sitting beside it in the comparison; in that sense, many of their users are also TRL users indirectly.

### Ecosystem

|  | [TRL](https://github.com/huggingface/trl) | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | [veRL](https://github.com/volcengine/verl) | [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) | [PipelineRL](https://github.com/ServiceNow/PipelineRL) | [OAT](https://github.com/sail-sg/oat) | [Tinker](https://github.com/thinking-machines-lab/tinker) | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [torchtune](https://github.com/meta-pytorch/torchtune) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hugging Face Hub integration | 🟢 Full | 🟡 No push | 🟡 Model loading only | 🟡 No push | 🟡 No push | 🟡 No push | 🟡 No dataset loading | 🟢 Full | 🟡 Dataset loading only |
| PEFT / LoRA / QLoRA support | 🟢 LoRA + QLoRA | 🟢 LoRA + QLoRA | 🟡 LoRA only (QAT instead of QLoRA) | 🟡 LoRA only | 🔴 Not supported | 🟢 LoRA + QLoRA | 🟡 LoRA only | 🟢 LoRA + QLoRA | 🟢 LoRA + QLoRA (torchao, not bitsandbytes) |
| Experiment tracker flexibility | 🟢 Any (via `report_to`) | 🟡 wandb + tensorboard | 🟢 wandb, mlflow, swanlab, tensorboard | 🔴 wandb only | 🔴 wandb only | 🔴 wandb only | 🟡 DIY (metrics returned via API, no built-in tracker) | 🟢 Any (via `report_to`) + swanlab | 🟡 wandb + tensorboard (manual config) |
| Infrastructure burden | 🟢 Low (single GPU, standard stack) | 🟠 High (Ray required) | 🔴 Very high (Ray + rollout engine) | 🟠 High (separate vLLM server + ZMQ) | 🟠 High (async vLLM pipeline) | 🟡 Medium (vLLM needed for RL) | 🟢 Low (managed cloud service) | 🟢 Low (single script) | 🟢 Low (single script) |

### Training methods

|  | [TRL](https://github.com/huggingface/trl) | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | [veRL](https://github.com/volcengine/verl) | [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) | [PipelineRL](https://github.com/ServiceNow/PipelineRL) | [OAT](https://github.com/sail-sg/oat) | [Tinker](https://github.com/thinking-machines-lab/tinker) | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [torchtune](https://github.com/meta-pytorch/torchtune) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VLM support | 🟢 Yes (SFT, DPO, GRPO) | 🔴 No | 🟢 Yes (in SFT + PPO trainers) | 🟡 Partial (Qwen3-VL only) | 🟢 Yes (via processor_factory) | 🔴 No | 🔴 No | 🟢 Yes (via mm_plugin) | 🟡 Partial (Llama Vision only) |
| Supervised post-training | 🟢 Yes (via SFTTrainer) | 🟢 Yes (via SFTTrainer) | 🟢 Yes (via SFTTrainer) | 🟢 Yes (via SFT entrypoint) | 🔴 No | 🟢 Yes (via SFTLearner) | 🔴 No (low-level primitives only) | 🟢 Yes (via SFT trainer) | 🟢 Yes (via finetune recipes) |
| Distillation post-training | 🟢 Yes (GKD, SDFT, SDPO) | 🔴 No | 🟢 Yes (dedicated distillation trainer) | 🟢 Yes (on-policy distillation) | 🔴 No | 🔴 No | 🔴 No (low-level primitives only) | 🔴 No | 🟢 Yes (native KD recipes) |
| Preference post-training | 🟢 Yes (DPO, KTO, ORPO, CPO, SimPO, IPO, …) | 🟡 DPO only | 🔴 No | 🔴 No | 🔴 No | 🟢 Yes (DPO, SimPO, IPO, XPO) | 🔴 No (low-level primitives only) | 🟡 DPO, KTO, ORPO (via TRL) | 🟡 DPO only |
| RL post-training | 🟢 Yes (PPO, GRPO, RLOO, …) | 🟢 Yes (PPO, REINFORCE++, GRPO, RLOO) | 🟢 Yes (PPO, GRPO, RLOO, REINFORCE++, DAPO, PRIME, …) | 🟢 Yes (async GRPO-style) | 🟢 Yes (GRPO, async) | 🟢 Yes (PPO, GRPO, Online DPO) | 🔴 No (low-level primitives only) | 🟠 PPO only (via TRL) | 🟠 PPO (GRPO in development) |
| Agent / environment support | 🟢 Yes (flexible `environment_factory` in GRPO) | 🟢 Yes (flexible `AgentInstance` interface) | 🟢 Yes (flexible `BaseInteraction` interface) | 🟡 Partial (tied to Prime Intellect’s Environments Hub) | 🟡 Partial (built-in domains: fn_calling, miniwob, …) | 🔴 No | 🔴 No (low-level primitives only) | 🔴 No | 🔴 No |

### Project health

|  | [TRL](https://github.com/huggingface/trl) | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | [veRL](https://github.com/volcengine/verl) | [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) | [PipelineRL](https://github.com/ServiceNow/PipelineRL) | [OAT](https://github.com/sail-sg/oat) | [Tinker](https://github.com/thinking-machines-lab/tinker) | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [torchtune](https://github.com/meta-pytorch/torchtune) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semver stability | 🟢 Yes | 🟡 Partly | 🟡 Partly | 🔴 No | 🔴 No | 🔴 No | 🟡 Partly | 🟡 Partly | 🟡 Partly |
| PyPI downloads / month | 3.0M | 3.6k | 101.6k | N/A | N/A | 218 | 363.3k | 25.6k | 370.9k |
| GitHub stars | 17.8k | 9.2k | 20.2k | 1.2k | 0.4k | 0.6k | 3.0k | 69.0k | 5.7k |
| Last release | 🟢 today | 🟢 yesterday | 🟡 1 week ago | 🟠 7 weeks ago | ⚫ No release | 🔴 3 months ago | ⚫ No release | 🔴 3 months ago | 🔴 11 months ago |
| Last commit | 🟢 yesterday | 🟢 yesterday | 🟢 today | 🟢 today | 🟡 3 weeks ago | 🟠 8 weeks ago | 🟢 yesterday | 🟢 yesterday | 🟡 1 week ago |

Some rows are factual (`GitHub stars`, `Last release`, `Last commit`), others are qualitative judgments (`Semver stability`).

Taken together, these comparisons point to a clear role for TRL: a general-purpose library designed to keep breadth, simplicity, integration, and stability in the same place. Its full downstream footprint is hard to measure, since most deployments are private and reverse dependencies are largely invisible, but the available signals already show that TRL operates at a distinctly different scale.

## 4. What’s next

By now, the logic of v1.0 should be clear: it is not a claim that post-training has stabilized. On the contrary, it is an acknowledgment that the field will keep shifting, and that we're confident that the library has the right shape to absorb whatever comes next. The question is not what comes **after** v1.0, but what’s next **for** v1.0.

### Asyncronous GRPO

Today, GRPO in TRL is primarily used through a synchronous loop: generate rollouts, score them, then step the optimizer. That shape is simple and dependable, but it ties throughput to the slowest stage and leaves performance on the table at scale.

![async-grpo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/trl-v1/async-grpo.png)

We already have an [early asynchronous GRPO design](https://huggingface.co/docs/trl/main/en/async_grpo_trainer), and the next step is to harden it. The core idea is to decouple generation and training, letting generation run continuously on dedicated inference resources while training consumes a steady stream of scored trajectories, with buffering, backpressure, and clear policy-version accounting. This improves utilization and scales across GPUs and nodes. Other libraries already offer forms of asynchronous RL, but bringing it to TRL would make this style of training available through broader integrations, simpler APIs, and a much lower barrier to adoption.

### Graduating methods to stable

The next candidates include [KTO](https://huggingface.co/docs/trl/main/en/kto_trainer) and newer distillation trainers such as [SDFT](https://huggingface.co/docs/trl/main/en/sdft_trainer), [SDPO](https://huggingface.co/docs/trl/main/en/sdpo_trainer), and possibly [GOLD](https://huggingface.co/docs/trl/main/en/gold_trainer) or [GKD](https://huggingface.co/docs/trl/main/en/gkd_trainer). As discussed in Section 2, before moving them to stable, the goal is to minimize code differences across implementations and monitor sustained community interest relative to maintenance cost.

### Scaling

TRL supports large-scale training, including multi-node runs and larger models; the next step is to make this path significantly more robust and easier to operate in production. That includes stronger guarantees around distributed stability, clearer scaling defaults, and deeper support for Mixture-of-Experts (MoE), especially expert parallelism, where routing, load balancing, and memory behavior become critical.

### Making training legible to agents

Training is still too often driven by vibes. Loss curves go down, reward curves go up, a few samples look better than before, and people convince themselves the run is working. When it fails, they scroll through logs, compare runs by eye, and guess. That is already a weak interface for humans. For agents, it is worse: it is barely an interface at all.

One of the most important directions for TRL is to make training legible to software, not just to people. That means going beyond dashboards and raw metrics to produce explicit signals: is the policy improving, collapsing, over-optimizing the verifier, drifting off-distribution, or plateauing? The goal is for TRL to surface these patterns automatically, explain them clearly, and turn them into actions.

If we get this right, the payoff is bigger than convenience. Beginners get guardrails instead of folklore. Advanced users get faster diagnosis and tighter iteration loops. And agents get something new entirely: a training stack they can inspect, reason about, and actively steer. That may end up being one of the most important upgrades in TRL v1.0: not just helping people run training, but making training interpretable enough to automate.

## 5. Conclusion

Post-training doesn't converge. It shifts, and the next shift is already coming.

v1.0 is not a claim that things have settled. It's an acknowledgment that they haven't yet, and a commitment that the library can be relied on anyway. Six years of evolving alongside the field shaped a design we're confident is ready for what comes next, whatever that turns out to be. The community and the downstream projects had already assumed that stability — v1.0 makes it real.

```bash
pip install --upgrade trl
```

Migration from the last 0.x release is minimal, and the [migration guide](https://github.com/huggingface/trl/blob/main/MIGRATION.md) covers everything. If you're new, now is a good time to start.
