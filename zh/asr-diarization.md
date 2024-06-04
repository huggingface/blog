---
title: "使用 Hugging Face 推理终端搭建强大的“语音识别 + 说话人分割 + 投机解码”工作流" 
thumbnail: /blog/assets/asr-diarization/thumbnail.png
authors:
- user: sergeipetrov
- user: reach-vb
- user: pcuenq
- user: philschmid
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 使用 Hugging Face 推理终端搭建强大的“语音识别 + 说话人分割 + 投机解码”工作流

Whisper 是当前最先进的开源语音识别模型之一，毫无疑问，也是应用最广泛的模型。如果你想部署 Whisper 模型，Hugging Face [推理终端](https://huggingface.co/inference-endpoints/dedicated) 能够让你开箱即用地轻松部署任何 Whisper 模型。但是，如果你还想叠加其它功能，如用于分辨不同说话人的说话人分割，或用于投机解码的辅助生成，事情就有点麻烦了。因为此时你需要将 Whisper 和其他模型结合起来，但对外仍只发布一个 API。

本文，我们将使用推理终端的 [自定义回调函数](https://huggingface.co/docs/inference-endpoints/guides/custom_handler) 来解决这一挑战，将其它把自动语音识别 (ASR) 、说话人分割流水线以及投机解码串联起来并嵌入推理端点。这一设计主要受 [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper#insanely-fast-whisper) 的启发，其使用了 [Pyannote](https://github.com/pyannote/pyannote-audio) 说话人分割模型。

我们也希望能通过这个例子展现出推理终端的灵活性以及其“万物皆可托管”的无限可能性。你可在 [此处](https://huggingface.co/sergeipetrov/asrdiarization-handler/) 找到我们的自定义回调函数的完整代码。请注意，终端在初始化时会安装整个代码库，因此如果你不喜欢将所有逻辑放在单个文件中的话，可以采用 `handler.py` 作为入口并调用代码库中的其他文件的方法。为清晰起见，本例分为以下几个文件:

- `handler.py` : 包含初始化和推理代码
- `diarization_utils.py` : 含所有说话人分割所需的预处理和后处理方法
- `config.py` : 含 `ModelSettings` 和 `InferenceConfig` 。其中，`ModelSettings` 定义流水线中用到的模型 (可配，无须使用所有模型)，而 `InferenceConfig` 定义默认的推理参数

**_从 [PyTorch 2.2](https://pytorch.org/blog/pytorch2-2/) 开始，SDPA 开箱即用支持 Flash Attention 2，因此本例使用 PyTorch 2.2 以加速推理。_**

## 主要模块

下图展示了我们设计的方案的系统框图:

![系统框图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/asr-diarization/pipeline_schema.png)

在实现时，ASR 和说话人分割流水线采用了模块化的方法，因此是可重用的。说话人分割流水线是基于 ASR 的输出的，如果不需要说话人分割，则可以仅用 ASR 的部分。我们建议使用 [Pyannote 模型](https://huggingface.co/pyannote/speaker-diarization-3.1) 做说话人分割，该模型目前是开源模型中的 SOTA。

我们还使用了投机解码以加速模型推理。投机解码通过使用更小、更快的模型来打草稿，再由更大的模型来验证，从而实现加速。具体请参阅 [这篇精彩的博文](https://huggingface.co/blog/whisper-speculative-decoding) 以详细了解如何对 Whisper 模型使用投机解码。

投机解码有如下两个限制:

- 辅助模型和主模型的解码器的架构应相同
- 在很多实现中，batch size 须为 1

在评估是否使用投机解码时，请务必考虑上述因素。根据实际用例不同，有可能支持较大 batch size 带来的收益比投机解码更大。如果你不想使用辅助模型，只需将配置中的 `assistant_model` 置为 `None` 即可。

如果你决定使用辅助模型，[distil-whisper](https://huggingface.co/distil-whisper) 是一个不错的 Whisper 辅助模型候选。

## 创建一个自己的终端

上手很简单，用 [代码库拷贝神器](https://huggingface.co/spaces/huggingface-projects/repo_duplicator) 拷贝一个现有的带 [自定义回调函数](https://huggingface.co/sergeipetrov/asrdiarization-handler/blob/main/handler.py) 的代码库。

以下是其 `handler.py` 中的模型加载部分:

```python
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM

...

self.asr_pipeline = pipeline(
      "automatic-speech-recognition",
      model=model_settings.asr_model,
      torch_dtype=torch_dtype,
      device=device
  )

  self.assistant_model = AutoModelForCausalLM.from_pretrained(
      model_settings.assistant_model,
      torch_dtype=torch_dtype,
      low_cpu_mem_usage=True,
      use_safetensors=True
  )
  
  ...

  self.diarization_pipeline = Pipeline.from_pretrained(
      checkpoint_path=model_settings.diarization_model,
      use_auth_token=model_settings.hf_token,
  )
  
  ...
```

然后，你可以根据需要定制流水线。 `config.py` 文件中的 `ModelSettings` 包含了流水线的初始化参数，并定义了推理期间要使用的模型:

```python
class ModelSettings(BaseSettings):
    asr_model: str
    assistant_model: Optional[str] = None
    diarization_model: Optional[str] = None
    hf_token: Optional[str] = None
```

如果你用的是自定义容器或是自定义推理回调函数的话，你还可以通过设置相应的环境变量来调整参数，你可通过 [Pydantic](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) 来达成此目的。要在构建期间将环境变量传入容器，你须通过 API 调用 (而不是通过 GUI) 创建终端。

你还可以在代码中硬编码模型名，而不将其作为环境变量传入，但 _请注意，说话人分割流水线需要显式地传入 HF 令牌 (`hf_token` )。_ 出于安全考量，我们不允许对令牌进行硬编码，这意味着你必须通过 API 调用创建终端才能使用说话人分割模型。

提醒一下，所有与说话人分割相关的预处理和后处理工具程序都在 `diarization_utils.py` 中。

该方案中，唯一必选的组件是 ASR 模型。可选项是: 1) 投机解码，你可指定一个辅助模型用于此; 2) 说话人分割模型，可用于对转录文本按说话人进行分割。

### 部署至推理终端

如果仅需 ASR 组件，你可以在 `config.py` 中指定 `asr_model` 和/或 `assistant_model` ，并单击按钮直接部署:

![一键部署](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/asr-diarization/deploy_oneclick.png)

如要使用环境变量来配置推理终端托管的容器，你需要用 [API](https://api.endpoints.huggingface.cloud/#post-/v2/endpoint/-namespace-) 以编程方式创建终端。下面给出了一个示例:

```python
body = {
    "compute": {
        "accelerator": "gpu",
        "instanceSize": "medium",
        "instanceType": "g5.2xlarge",
        "scaling": {
            "maxReplica": 1,
            "minReplica": 0
        }
    },
    "model": {
        "framework": "pytorch",
        "image": {
            # a default container
            "huggingface": {
                "env": {
		    # this is where a Hub model gets mounted
                    "HF_MODEL_DIR": "/repository",
                    "DIARIZATION_MODEL": "pyannote/speaker-diarization-3.1",
                    "HF_TOKEN": "<your_token>",
                    "ASR_MODEL": "openai/whisper-large-v3",
                    "ASSISTANT_MODEL": "distil-whisper/distil-large-v3"
                }
            }
        },
        # a model repository on the Hub
        "repository": "sergeipetrov/asrdiarization-handler",
        "task": "custom"
    },
    # the endpoint name
    "name": "asr-diarization-1",
    "provider": {
        "region": "us-east-1",
        "vendor": "aws"
    },
    "type": "private"
}
```

### 何时使用辅助模型

为了更好地了解辅助模型的收益情况，我们使用 [k6](https://k6.io/docs/) 进行了一系列基准测试，如下:

```bash
# 设置:
# GPU: A10
ASR_MODEL=openai/whisper-large-v3
ASSISTANT_MODEL=distil-whisper/distil-large-v3

# 长音频: 60s; 短音频: 8s
长音频 _ 投机解码 ..................: avg=4.15s min=3.84s med=3.95s max=6.88s p(90)=4.03s p(95)=4.89s
长音频 _ 直接解码 ..............: avg=3.48s min=3.42s med=3.46s max=3.71s p(90)=3.56s p(95)=3.61s
短音频 _ 辅助解码 .................: avg=326.96ms min=313.01ms med=319.41ms max=960.75ms p(90)=325.55ms p(95)=326.07ms
短音频 _ 直接解码 .............: avg=784.35ms min=736.55ms med=747.67ms max=2s p(90)=772.9ms p(95)=774.1ms
```

如你所见，当音频较短 (batch size 为 1) 时，辅助生成能带来显著的性能提升。如果音频很长，推理系统会自动将其切成多 batch，此时由于上文述及的限制，投机解码可能会拖慢推理。

### 推理参数

所有推理参数都在 `config.py` 中:

```python
class InferenceConfig(BaseModel):
    task: Literal["transcribe", "translate"] = "transcribe"
    batch_size: int = 24
    assisted: bool = False
    chunk_length_s: int = 30
    sampling_rate: int = 16000
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
```

当然，你可根据需要添加或删除参数。与说话者数量相关的参数是给说话人分割流水线的，其他所有参数主要用于 ASR 流水线。 `sampling_rate` 表示要处理的音频的采样率，用于预处理环节; `assisted` 标志告诉流水线是否使用投机解码。请记住，辅助生成的 `batch_size` 必须设置为 1。

### 请求格式

服务一旦部署，用户就可将音频与推理参数一起组成请求包发送至推理终端，如下所示 (Python):

```python
import base64
import requests

API_URL = "<your endpoint URL>"
filepath = "/path/to/audio"

with open(filepath, "rb") as f:
    audio_encoded = base64.b64encode(f.read()).decode("utf-8")

data = {
    "inputs": audio_encoded,
    "parameters": {
        "batch_size": 24
    }
}

resp = requests.post(API_URL, json=data, headers={"Authorization": "Bearer <your token>"})
print(resp.json())
```

这里的 **“parameters”** 字段是一个字典，其中包含你想调整的所有 `InferenceConfig` 参数。请注意，我们会忽略 `InferenceConfig` 中没有的参数。

你还可以使用 [InferenceClient](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#huggingface_hub.InferenceClient) 类，或其 [异步版](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient) 来发送请求:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model = "<your endpoint URL>", token="<your token>")

with open("/path/to/audio", "rb") as f:
    audio_encoded = base64.b64encode(f.read()).decode("utf-8")
data = {
    "inputs": audio_encoded,
    "parameters": {
        "batch_size": 24
    }
}

res = client.post(json=data)
```

## 总结

本文讨论了如何使用 Hugging Face 推理终端搭建模块化的 “ASR + 说话人分割 + 投机解码”工作流。该方案使用了模块化的设计，使用户可以根据需要轻松配置并调整流水线，并轻松地将其部署至推理终端！更幸运的是，我们能够基于社区提供的优秀公开模型及工具实现我们的方案:

- OpenAI 的一系列 [Whisper](https://huggingface.co/openai/whisper-large-v3) 模型
- Pyannote 的 [说话人分割模型](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Insanely Fast Whisper 代码库](https://github.com/Vaibhavs10/insanely-fast-whisper/tree/main)，这是本文的主要灵感来源

本文相关的代码已上传至 [这个代码库中](https://github.com/plaggy/fast-whisper-server)，其中包含了本文论及的流水线及其服务端代码 (FastAPI + Uvicorn)。如果你想根据本文的方案进一步进行定制或将其托管到其他地方，这个代码库可能会派上用场。