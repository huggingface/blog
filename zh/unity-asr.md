---
title: "如何在 Unity 游戏中集成 AI 语音识别？"
thumbnail: /blog/assets/124_ml-for-games/unity-asr-thumbnail.png
authors:
- user: dylanebert
translators:
- user: SuSung-boy
- user: zhongdongy
  proofreader: true
---

# 如何在 Unity 游戏中集成 AI 语音识别？


![Open Source AI Game Jam](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gamejambanner.png)
[](https://itch.io/jam/open-source-ai-game-jam)

## 简介

语音识别是一项将语音转换为文本的技术，想象一下它如何在游戏中发挥作用？发出命令操纵控制面板或者游戏角色、直接与 NPC 对话、提升交互性等等，都有可能。本文将介绍如何使用 Hugging Face Unity API 在 Unity 游戏中集成 SOTA 语音识别功能。

您可以访问 [itch.io 网站](https://individualkex.itch.io/speech-recognition-demo) 下载 Unity 游戏样例，亲自尝试一下语音识别功能。

### 先决条件

阅读文本可能需要了解一些 Unity 的基本概念。除此之外，您还需安装 [Hugging Face Unity API](https://github.com/huggingface/unity-api)，可以点击 [之前的博文](https://huggingface.co/blog/zh/unity-api) 阅读 API 安装说明。

## 步骤

### 1. 设置场景

在本教程中，我们将设置一个非常简单的场景。玩家可以点击按钮来开始或停止录制语音，识别音频并转换为文本。

首先我们新建一个 Unity 项目，然后创建一个包含三个 UI 组件的画布 (Canvas):

1. **开始按钮**: 按下以开始录制语音。
2. **停止按钮**: 按下以停止录制语音。
3. **文本组件 (TextMeshPro)**: 显示语音识别结果文本的地方。

### 2. 创建脚本

创建一个名为 `SpeechRecognitionTest` 的脚本，并将其附加到一个空的游戏对象 (GameObject) 上。

在脚本中，首先定义对 UI 组件的引用:

```
[SerializeField] private Button startButton;
[SerializeField] private Button stopButton;
[SerializeField] private TextMeshProUGUI text;
```

在 inspector 窗口中分配对应组件。

然后，使用 `Start()` 方法为开始和停止按钮设置监听器:

```
private void Start() {
    startButton.onClick.AddListener(StartRecording);
    stopButton.onClick.AddListener(StopRecording);
}
```

此时，脚本中的代码应该如下所示:

```
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class SpeechRecognitionTest : MonoBehaviour {
    [SerializeField] private Button startButton;
    [SerializeField] private Button stopButton;
    [SerializeField] private TextMeshProUGUI text;

    private void Start() {
        startButton.onClick.AddListener(StartRecording);
        stopButton.onClick.AddListener(StopRecording);
    }

    private void StartRecording() {

    }

    private void StopRecording() {

    }
}
```

### 3. 录制麦克风语音输入

现在，我们来录制麦克风语音输入，并将其编码为 WAV 格式。这里需要先定义成员变量:

```
private AudioClip clip;
private byte[] bytes;
private bool recording;
```

然后，在 `StartRecording()` 中，使用 `Microphone.Start()` 方法实现开始录制语音的功能:

```
private void StartRecording() {
    clip = Microphone.Start(null, false, 10, 44100);
    recording = true;
}
```

上面代码实现以 44100 Hz 录制最长为 10 秒的音频。

当录音时长达到 10 秒的最大限制，我们希望录音行为自动停止。为此，需要在 `Update()` 方法中写上以下内容:

```
private void Update() {
    if (recording && Microphone.GetPosition(null) >= clip.samples) {
        StopRecording();
    }
}
```

接着，在 `StopRecording()` 中，截取录音片段并将其编码为 WAV 格式:

```
private void StopRecording() {
    var position = Microphone.GetPosition(null);
    Microphone.End(null);
    var samples = new float[position * clip.channels];
    clip.GetData(samples, 0);
    bytes = EncodeAsWAV(samples, clip.frequency, clip.channels);
    recording = false;
}
```

最后，我们需要实现音频编码的 `EncodeAsWAV()` 方法，这里直接使用 Hugging Face API，只需要将音频数据准备好即可:

```
private byte[] EncodeAsWAV(float[] samples, int frequency, int channels) {
    using (var memoryStream = new MemoryStream(44 + samples.Length * 2)) {
        using (var writer = new BinaryWriter(memoryStream)) {
            writer.Write("RIFF".ToCharArray());
            writer.Write(36 + samples.Length * 2);
            writer.Write("WAVE".ToCharArray());
            writer.Write("fmt ".ToCharArray());
            writer.Write(16);
            writer.Write((ushort)1);
            writer.Write((ushort)channels);
            writer.Write(frequency);
            writer.Write(frequency * channels * 2);
            writer.Write((ushort)(channels * 2));
            writer.Write((ushort)16);
            writer.Write("data".ToCharArray());
            writer.Write(samples.Length * 2);

            foreach (var sample in samples) {
                writer.Write((short)(sample * short.MaxValue));
            }
        }
        return memoryStream.ToArray();
    }
}
```

完整的脚本如下所示:

```
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class SpeechRecognitionTest : MonoBehaviour {
    [SerializeField] private Button startButton;
    [SerializeField] private Button stopButton;
    [SerializeField] private TextMeshProUGUI text;

    private AudioClip clip;
    private byte[] bytes;
    private bool recording;

    private void Start() {
        startButton.onClick.AddListener(StartRecording);
        stopButton.onClick.AddListener(StopRecording);
    }

    private void Update() {
        if (recording && Microphone.GetPosition(null) >= clip.samples) {
            StopRecording();
        }
    }

    private void StartRecording() {
        clip = Microphone.Start(null, false, 10, 44100);
        recording = true;
    }

    private void StopRecording() {
        var position = Microphone.GetPosition(null);
        Microphone.End(null);
        var samples = new float[position * clip.channels];
        clip.GetData(samples, 0);
        bytes = EncodeAsWAV(samples, clip.frequency, clip.channels);
        recording = false;
    }

    private byte[] EncodeAsWAV(float[] samples, int frequency, int channels) {
        using (var memoryStream = new MemoryStream(44 + samples.Length * 2)) {
            using (var writer = new BinaryWriter(memoryStream)) {
                writer.Write("RIFF".ToCharArray());
                writer.Write(36 + samples.Length * 2);
                writer.Write("WAVE".ToCharArray());
                writer.Write("fmt ".ToCharArray());
                writer.Write(16);
                writer.Write((ushort)1);
                writer.Write((ushort)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2);
                writer.Write((ushort)(channels * 2));
                writer.Write((ushort)16);
                writer.Write("data".ToCharArray());
                writer.Write(samples.Length * 2);

                foreach (var sample in samples) {
                    writer.Write((short)(sample * short.MaxValue));
                }
            }
            return memoryStream.ToArray();
        }
    }
}
```

如要测试该脚本代码是否正常运行，您可以在 `StopRecording()` 方法末尾添加以下代码:

```
File.WriteAllBytes(Application.dataPath + "/test.wav", bytes);
```

好了，现在您点击 `Start` 按钮，然后对着麦克风说话，接着点击 `Stop` 按钮，您录制的音频将会保存为 `test.wav` 文件，位于工程目录的 Unity 资产文件夹中。

### 4. 语音识别

接下来，我们将使用 Hugging Face Unity API 对编码音频实现语音识别。为此，我们创建一个 `SendRecording()` 方法:

```
using HuggingFace.API;

private void SendRecording() {
    HuggingFaceAPI.AutomaticSpeechRecognition(bytes, response => {
        text.color = Color.white;
        text.text = response;
    }, error => {
        text.color = Color.red;
        text.text = error;
    });
}
```

该方法实现将编码音频发送到语音识别 API，如果发送成功则以白色显示响应，否则以红色显示错误消息。

别忘了在 `StopRecording()` 方法的末尾调用 `SendRecording()`:

```
private void StopRecording() {
    /* other code */
    SendRecording();
}
```

### 5. 最后润色

最后来提升一下用户体验，这里我们使用交互性按钮和状态消息。

开始和停止按钮应该仅在适当的时候才产生交互效果，比如: 准备录制、正在录制、停止录制。

在录制语音或等待 API 返回识别结果时，我们可以设置一个简单的响应文本来显示对应的状态信息。

完整的脚本如下所示:

```
using System.IO;
using HuggingFace.API;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class SpeechRecognitionTest : MonoBehaviour {
    [SerializeField] private Button startButton;
    [SerializeField] private Button stopButton;
    [SerializeField] private TextMeshProUGUI text;

    private AudioClip clip;
    private byte[] bytes;
    private bool recording;

    private void Start() {
        startButton.onClick.AddListener(StartRecording);
        stopButton.onClick.AddListener(StopRecording);
        stopButton.interactable = false;
    }

    private void Update() {
        if (recording && Microphone.GetPosition(null) >= clip.samples) {
            StopRecording();
        }
    }

    private void StartRecording() {
        text.color = Color.white;
        text.text = "Recording...";
        startButton.interactable = false;
        stopButton.interactable = true;
        clip = Microphone.Start(null, false, 10, 44100);
        recording = true;
    }

    private void StopRecording() {
        var position = Microphone.GetPosition(null);
        Microphone.End(null);
        var samples = new float[position * clip.channels];
        clip.GetData(samples, 0);
        bytes = EncodeAsWAV(samples, clip.frequency, clip.channels);
        recording = false;
        SendRecording();
    }

    private void SendRecording() {
        text.color = Color.yellow;
        text.text = "Sending...";
        stopButton.interactable = false;
        HuggingFaceAPI.AutomaticSpeechRecognition(bytes, response => {
            text.color = Color.white;
            text.text = response;
            startButton.interactable = true;
        }, error => {
            text.color = Color.red;
            text.text = error;
            startButton.interactable = true;
        });
    }

    private byte[] EncodeAsWAV(float[] samples, int frequency, int channels) {
        using (var memoryStream = new MemoryStream(44 + samples.Length * 2)) {
            using (var writer = new BinaryWriter(memoryStream)) {
                writer.Write("RIFF".ToCharArray());
                writer.Write(36 + samples.Length * 2);
                writer.Write("WAVE".ToCharArray());
                writer.Write("fmt ".ToCharArray());
                writer.Write(16);
                writer.Write((ushort)1);
                writer.Write((ushort)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2);
                writer.Write((ushort)(channels * 2));
                writer.Write((ushort)16);
                writer.Write("data".ToCharArray());
                writer.Write(samples.Length * 2);

                foreach (var sample in samples) {
                    writer.Write((short)(sample * short.MaxValue));
                }
            }
            return memoryStream.ToArray();
        }
    }
}
```

祝贺！现在您可以在 Unity 游戏中集成 SOTA 语音识别功能了！

如果您有任何疑问，或想更多地参与 Hugging Face for Games 系列，可以加入 [Hugging Face Discord](https://hf.co/join/discord) 频道！