---
title: "AI Speech Recognition in Unity"
thumbnail: /blog/assets/124_ml-for-games/unity-asr-thumbnail.png
authors:
- user: dylanebert
---

# AI Speech Recognition in Unity


[![Open Source AI Game Jam](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/gamejambanner.png)](https://itch.io/jam/open-source-ai-game-jam)

## Introduction

This tutorial guides you through the process of implementing state-of-the-art Speech Recognition in your Unity game using the Hugging Face Unity API. This feature can be used for giving commands, speaking to an NPC, improving accessibility, or any other functionality where converting spoken words to text may be useful.

To try Speech Recognition in Unity for yourself, check out the [live demo in itch.io](https://individualkex.itch.io/speech-recognition-demo).

### Prerequisites

This tutorial assumes basic knowledge of Unity. It also requires you to have installed the [Hugging Face Unity API](https://github.com/huggingface/unity-api). For instructions on setting up the API, check out our [earlier blog post](https://huggingface.co/blog/unity-api).

## Steps

### 1. Set up the Scene

In this tutorial, we'll set up a very simple scene where the player can start and stop a recording, and the result will be converted to text.

Begin by creating a Unity project, then creating a Canvas with four UI elements:

1. **Start Button**: This will start the recording.
2. **Stop Button**: This will stop the recording.
3. **Text (TextMeshPro)**: This is where the result of the speech recognition will be displayed.

### 2. Set up the Script

Create a script called `SpeechRecognitionTest` and attach it to an empty GameObject.

In the script, define references to your UI components:
```
[SerializeField] private Button startButton;
[SerializeField] private Button stopButton;
[SerializeField] private TextMeshProUGUI text;
```
Assign them in the inspector.

Then, use the `Start()` method to set up listeners for the start and stop buttons:
```
private void Start() {
    startButton.onClick.AddListener(StartRecording);
    stopButton.onClick.AddListener(StopRecording);
}
```

At this point, your script should look something like this:
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

### 3. Record Microphone Input

Now let's record Microphone input and encode it in WAV format. Start by defining the member variables:
```
private AudioClip clip;
private byte[] bytes;
private bool recording;
```

Then, in `StartRecording()`, using the `Microphone.Start()` method to start recording:
```
private void StartRecording() {
    clip = Microphone.Start(null, false, 10, 44100);
    recording = true;
}
```
This will record up to 10 seconds of audio at 44100 Hz.

In case the recording reaches its maximum length of 10 seconds, we'll want to stop the recording automatically. To do so, write the following in the `Update()` method:
```
private void Update() {
    if (recording && Microphone.GetPosition(null) >= clip.samples) {
        StopRecording();
    }
}
```

Then, in `StopRecording()`, truncate the recording and encode it in WAV format:
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

Finally, we'll need to implement the `EncodeAsWAV()` method, to prepare the audio data for the Hugging Face API:
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

The full script should now look something like this:
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

To test whether this code is working correctly, you can add the following line to the end of the `StopRecording()` method:
```
File.WriteAllBytes(Application.dataPath + "/test.wav", bytes);
```
Now, if you click the `Start` button, speak into the microphone, and click `Stop`, a `test.wav` file should be saved in your Unity Assets folder with your recorded audio.

### 4. Speech Recognition

Next, we'll want to use the Hugging Face Unity API to run speech recognition on our encoded audio. To do so, we'll create a `SendRecording()` method:
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
This will send the encoded audio to the API, displaying the response in white if successful, otherwise the error message in red.

Don't forget to call `SendRecording()` at the end of the `StopRecording()` method:
```
private void StopRecording() {
    /* other code */
    SendRecording();
}
```

### 5. Final Touches

Finally, let's improve the UX of this demo a bit using button interactability and status messages.

The Start and Stop buttons should only be interactable when appropriate, i.e. when a recording is ready to be started/stopped.

Then, set the response text to a simple status message while recording or waiting for the API.

The finished script should look something like this:
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

Congratulations, you can now use state-of-the-art Speech Recognition in Unity!

If you have any questions or would like to get more involved in using Hugging Face for Games, join the [Hugging Face Discord](https://hf.co/join/discord)!