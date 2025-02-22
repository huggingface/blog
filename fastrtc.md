---
title: "FastRTC: The Real-Time Communication Library for Python" 
thumbnail: /blog/assets/fastrtc/fastrtc_logo.png
authors:
- user: freddyaboulton
---

# FastRTC: The Real-Time Communication Library for Python

In the last six months, the AI audio space has exploded with model releases (for both open and closed source models) and investor and developer interest. To name a few milestones:

- OpenAI and Google released their live multimodal APIs for ChatGPT and Gemini. OpenAI even went so far as to release a 1-800-ChatGPT phone number!
- Kyutai released Moshi, a fully open-source audio-to-audio LLM. Alibaba released Qwen2-Audio and Fixie.ai released Ultravox - two open-source LLMs that natively understand audio.
- EleveLabs raised $180m in their Series C.

Despite the explosion in the model and funding side, it's still difficult to build real-time AI applications, especially in Python.

- ML engineers may not have experience with the technologies needed to build real-time applications.
- Even code assistant tools like Cursor and Copilot struggle to write python code that supports real-time audio/video applications. I know from experience!

That's why we're excited to announce `FastRTC`, the real-time communication library for Python. The library is designed to make it super easy to build real-time audio and video AI applications entirely in Python! Let's dive in.

## Core Features

At the heart of `FastRTC` is the `Stream` class. A powerful abstraction that wraps any python function with a real-time communication layer with the following core features:

- 🗣️ Automatic Voice Detection and Turn Taking built-in, only worry about the logic for responding to the user.
- 💻 Automatic UI - Built-in webRTC-enabled [Gradio](https://www.gradio.app/) UI for testing (or deploying to prod!).
- 📞 Call via Phone - Use fastphone() to get a FREE phone number to call into your audio stream (HF Token required. Increased limits for PRO accounts).
- ⚡️ WebRTC and Websocket support
- 💪 Customizable - You can mount the stream to any FastAPI app so you can serve a custom UI or deploy beyond Gradio
- 🧰 Lots of utils for text-to-speech, speech-to-text, stop word detection to get you started


## Getting Started

Let's start into the "hello word" of real-time audio: echoing back what the user says. In `FastRTC`, this is as simple as:

```python
from fastrtc import Stream, ReplyOnPause
from numpy import np

def echo(audio: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    yield audio

stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
stream.ui.launch()
```

Let's break it down:
- The `ReplyOnPause` will handle the voice detection and turn taking for you. You just have to worry about the logic for responding to the user. Any generator that returns a tuple of audio, (represented as `(sample_rate, audio_data)`) will work.
- The `Stream` class will build a production-ready Gradio UI for you to quickly test out your stream (or deploy to prod!).

Here it is in action:

<video src="https://github.com/user-attachments/assets/fcf2d30e-3e98-47c9-8dc3-23340784c441" controls /></video>

## Leveling-Up: LLM Voice Chat

The next level is to use an LLM to respond to the user. `FastRTC` comes with built-in speech-to-text and text-to-speech capabilities so working with LLMs is really easy. Let's change our `echo` function accordingly:

```python
import os

from fastrtc import (ReplyOnPause, Stream, get_stt_model, get_tts_model)
from openai import OpenAI

sambanova_client = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"), base_url="https://api.sambanova.ai/v1"
)
stt_model = get_stt_model()
tts_model = get_tts_model()

def echo(audio):
    prompt = stt_model.stt(audio)
    response = sambanova_client.chat.completions.create(
        model="Meta-Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    prompt = response.choices[0].message.content
    for audio_chunk in tts_model.stream_tts_sync(prompt):
        yield audio_chunk

stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
stream.ui.launch()
```

We're using the SambaNova API since it's fast. But you can use any LLM/text-to-speech/speech-to-text API. Bring the tools you love - `FastRTC` just handles the real-time communication layer.

<video src="https://github.com/user-attachments/assets/85dfbd52-b3f9-4354-b8fe-7ab9abb04bfd" controls /></video>

## Bonus: Call via Phone

If instead of `stream.ui.launch()`, you call `stream.fastphone()`, you'll get a free phone number to call into your stream. Note, a HF token is required. Increased limits for PRO accounts.

You'll see something like this in your terminal:

```
INFO:	  Your FastPhone is now live! Call +1 877-713-4471 and use code 530574 to connect to your stream.
INFO:	  You have 30:00 minutes remaining in your quota (Resetting on 2025-03-23)
```

You can then call the number and it will connect you to your stream!

<video src="https://github.com/user-attachments/assets/de2a27b1-1e08-4959-92f4-6baa01d98bb3" controls /></video>


## Next Steps

- Read the [docs](https://fastrtc.org/pr-preview/pr-60/) to learn more about the basics of `FastRTC`
- The best way to start building is by checking out the [cookbook](https://fastrtc.org/pr-preview/pr-60/cookbook). Find out how to intergrate with popular LLM providers (including OpenAI and Gemini's real-time APIs), integrate your stream with a FastAPI app and do a custom deployment, return additional data from your handler, do video processing, and more!
- ⭐️ Star the [repo](https://github.com/freddyaboulton/gradio-webrtc) and file bug and issue requests!
- Follow the [FastRTC Org](https://huggingface.co/fastrtc) on HuggingFace for updates and check out deployed examples!

Thank you for checking out `FastRTC`!




