---
title: "Hugging Face and Cerebras bring Gemma 4 to real-time voice AI"
thumbnail: /blog/assets/cerebras-gemma4-voice-ai/thumbnail.png
authors:
- user: A-Mahla
- user: andito
- user: lvwerra
- user: vyassaurabh
  guest: true
  org: cerebras
---

# Hugging Face and Cerebras bring Gemma 4 to real-time voice AI

For voice AI, latency is a critical parameter. Developers have made tremendous progress in model quality, but the user experience is still often limited by response times. Hugging Face and Cerebras are changing that experience. Today, we demonstrate what becomes possible when an open, modular voice AI architecture is paired with industry-leading inference speed.

The result is a speech-to-speech experience that feels dramatically more natural. Instead of waiting for an AI to respond, conversations flow with the responsiveness users expect from human interaction.

## Architecture: an Open, Cascaded Speech-to-Speech stack

The demo is built as a real-time speech-to-speech pipeline. Each part of the system is modular, open, and replaceable, making it easy for developers to adapt the stack for different assistants, robots, products, or research projects.

This creates a fully open speech-to-speech loop:

Speech → speech recognition with Nvidia's Parakeet → Gemma 4 VLM inference on Cerebras → text-to-speech with Alibaba's Qwen3TTS → spoken response

The architecture brings together the strength of the open-source AI ecosystem: Cerebras for fast inference, Google DeepMind’s Gemma 4 31B for the language model, and Qwen for text-to-speech. Every layer can be inspected, modified, and extended by the developers

## Cerebras and Hugging Face Partnership

Today, some production systems see a reasonable median latency while still experiencing frustrating multi-second delays at the P95. Those delays become even more noticeable when tool calls or multimodal steps require multiple turns.

Cerebras helps solve one of the most important bottlenecks in the stack: the language-model response time. By making inference dramatically faster and more stable, Cerebras allows the rest of the Hugging Face pipeline to shine.

That stability is especially important at the long tail. Many systems can deliver acceptable median response times, but occasional slow responses still make conversations feel unreliable.

## Built for real-world interaction

This same Hugging Face speech-to-speech pipeline already powers Reachy Mini robots, with more than 9,000 robots in the wild. For robots, voice assistants, and embodied AI, responsiveness is not a cosmetic improvement. It is what makes the interaction feel alive.

The motivation to use Cerebras is therefore not simply cost reduction. It is low latency, predictable performance, and the ability to create real-time experiences that feel natural at scale.

This collaboration reflects a shared belief that the future of AI will be both open and performant. Open-source models, open infrastructure, and breakthrough inference speed together create a foundation for the next generation of conversational AI.

We invite developers to explore the demo, experiment with the code, and help shape what comes next for real-time voice AI.

Demo: [Hugging Face Space](https://huggingface.co/spaces/amir-tfrere/minimal-conversation-app-s2s-backend-websocket)

Repository: huggingface/speech-to-speech
