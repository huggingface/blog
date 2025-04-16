---
title: "Hugging Face and Cloudflare Partner to Make Real-Time Speech and Video Seamless with FastRTC" 
thumbnail: /blog/assets/fastrtc-cloudflare/fastrtc_cloudflare.png
authors:
- user: freddyaboulton
---

# Hugging Face and Cloudflare Partner to Make Real-Time Speech and Video Seamless with FastRTC

We're excited to announce a new partnership between Cloudflare and Hugging Face that gives FastRTC developers instant access to enterprise-grade WebRTC infrastructure with a Hugging Face token.

As a preview of what you can build with FastRTC and Cloudflare, check out this voice chat [app](https://huggingface.co/spaces/fastrtc/talk-to-llama4) built with Meta's new Llama 4 model!

<video src="https://github.com/user-attachments/assets/f6d09e47-5e40-4296-b6cd-11d7f68baee2" controls /></video>

## Meeting a Gap in the Toolbox of AI Developers

As conversational AI becomes a core interface for tools, products, and services, real-time communication infrastructure is increasingly essential to support natural, multimodal interactions.
Hugging Face built [FastRTC](https://fastrtc.org/) to let AI developers build low-latency AI-powered audio and video streams with minimal Python code by abstracting away the complexities of WebRTC – the gold standard technology for real-time communication.

WebRTC-powered applications often face deployment challenges due to the need for specialized TURN servers, which enable reliable connections across different network environments. To address this issue, Cloudflare has built a global network of these servers that spans over 335 locations worldwide.

This partnership combines FastRTC’s easy development approach with Cloudflare's global TURN network, ensuring developers can create fast and reliable WebRTC applications with global connectivity. 

## Free Access with Your Hugging Face Account

FastRTC developers with a valid Hugging Face Access Token can stream 10GB of data for FREE every month without a credit card. Once the monthly limit is reached, developers can switch to their Cloudflare account for higher capacity ([instructions](https://fastrtc.org/deployment/#cloudflare-calls-api)). 

## Why This Matters for AI Developers

This partnership is especially valuable for AI developers building:
- Voice assistants that need reliable, low-latency audio streaming
- Video analysis applications that process camera feeds in real-time
- Multimodal AI applications that combine audio, video, and text

This partnership lets developers focus on their core application logic with FastRTC, while eliminating the need to build and maintain TURN infrastructure. Cloudflare's managed service handles global scalability and reliability, allowing AI developers to deliver exceptional experiences without the overhead of maintaining infrastructure.

## Getting Started

The integration will be available in the FastRTC version `0.0.20` and above. To get started:
- Ensure you have a Hugging Face token with read access(get one [here](https://huggingface.co/settings/tokens))
- Install or upgrade FastRTC: `pip install --upgrade 'fastrtc[vad]'`
- Configure your Stream to use the Cloudflare TURN network as shown below:

```python
from fastrtc import ReplyOnPause, Stream, get_cloudflare_turn_credentials
import os

os.environ["HF_TOKEN"] = "<your-hf-token>"

def echo(audio):
    yield audio

stream = Stream(ReplyOnPause(echo),
                rtc_config=get_cloudflare_turn_credentials)
stream.ui.launch()
```

Launch your script with python, `python <name of your script>.py`  


See this [Collection](https://huggingface.co/collections/fastrtc/cloudflare-partnership-67f437e0dfd19818d62ccb81) on Hugging Face as well as the [FastRTC Cookbook](https://fastrtc.org/cookbook/) for more examples.

## What's Next?

If you have any questions or feedback, please reach out to us on [GitHub](https://github.com/freddyaboulton/fastrtc) or [Hugging Face](https://huggingface.co/fastrtc). Please follow us on [Hugging Face](https://huggingface.co/fastrtc) for latest updates and announcements.