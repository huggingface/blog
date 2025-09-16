---
title: "Visible Watermarking with Gradio"
thumbnail: /blog/assets/watermarking_with_gradio/thumbnail.png
authors:
- user: meg
---

# Visible Watermarking with Gradio



![image/webp](https://cdn-uploads.huggingface.co/production/uploads/60c757ea5f9a76ab3f844f12/DM6_nlPtgsfRytok2F1Ib.webp)


Last year, we shared a 
<a href="https://huggingface.co/blog/watermarking" target="_blank">blogpost on watermarking</a>, explaining what it means to watermark generative AI content, and why it's important. The need for watermarking has become even more critical as people all over the world have begun to generate and share AI-generated images, video, audio, and text. Images and video have become so realistic that they’re nearly impossible to distinguish from what you’d see captured by a real camera.  Addressing this issue is multi-faceted, but there is one, clear, low-hanging fruit 🍇:

> [!NOTE]
> ### _In order for people to know what's real and what's synthetic, use visible watermarks._

To help out, we at Hugging Face have made visible watermarking trivially easy: Whenever you [create a Space like an app or a demo](https://huggingface.co/new-space), you can use our in-house app-building library Gradio to display watermarks with a single command. 

For images and video, simply add the `watermark` parameter, like so:

```
gr.Image(my_generated_image, watermark=my_watermark_image)
```

```
gr.Video(my_generated_video, watermark=my_watermark_image)
```

> [!TIP]
> ### See a demonstration of this in action: [check out our example image and video watermarking Space](https://huggingface.co/spaces/meg/watermark_demo).

Watermarks can be specified as filenames, and for images we additionally support open images or even numpy arrays, to work best with how you want to set up your interface. One option I particularly like is QR watermarks, which can be used to get much more information about the content, [and can even be matched to the style of your image or video](https://huggingface.co/spaces/huggingface-projects/QR-code-AI-art-generator).

You can also add custom visible watermarks for AI-generated text, so that whenever it is copied, the watermark will appear.  Like so:

```
gr.Chatbot(label=my_model_name, watermark=my_watermark_text, type="messages", show_copy_button=True, show_copy_all_button=True)
```

> [!TIP]
> ### See a demonstration of this in action: [check out our example chatbot watermarking Space](https://huggingface.co/spaces/meg/chatbot_watermark_demo).

This automatically adds attribution when users copy text from AI responses, further aiding in AI transparency and disclosure for text generation.

Try it all out today, build your own watermark, have fun!

Happy Coding!


_Acknowledgements: [Abubakar Abid](https://huggingface.co/abidlabs) and [Yuvraj Sharma](https://huggingface.co/ysharma) collaborated on this work and blog post._