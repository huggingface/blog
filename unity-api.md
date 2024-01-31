---
title: "How to Install and Use the Hugging Face Unity API"
thumbnail: /blog/assets/124_ml-for-games/unity-api-thumbnail.png
authors:
- user: dylanebert
---

# How to Install and Use the Hugging Face Unity API

<!-- {authors} --> 

The [Hugging Face Unity API](https://github.com/huggingface/unity-api) is an easy-to-use integration of the [Hugging Face Inference API](https://huggingface.co/inference-api), allowing developers to access and use Hugging Face AI models in their Unity projects. In this blog post, we'll walk through the steps to install and use the Hugging Face Unity API.

## Installation

1. Open your Unity project
2. Go to `Window` -> `Package Manager`
3. Click `+` and select `Add Package from git URL`
4. Enter `https://github.com/huggingface/unity-api.git`
5. Once installed, the Unity API wizard should pop up. If not, go to `Window` -> `Hugging Face API Wizard`

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/packagemanager.gif">
</figure> 

6. Enter your API key. Your API key can be created in your [Hugging Face account settings](https://huggingface.co/settings/tokens).
7. Test the API key by clicking `Test API key` in the API Wizard.
8. Optionally, change the model endpoints to change which model to use. The model endpoint for any model that supports the inference API can be found by going to the model on the Hugging Face website, clicking `Deploy` -> `Inference API`, and copying the url from the `API_URL` field.
9. Configure advanced settings if desired. For up-to-date information, visit the project repository at `https://github.com/huggingface/unity-api`
10. To see examples of how to use the API, click `Install Examples`. You can now close the API Wizard.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/apiwizard.png">
</figure> 

Now that the API is set up, you can make calls from your scripts to the API. Let's look at an example of performing a Sentence Similarity task:

```
using HuggingFace.API;

/* other code */

// Make a call to the API
void Query() {
    string inputText = "I'm on my way to the forest.";
    string[] candidates = {
        "The player is going to the city",
        "The player is going to the wilderness",
        "The player is wandering aimlessly"
    };
    HuggingFaceAPI.SentenceSimilarity(inputText, OnSuccess, OnError, candidates);
}

// If successful, handle the result
void OnSuccess(float[] result) {
    foreach(float value in result) {
        Debug.Log(value);
    }
}

// Otherwise, handle the error
void OnError(string error) {
    Debug.LogError(error);
}

/* other code */
```

## Supported Tasks and Custom Models

The Hugging Face Unity API also currently supports the following tasks:

- [Conversation](https://huggingface.co/tasks/conversational)
- [Text Generation](https://huggingface.co/tasks/text-generation)
- [Text to Image](https://huggingface.co/tasks/text-to-image)
- [Text Classification](https://huggingface.co/tasks/text-classification)
- [Question Answering](https://huggingface.co/tasks/question-answering)
- [Translation](https://huggingface.co/tasks/translation)
- [Summarization](https://huggingface.co/tasks/summarization)
- [Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition)

Use the corresponding methods provided by the `HuggingFaceAPI` class to perform these tasks.

To use your own custom model hosted on Hugging Face, change the model endpoint in the API Wizard.

## Usage Tips

1. Keep in mind that the API makes calls asynchronously, and returns a response or error via callbacks.
2. Address slow response times or performance issues by changing model endpoints to lower resource models.

## Conclusion

The Hugging Face Unity API offers a simple way to integrate AI models into your Unity projects. We hope you found this tutorial helpful. If you have any questions or would like to get more involved in using Hugging Face for Games, join the [Hugging Face Discord](https://hf.co/join/discord)!