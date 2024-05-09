---
title: "Energy Star Ratings for AI Models - a Proposal"
thumbnail: /blog/assets/energy_star_intro/thumbnail.png
authors:
- user: sasha
- user: yacine
- user: regisss
- user: IlyasMoutawwakil
---

# Energy Star Ratings for AI Models

## Background and Inspiration - EPA Energy Star Ratings

Every time we use an electronic device like a computer, a phone, or even a washing machine, we are using energy to power the device; and depending on how that energy is generated, this can result in greenhouse gas (GHG) emissions. In 1992, the U.S. Environmental Protection Agency (EPA) launched the Energy Star program as a way to set energy efficiency specifications for different types of devices and help consumers make informed decisions. In the last 30 years, the Energy Star program has saved billions of metric tons of greenhouse gas emissions and has come to encompass dozens of product categories, from data centers to dishwashers.

## Energy Star AI Project

For each query sent to an AI model, we also use energy, whether it be locally on our computer or on a server in the cloud. The amount of energy that we use will depend on the characteristics of the model, such as its size and architecture, and the way in which it's deployed, i.e. the optimizations and engineering choices made.

The aim of the [Energy Star AI project](https://huggingface.co/EnergyStarAI) is to develop an Energy Star rating system for AI model deployment that will guide members of the community in choosing models (and ways to run them) for different tasks based on their energy efficiency and to analyze the effect of implementation choices on the downstream energy usage of different models.

## Tasks and Models

Since the original Energy Star ratings were developed to span a variety of use cases and consumer products, we picked 10 popular [tasks](https://huggingface.co/tasks) spanning both language, audio and computer vision, and including multi-modal tasks. By testing a variety of models across different tasks, we aim to cover different use cases and AI applications that are relevant to different groups:

**Language**
- Text generation
- Summarization
- Extractive question answering
- Text classification
- Semantic Similarity

**Vision**
- Image classification
- Object detection

**Audio**
- Automatic speech recognition

**Multimodal**
- Text-to-Image
- Image-to-text

We've developed a testing dataset for each task that consists of 1,000 samples from at least 3 datasets per tasks, in order to represent different use cases: for instance, the text generation task dataset consists of random samples from [WikiText](https://huggingface.co/datasets/wikitext), [OSCAR](https://huggingface.co/datasets/oscar) and [UltraChat-10K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).

For each task, we took a sample of popular and recent models from the [Hugging Face Hub](https://huggingface.co/models), spanning a wide variety of sizes and architectures. For each task, we also defined a set of control variables -- controlling for things like batch size, number of tokens generated, image dimensions, sampling rates -- to allow for the standardized testing of models.

## Initial Results

We are running the first series of tasks on NVIDIA H100 GPUs on the Hugging Face compute cluster: text classification, image classification, question answering and text generation.

Our initial results show that the spread between models differs depending on the nature of the task, ranging from a difference of a factor of 5  between the most efficient and least efficient models for image classification, all the way to a factor of 50 for text generation.

The experiments we've run for task-specific (fine-tuned) versus zero-shot (T5-family) models are consistent with the results that we found in our [previous work](https://arxiv.org/abs/2311.16863) -- that zero-shot models use orders of magnitude more energy for tasks like text classification and question answering compared to single-task models for the same tasks.

## Future Work

After we have finished testing all ten tasks of our project, our goal is to establish an average and deviations per task, which we will use to assign the final Energy Star Ratings. We will then present our results via a ‘Green AI Leaderboard’ Space, to allow members of the community compare and explore different open-source models and tasks.

We will also test different implementation choices and optimization strategies to test their impacts on model efficiency, with the aim of identifying simple steps that the AI community can take to make their models more efficient.

Stay tuned for more results in the coming weeks!
