---
title: Simple considerations for simple people building fancy neural networks
thumbnail: /blog/assets/13_simple-considerations/henry-co-3coKbdfnAFg-unsplash.jpg
authors:
- user: VictorSanh
---

![Builders](/blog/assets/13_simple-considerations/henry-co-3coKbdfnAFg-unsplash.jpg)

<span class="text-gray-500 text-xs">Photo by [Henry & Co.](https://unsplash.com/@hngstrm?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/builder?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)</span>

# 🚧 Simple considerations for simple people building fancy neural networks


As machine learning continues penetrating all aspects of the industry, neural networks have never been so hyped. For instance, models like GPT-3 have been all over social media in the past few weeks and continue to make headlines outside of tech news outlets with fear-mongering titles.

![Builders](/blog/assets/13_simple-considerations/1_sENCNdlC7zK4bg22r43KiA.png)

<div class="text-center text-xs text-gray-500">
	<a class="text-gray-500" href="https://www.theguardian.com/commentisfree/2020/sep/08/robot-wrote-this-article-gpt-3">An article</a> from The Guardian
</div>

At the same time, deep learning frameworks, tools, and specialized libraries democratize machine learning research by making state-of-the-art research easier to use than ever. It is quite common to see these almost-magical/plug-and-play 5 lines of code that promise (near) state-of-the-art results. Working at [Hugging Face](https://huggingface.co/) 🤗, I admit that I am partially guilty of that. 😅 It can give an inexperienced user the misleading impression that neural networks are now a mature technology while in fact, the field is in constant development.

In reality, **building and training neural networks can often be an extremely frustrating experience**:

*   It is sometimes hard to understand if your performance comes from a bug in your model/code or is simply limited by your model’s expressiveness.
*   You can make tons of tiny mistakes at every step of the process without realizing at first, and your model will still train and give a decent performance.

**In this post, I will try to highlight a few steps of my mental process when it comes to building and debugging neural networks.** By “debugging”, I mean making sure you align what you have built and what you have in mind. I will also point out things you can look at when you are not sure what your next step should be by listing the typical questions I ask myself.

_A lot of these thoughts stem from my experience doing research in natural language processing but most of these principles can be applied to other fields of machine learning._

## 1. 🙈 Start by putting machine learning aside

It might sound counter-intuitive but the very first step of building a neural network is to **put aside machine learning and simply focus on your data**. Look at the examples, their labels, the diversity of the vocabulary if you are working with text, their length distribution, etc. You should dive into the data to get a first sense of the raw product you are working with and focus on extracting general patterns that a model might be able to catch. Hopefully, by looking at a few hundred examples, you will be able to identify high-level patterns. A few standard questions you can ask yourself:

*   Are the labels balanced?
*   Are there gold-labels that you do not agree with?
*   How were the data obtained? What are the possible sources of noise in this process?
*   Are there any preprocessing steps that seem natural (tokenization, URL or hashtag removing, etc.)?
*   How diverse are the examples?
*   What rule-based algorithm would perform decently on this problem?

It is important to get a **high-level feeling (qualitative) of your dataset along with a fine-grained analysis (quantitative)**. If you are working with a public dataset, someone else might have already dived into the data and reported their analysis (it is quite common in Kaggle competition for instance) so you should absolutely have a look at these!

## 2. 📚 Continue as if you just started machine learning

Once you have a deep and broad understanding of your data, I always recommend **to put yourself in the shoes of your old self when you just started machine learning** and were watching introduction classes from Andrew Ng on Coursera. **Start as simple as possible to get a sense of the difficulty of your task and how well standard baselines would perform.** For instance, if you work with text, standard baselines for binary text classification can include a logistic regression trained on top of word2vec or fastText embeddings. With the current tools, running these baselines is as easy (if not more) as running BERT which can arguably be considered one of the standard tools for many natural language processing problems. If other baselines are available, run (or implement) some of them. It will help you get even more familiar with the data.

As developers, it easy to feel good when building something fancy but it is sometimes hard to rationally justify it if it beats easy baselines by only a few points, so it is central to make sure you have reasonable points of comparisons:

*   How would a random predictor perform (especially in classification problems)? Dataset can be unbalanced…
*   What would the loss look like for a random predictor?
*   What is (are) the best metric(s) to measure progress on my task?
*   What are the limits of this metric? If it’s perfect, what can I conclude? What can’t I conclude?
*   What is missing in “simple approaches” to reach a perfect score?
*   Are there architectures in my neural network toolbox that would be good to model the inductive bias of the data?

## 3. 🦸‍♀️ Don’t be afraid to look under the hood of these 5-liners templates

Next, you can start building your model based on the insights and understanding you acquired previously. As mentioned earlier, implementing neural networks can quickly become quite tricky: there are many moving parts that work together (the optimizer, the model, the input processing pipeline, etc.), and many small things can go wrong when implementing these parts and connecting them to each other. **The challenge lies in the fact that you can make these mistakes, train a model without it ever crashing, and still get a decent performance…**

Yet, it is a good habit when you think you have finished implementing to **overfit a small batch of examples** (16 for instance). If your implementation is (nearly) correct, your model will be able to overfit and remember these examples by displaying a 0-loss (make sure you remove any form of regularization such as weight decay). If not, it is highly possible that you did something wrong in your implementation. In some rare cases, it means that your model is not expressive enough or lacks capacity. Again, **start with a small-scale model** (fewer layers for instance): you are looking to debug your model so you want a quick feedback loop, not a high performance.

> Pro-tip: in my experience working with pre-trained language models, freezing the embeddings modules to their pre-trained values doesn’t affect much the fine-tuning task performance while considerably speeding up the training.

Some common errors include:

*   Wrong indexing… (these are really the worst 😅). Make sure you are gathering tensors along the correct dimensions for instance…
*   You forgot to call `model.eval()` in evaluation mode (in PyTorch) or `model.zero\_grad()` to clean the gradients
*   Something went wrong in the pre-processing of the inputs
*   The loss got wrong arguments (for instance passing probabilities when it expects logits)
*   Initialization doesn’t break the symmetry (usually happens when you initialize a whole matrix with a single constant value)
*   Some parameters are never called during the forward pass (and thus receive no gradients)
*   The learning rate is taking funky values like 0 all the time
*   Your inputs are being truncated in a suboptimal way

> Pro-tip: when you work with language, have a serious **look at the outputs of the tokenizers**. I can’t count the number of lost hours I spent trying to reproduce results (and sometimes my own old results) because something went wrong with the tokenization.🤦‍♂️

Another useful tool is **deep-diving into the training dynamic** and plot (in Tensorboard for instance) the evolution of multiple scalars through training. At the bare minimum, you should look at the dynamic of your loss(es), the parameters, and their gradients.

As the loss decreases, you also want to look at the model’s predictions: either by evaluating on your development set or, my personal favorite, **print a couple of model outputs**. For instance, if you are training a machine translation model, it is quite satisfying to see the generations become more and more convincing through the training. You want to be more specifically careful about overfitting: your training loss continues to decreases while your evaluation loss is aiming at the stars.💫

## 4. 👀 Tune but don’t tune blindly

Once you have everything up and running, you might want to tune your hyperparameters to find the best configuration for your setup. I generally stick with a random grid search as it turns out to be fairly effective in practice.

> Some people report successes using fancy hyperparameter tuning methods such as Bayesian optimization but in my experience, random over a reasonably manually defined grid search is still a tough-to-beat baseline.

Most importantly, there is no point of launching 1000 runs with different hyperparameters (or architecture tweaks like activation functions): **compare a couple of runs with different hyperparameters to get an idea of which hyperparameters have the highest impact** but in general, it is delusional to expect to get your biggest jumps of performance by simply tuning a few values. For instance, if your best performing model is trained with a learning rate of 4e2, there is probably something more fundamental happening inside your neural network and you want to identify and understand this behavior so that you can re-use this knowledge outside of your current specific context.

On average, experts use fewer resources to find better solutions.

To conclude, a piece of general advice that has helped me become better at building neural networks is to **favor (as most as possible) a deep understanding of each component of your neural network instead of blindly (not to say magically) tweak the architecture**. Keep it simple and avoid small tweaks that you can’t reasonably justify even after trying really hard. Obviously, there is the right balance to find between a “trial-and-error” and an “analysis approach” but a lot of these intuitions feel more natural as you accumulate practical experience. **You too are training your internal model.** 🤯

A few related pointers to complete your reading:

*   [Reproducibility (in ML) as a vehicle for engineering best practices](https://docs.google.com/presentation/d/1yHLPvPhUs2KGI5ZWo0sU-PKU3GimAk3iTsI38Z-B5Gw/edit#slide=id.p) from Joel Grus
*   [Checklist for debugging neural networks](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21) from Cecelia Shao
*   [How to unit test machine learning code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) from Chase Roberts
*   [A recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) from Andrej Karpathy
