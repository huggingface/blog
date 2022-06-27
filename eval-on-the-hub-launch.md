---
title: "Announcing Evaluation on the Hub"
thumbnail: /blog/assets/82_eval_on_the_hub_launch/thumbnail.png
---

<h1>
    Announcing Evaluation on the Hub
</h1>

<div class="blog-metadata">
    <small>Published June 28, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/eval-on-the-hub-launch.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/lewtun">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1594651707950-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>lewtun</code>
            <span class="fullname">Lewis Tunstall</span>
        </div>
    </a>
    <a href="/abhishek">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1604427578240-noauth.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>abhishek</code>
            <span class="fullname">Abhishek Thakur</span>
        </div>
    </a>
    <a href="/Tristan">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1648247133961-61e9e3d4e2a95338e04c9f33.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>Tristan</code>
            <span class="fullname">Tristan Thrush</span>
        </div>
    </a>
    <a href="/sasha">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1626198087984-60edd0133e2c73a9a21455f5.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>sasha</code>
            <span class="fullname">Sasha Luccioni</span>
        </div>
    </a>
    <a href="/lvwerra">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1627890220261-5e48005437cb5b49818287a5.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>lvwerra</code>
            <span class="fullname">Leandro von Werra</span>
        </div>
    </a>
    <a href="/nazneen">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1648508910554-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>nazneen</code>
            <span class="fullname">Nazneen Rajani</span>
        </div>
    </a>
    <a href="/ola13">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1600081648206-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>ola13</code>
            <span class="fullname">Ola Piktus</span>
        </div>
    </a>
    <a href="/osanseviero">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1647857934921-6032802e1f993496bc14d9e3.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>osanseviero</code>
            <span class="fullname">Omar Sanseviero</span>
        </div>
    </a>
    <a href="/douwekiela">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1641847245435-61dc997715b47073db1620dc.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>douwekiela</code>
            <span class="fullname">Douwe Kiela</span>
        </div>
    </a>
</div>

<em>TL;DR</em>: Today we introduce [Evaluation on the Hub](https://huggingface.co/spaces/autoevaluate/model-evaluator), a new tool powered by [AutoTrain](https://huggingface.co/autotrain) that lets you evaluate any model on any dataset on the Hub without writing a single line of code!

Progress in AI has been nothing short of amazing, to the point where some people are now seriously debating whether AI models may be better than humans at certain tasks. However, that progress has not at all been even: to a machine learner from several decades ago, modern hardware and algorithms might look incredible, as might the sheer quantity of data and compute at our disposal, but the way we evaluate these models has stayed roughly the same.

However, it is no exaggeration to say that modern AI is in an evaluation crisis. Proper evaluation these days involves measuring many models, often on many datasets and with multiple metrics. But doing so is unnecessarily cumbersome. This is especially the case if we care about reproducibility, since self-reported results may have suffered from inadvertent bugs, subtle differences in implementation, or worse.

We believe that better evaluation can happen, if we - the community - establish a better set of best practices and try to remove the hurdles. Over the past few months, we've been hard at work on [Evaluation on the Hub](https://huggingface.co/spaces/autoevaluate/model-evaluator): evaluate any model on any dataset using any metric, at the click of a button. To get started, we evaluated hundreds models on several key datasets, and using the nifty new [Pull Request feature](https://huggingface.co/blog/community-update) on the Hub, opened up loads of PRs on model cards to display their verified performance. Evaluation results are encoded directly in the model card metadata, following [a format](https://github.com/huggingface/hub-docs/blob/main/modelcard.md) for all models on the Hub. Check out the model card for [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/README.md#L7-L42) to see how it looks!

## On the Hub

Evaluation on the Hub opens the door to so many interesting use cases. From the data scientist or executive who needs to decide which model to deploy, to the academic trying to reproduce a paper’s results on a new dataset, to the ethicist who wants to better understand risks of deployment. If we have to single out three primary initial use case scenarios, they are these:

**Finding the best model for your task**
Suppose you know exactly what your task is and you want to find the right model for the job. You can check out the leaderboard for a dataset representative of your task, which aggregates all the results. That’s great! And what if that fancy new model you’re interested in isn’t on the [leaderboard](https://huggingface.co/spaces/autoevaluate/leaderboards) yet for that dataset? Simply run an evaluation for it, without leaving the Hub.

**Evaluating models on your brand new dataset**
Now what if you have a brand spanking new dataset that you want to run baselines on? You can upload it to the Hub and evaluate as many models on it as you like. No code required. What’s more, you can be sure that the way you are evaluating these models on your dataset is exactly the same as how they’ve been evaluated on other datasets.

**Evaluating your model on many other related datasets**
Or suppose you have a brand new question answering model, trained on SQuAD? There are hundreds of different question answering datasets to evaluate on :scream: You can pick the ones you are interested in and evaluate your model, directly from the Hub.

## Ecosystem

![The Hugging Face Ecosystem and Evaluation on the Hub](/blog/assets/82_eval_on_the_hub_launch/ecosystem.png)
<figcaption><center><i>Evaluation on the Hub fits neatly into the Hugging Face ecosystem.</i></center></figcaption>

Evaluation on the Hub is meant to make your life easier. But of course, there’s a lot happening in the background. What we really like about Evaluation on the Hub: it fits so neatly into the existing Hugging Face ecosystem, we almost had to do it. Users start on dataset pages, from where they can launch evaluations or see leaderboards. The model evaluation submission interface and the leaderboards are regular Hugging Face Spaces. The evaluation backend is powered by AutoTrain, which opens up a PR on the Hub for the given model’s model card.

## DogFood - Distinguishing Dogs, Muffins and Fried Chicken

So what does it look like in practice? Let’s run through an example. Suppose you are in the business of telling apart dogs, muffins and fried chicken (a.k.a. dogfooding!).

![Dog Food Examples](/blog/assets/82_eval_on_the_hub_launch/dogfood-example.png)
<figcaption><center><i>Example images of dogs and food (muffins and fried chicken). <a href="https://github.com/qw2243c/Image-Recognition-Dogs-Fried-Chicken-or-Blueberry-Muffins-/">Source</a> / <a href="https://twitter.com/teenybiscuit/status/667777205397680129?s=20&t=wPgYJMp-JPwRsNAOMvEbxg">Original source</a>.</i></center></figcaption>

As the above image shows, to solve this problem, you’ll need:

* A dataset of dog, muffin, and fried chicken images
* Image classifiers that have been trained on these images

Fortunately, your data science team has uploaded [a dataset](https://huggingface.co/datasets/lewtun/dog_food) to the Hugging Face Hub and trained [a few different models on it](https://huggingface.co/models?datasets=lewtun/dog_food). So now you just need to pick the best one - let’s use Evaluation on the Hub to see how well they perform on the test set!

### Configuring an evaluation job

To get started, head over to the [`model-evaluator` Space](https://huggingface.co/spaces/autoevaluate/model-evaluator) and select the dataset you want to evaluate models on. For our dataset of dog and food images, you’ll see something like the image below:

![Model Evaluator](/blog/assets/82_eval_on_the_hub_launch/model-evaluator.png)

Now, many datasets on the Hub contain metadata that specifies how an evaluation should be configured (check out [acronym_identification](https://huggingface.co/datasets/acronym_identification/blob/main/README.md#L22-L30) for an example). This allows you to evaluate models with a single click, but in our case we’ll show you how to configure the evaluation manually.

Clicking on the <em>Advanced configuration</em> button will show you the various settings to choose from:

* The task, dataset, and split configuration
* The mapping of the dataset columns to a standard format
* The choice of metrics

As shown in the image below, configuring the task, dataset, and split to evaluate on is straightforward:

![Advanced Configuration](/blog/assets/82_eval_on_the_hub_launch/config.png)

The next step is to define which dataset columns contain the images, and which ones contain the labels:

![Dataset Mapping](/blog/assets/82_eval_on_the_hub_launch/mapping.png)

Now that the task and dataset are configured, the final (optional) step is to select the metrics to evaluate with. Each task is associated with a set of default metrics. For example, the image below shows that F1 score, accuracy etc will be computed automatically. To spice things up, we’ll also calculate the [Matthew’s correlation coefficient](https://huggingface.co/spaces/evaluate-metric/matthews_correlation), which provides a balanced measure of classifier performance:

![Selecting Metrics](/blog/assets/82_eval_on_the_hub_launch/select-metrics.png)

And that’s all it takes to configure an evaluation job! Now we just need to pick some models to evaluate - let’s take a look.

### Selecting models to evaluate

Evaluation on the Hub links datasets and models via tags in the model card metadata. In our example, we have three models to choose from, so let’s select them all!

![Selecting Models](/blog/assets/82_eval_on_the_hub_launch/select-model.png)

Once the models are selected, simply enter your Hugging Face Hub username (to be notified when the evaluation is complete) and hit the big <em>Evaluate models</em> button:

![Launching the Evaluation](/blog/assets/82_eval_on_the_hub_launch/evaluate.png)

Once a job is submitted, the models will be automatically evaluated and a Hub pull request will be opened with the evaluation results:

![Pull Request](/blog/assets/82_eval_on_the_hub_launch/pr.png)

You can also copy-paste the evaluation metadata into the dataset card so that you and the community can skip the manual configuration next time!

![Metadata Pull Request](/blog/assets/82_eval_on_the_hub_launch/metadata.png)

### Check out the leaderboard

To facilitate the comparison of models, Evaluation on the Hub also provides leaderboards that allow you to examine which models perform best on which split and metric:

![Leaderboard](/blog/assets/82_eval_on_the_hub_launch/leaderboard.png)

Looks like the Swin Transformer came out on top!

### Try it yourself!

If you’d like to evaluate your own choice of models, give Evaluation on the Hub a spin by checking out these popular datasets:

* emotion: https://huggingface.co/spaces/autoevaluate/model-evaluator?dataset=emotion
* conll2003: https://huggingface.co/spaces/autoevaluate/model-evaluator?dataset=conll2003
* masakhaner: https://huggingface.co/spaces/autoevaluate/model-evaluator?dataset=masakhaner

## The Bigger Picture

Since the dawn of machine learning, we've evaluated models by computing some form of accuracy on a held-out test set that is assumed to be independent and identically distributed. Under the pressures of modern AI, that paradigm is now starting to show serious cracks.

Benchmarks are saturating, meaning that machines outperform humans on certain test sets, almost faster than we can come up with new ones. Yet, AI systems are known to be brittle and suffer from, or even worse amplify, severe malicious biases. Reproducibility is lacking. Openness is an afterthought. While people fixate on leaderboards, practical considerations for deploying models, such as efficiency and fairness, are often glossed over. The hugely important role data plays in model development is still not taken seriously enough. What is more, the practices of pretraining and prompt-based in-context learning have blurred what it means to be “in distribution” in the first place. Machine learning is slowly catching up to these things, and we hope to help the field move forward with our work.

## Next Steps

A few weeks ago, we launched the Hugging Face [Evaluate library](https://github.com/huggingface/evaluate), aimed at lowering barriers to the best practices of machine learning evaluation. We have also been hosting benchmarks, like [RAFT](https://huggingface.co/spaces/ought/raft-leaderboard) and [GEM](https://huggingface.co/spaces/GEM/submission-form). Evaluation on the Hub is a logical next step in our efforts to enable a future where models are evaluated in a more holistic fashion, along many axes of evaluation, in a trustable and guaranteeably reproducible manner. Stay tuned for more launches soon, including more tasks, and a new and improved [data measurements tool](https://huggingface.co/spaces/huggingface/data-measurements-tool)!

We’re excited to see where the community will take this! If you'd like to help out, evaluate as many models on as many datasets as you like. And as always, please give us lots of feedback, either on the [Community tabs](https://huggingface.co/spaces/autoevaluate/model-evaluator/discussions) or the [forums](https://discuss.huggingface.co/)!
