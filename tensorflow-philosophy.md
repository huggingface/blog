---
title: "Hugging Face's TensorFlow Philosophy"
thumbnail: /blog/assets/96_tensorflow_philosophy/thumbnail.png
---

<h1>
    Hugging Face's TensorFlow Philosophy
</h1>

<div class="blog-metadata">
    <small>Published Aug 12, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/tensorflow-philosophy.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/rocketknight1">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1660312628256-60ba519750effef3a58beac3.png?w=200&h=200&f=face">
        <div class="bfc">
            <code>rocketknight1</code>
            <span class="fullname">Matthew Carrigan</span>
        </div>
    </a>
</div>


### Introduction


Despite increasing competition from PyTorch and JAX, TensorFlow remains [the most-used deep learning framework](https://twitter.com/fchollet/status/1478404084881190912?lang=en). It also differs from those other two libraries in some very important ways. In particular, it’s quite tightly integrated with its high-level API `Keras`, and its data loading library `tf.data`.

There is a tendency among PyTorch engineers (picture me staring darkly across the open-plan office here) to see this as a problem to be overcome; their goal is to figure out how to make TensorFlow get out of their way so they can use the low-level training and data-loading code they’re used to. This is entirely the wrong way to approach TensorFlow! Keras is a great high-level API. If you push it out of the way in any project bigger than a couple of modules you’ll end up reproducing most of its functionality yourself when you realize you need it.

As refined, respected and highly attractive TensorFlow engineers, we want to use the incredible power and flexibility of cutting-edge models, but we want to handle them with the tools and API we’re familiar with. This blogpost will be about the choices we make at Hugging Face to enable that, and what to expect from the framework as a TensorFlow programmer.

### Interlude: 30 Seconds to :huggingface:

Experienced users can feel free to skim or skip this section, but if this is your first encounter with Hugging Face and `transformers`, I should start by giving you an overview of the core idea of the library: You just ask for a pretrained model by name, and you get it in one line of code. The easiest way is to just use the `TFAutoModel` class:

```py
from transformers import TFAutoModel

model = TFAutoModel.from_pretrained("bert-base-cased")
```

This one line will instantiate the model architecture and load the weights, giving you an exact replica of the original, famous [BERT](https://arxiv.org/abs/1810.04805) model. This model won’t do much on its own, though - it lacks an output head or a loss function. In effect, it is the “stem” of a neural net that stops right after the last hidden layer. So how do you put an output head on it? Simple, just use a different `AutoModel` class. Here we load the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) model and add an image classification head:

```py
from transformers import TFAutoModelForImageClassification

model_name = "google/vit-base-patch16-224"
model = TFAutoModelForImageClassification.from_pretrained(model_name)
```

Now our `model` has an output head and, optionally, a loss function appropriate for its new task. If the new output head differs from the original model, then its weights will be randomly initialized. All other weights will be loaded from the original model. But why do we do this? Why would we use the stem of an existing model, instead of just making the model we need from scratch?

It turns out that large models pretrained on lots of data are much, much better starting points for almost any ML problem than the standard method of simply randomly initializing your weights. This is called **transfer learning**, and if you think about it, it makes sense - solving a textual task well requires some knowledge of language, and solving a visual task well requires some knowledge of images and space. The reason ML is so data-hungry without transfer learning is simply that this basic domain knowledge has to be relearned from scratch for every problem, which necessitates a huge volume of training examples. By using transfer learning, however, a problem can be solved with a thousand training examples that might have required a million without it, and often with a higher final accuracy. For more on this topic, check out the relevant sections of the [Hugging Face Course](https://www.youtube.com/watch?v=BqqfQnyjmgg)!

When using transfer learning, however, it's very important that you process inputs to the model the same way that they were processed during training. This ensures that the model has to relearn as little as possible when we transfer its knowledge to a new problem. In `transformers`, this preprocessing is often handled with **tokenizers**. Tokenizers can be loaded in the same way as models, using the `AutoTokenizer` class. Be sure that you load the tokenizer that matches the model you want to use!

```py
from transformers import TFAutoModel, AutoTokenizer

# Make sure to always load a matching tokenizer and model!
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = TFAutoModel.from_pretrained("bert-base-cased")

# Let's load some data and tokenize it
test_strings = ["This is a sentence!", "This is another one!"]
tokenized_inputs = tokenizer(test_strings, return_tensors="np", padding=True)

# Now our data is tokenized, we can pass it to our model, or use it in fit()!
outputs = model(tokenized_inputs)
```

This is just a taste of the library, of course - if you want more, you can check out our [notebooks](https://huggingface.co/docs/transformers/notebooks), or our [code examples](https://github.com/huggingface/transformers/tree/main/examples/tensorflow). There are also several other [examples of the library in action at keras.io](https://keras.io/examples/#natural-language-processing)!

At this point, you now understand some of the basic concepts and classes in `transformers`. Everything I’ve written above is framework-agnostic (with the exception of the “TF” in `TFAutoModel`), but when you want to actually train and serve your model, that’s when things will start to diverge between the frameworks. And that brings us to the main focus of this article: As a TensorFlow engineer, what should you expect from `transformers`?

#### Philosophy #1: All TensorFlow models should be Keras Model objects, and all TensorFlow layers should be Keras Layer objects.

This almost goes without saying for a TensorFlow library, but it’s worth emphasizing regardless. From the user’s perspective, the most important effect of this choice is that you can call Keras methods like `fit()`, `compile()` and `predict()` directly on our models.

For example, assuming your data is already prepared and tokenized, then getting predictions from a sequence classification model with TensorFlow is as simple as:

```py
model = TFAutoModelForSequenceClassification.from_pretrained(my_model)
model.predict(my_data)
```

And if you want to train that model instead, it's just:

```py
model.fit(my_data, my_labels)
```

However, this convenience doesn’t mean you’re limited to tasks that we support out of the box. Keras models can be composed as layers in other models, so if you have a giant galactic brain idea that involves splicing together five different models then there’s nothing stopping you, except possibly your limited GPU memory. Maybe you want to merge a pretrained language model with a pretrained vision transformer to create a hybrid, like [Deepmind’s recent Flamingo](https://arxiv.org/abs/2204.14198), or you want to create the next viral text-to-image sensation like ~Dall-E Mini~ [Craiyon](https://www.craiyon.com/)? Here's an example of a hybrid model using Keras [subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models):

```py
class HybridVisionLanguageModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.language = TFAutoModel.from_pretrained("gpt2")
    self.vision = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

  def call(self, inputs):
    # I have a truly wonderful idea for this
    # which this code box is too short to contain
```

#### Philosophy #2: Loss functions are provided by default, but can be easily changed.

In Keras, the standard way to train a model is to create it, then `compile()` it with an optimizer and loss function, and finally `fit()` it. It’s very easy to load a model with transformers, but setting the loss function can be tricky - even for standard language model training, your loss function can be surprisingly non-obvious, and some hybrid models have extremely complex losses.

Our solution to that is simple: If you `compile()` without a loss argument, we’ll give you the one you probably wanted. Specifically, we’ll give you one that matches both your base model and output type - if you `compile()` a BERT-based masked language model without a loss, we’ll give you a masked language modelling loss that handles padding and masking correctly, and will only compute losses on corrupted tokens, exactly matching the original BERT training process. If for some reason you really, really don’t want your model to be compiled with any loss at all, then simply specify `loss=None` when compiling.

```py
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
model.compile(optimizer="adam")  # No loss argument!
model.fit(my_data, my_labels)
```

But also, and very importantly, we want to get out of your way as soon as you want to do something more complex. If you specify a loss argument to `compile()`, then the model will use that instead of the default loss. And, of course, if you make your own subclassed model like the `HybridVisionLanguageModel` above, then you have complete control over every aspect of the model’s functionality via the `call()` and `train_step()` methods you write.

#### ~Philosophy~ Implementation Detail #3: Labels are flexible

One source of confusion in the past was where exactly labels should be passed to the model. The standard way to pass labels to a Keras model is as a separate argument, or as part of an (inputs, labels) tuple:

```py
model.fit(inputs, labels)
```

In the past, we instead asked users to pass labels in the input dict when using the default loss. The reason for this was that the code for computing the loss for that particular model was contained in the `call()` forward pass method. This worked, but it was definitely non-standard for Keras models, and caused several issues, including incompatibilities with standard Keras metrics, not to mention some user confusion. Thankfully, this is no longer necessary. We now recommend that labels are passed in the normal Keras way, although the old method still works for backward compatibility reasons. In general, a lot of things that used to be fiddly should now “just work” for our TensorFlow models - give them a try!

#### Philosophy #4: You shouldn’t have to write your own data pipeline, especially for common tasks

In addition to `transformers`, a huge open repository of pre-trained models, there is also 🤗 `datasets`, a huge open repository of datasets - text, vision, audio and more. These datasets convert easily to TensorFlow Tensors and Numpy arrays, making it easy to use them as training data. Here’s a quick example showing us tokenizing a dataset and converting it to Numpy. As always, make sure your tokenizer matches the model you want to train with, or things will get very weird!

```py
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

dataset = load_dataset("glue", "cola")  # Simple text classification dataset
dataset = dataset["train"]  # Just take the training split for now

# Load our tokenizer and tokenize our data
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(dataset["text"], return_tensors="np", padding=True)
labels = np.array(dataset["label"]) # Label is already an array of 0 and 1

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))

model.fit(tokenized_data, labels)
```

This approach is great when it works, but for larger datasets you might find it starting to become a problem. Why? Because the tokenized array and labels would have to be fully loaded into memory, and because Numpy doesn’t handle “jagged” arrays, so every tokenized sample would have to be padded to the length of the longest sample in the whole dataset. That’s going to make your array even bigger, and all those padding tokens will slow down training too!

As a TensorFlow engineer, this is normally where you’d turn to `tf.data` to make a pipeline that will stream the data from storage rather than loading it all into memory. That’s a hassle, though, so we’ve got you. First, let’s use the `map()` method to add the tokenizer columns to the dataset. Remember that our datasets are disc-backed by default - they won’t load into memory until you convert them into arrays!

```py
def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])

dataset = dataset.map(tokenize_dataset)
```

Now our dataset has the columns we want, but how do we train on it? Simple - wrap it with a `tf.data.Dataset` and all our problems are solved - data is loaded on-the-fly, and padding is applied only to batches rather than the whole dataset, which means that we need way fewer padding tokens:

```py
tf_dataset = model.prepare_tf_dataset(
	dataset,
	batch_size=16,
	shuffle=True
)

model.fit(tf_dataset)
```
Why is [prepare_tf_dataset()](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset) a method on your model? Simple: Because your model knows which columns are valid as inputs, and automatically filters out columns in the dataset that aren't valid input names! If you’d rather have more precise control over the `tf.data.Dataset` being created, you can use the lower level [Dataset.to_tf_dataset()](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset) instead.

#### Philosophy #5: XLA is great!

[XLA](https://www.tensorflow.org/xla) is the just-in-time compiler shared by TensorFlow and JAX. It converts linear algebra code into more optimized versions that run quicker and use less memory. It’s really cool and we try to make sure that we support it as much as possible. It’s extremely important for allowing models to be run on TPU, but it offers speed boosts for GPU and even CPU as well! To use it, simply `compile()` your model with the `jit_compile=True` argument (this works for all Keras models, not just Hugging Face ones):

```py
model.compile(optimizer="adam", jit_compile=True)
```

We’ve made a number of major improvements recently in this area. Most significantly, we’ve updated our `generate()` code to use XLA - this is a function that iteratively generates text output from language models. This has resulted in massive performance improvements - our legacy TF code was much slower than PyTorch, but the new code is much faster than it, and similar to JAX in speed! For more information, please see [our blogpost about XLA generation](https://huggingface.co/blog/tf-xla-generate).

XLA is useful for things besides generation too, though! We’ve also made a number of fixes to ensure that you can train your models with XLA, and as a result our TF models have reached JAX-like speeds for tasks like language model training.

It’s important to be clear about the major limitation of XLA, though: XLA expects input shapes to be static. This means that if your task involves variable sequence lengths, you will need to run a new XLA compilation for each different input shape you pass to your model, which can really negate the performance benefits! You can see some examples of how we deal with this in our [TensorFlow notebooks](https://huggingface.co/docs/transformers/notebooks) and in the XLA generation blogpost above.

#### Philosophy #6: Deployment is just as important as training

TensorFlow has a rich ecosystem, particularly around model deployment, that the other more research-focused frameworks lack. We’re actively working on letting you use those tools to deploy your whole model for inference. We're particularly interested in supporting `TF Serving` and `TFX`. If this is interesting to you, please check out [our blogpost on deploying models with TF Serving](https://huggingface.co/blog/tf-serving-vision)!

One major obstacle in deploying NLP models, however, is that inputs will still need to be tokenized, which means it isn't enough to just deploy your model. A dependency on `tokenizers` can be annoying in a lot of deployment scenarios, and so we're working to make it possible to embed tokenization into your model itself, allowing you to deploy just a single model artifact to handle the whole pipeline from input strings to output predictions. Right now, we only support the most common models like BERT, but this is an active area of work! If you want to try it, though, you can use a code snippet like this:

```py
# This is a new feature, so make sure to update to the latest version of transformers!
# You will also need to pip install tensorflow_text

import tensorflow as tf
from transformers import TFAutoModel, TFBertTokenizer


class EndToEndModel(tf.keras.Model):
    def __init__(self, checkpoint):
        super().__init__()
        self.tokenizer = TFBertTokenizer.from_pretrained(checkpoint)
        self.model = TFAutoModel.from_pretrained(checkpoint)

    def call(self, inputs):
        tokenized = self.tokenizer(inputs)
        return self.model(**tokenized)

model = EndToEndModel(checkpoint="bert-base-cased")

test_inputs = [
    "This is a test sentence!",
    "This is another one!",
]
model.predict(test_inputs)  # Pass strings straight to model!
```

#### Conclusion: We’re an open-source project, and that means community is everything

Made a cool model? Share it! Once you’ve [made an account and set your credentials](https://huggingface.co/docs/transformers/main/en/model_sharing) it’s as easy as:

```py
model_name = "google/vit-base-patch16-224"
model = TFAutoModelForImageClassification.from_pretrained(model_name)

model.fit(my_data, my_labels)

model.push_to_hub("my-new-model")
```

You can also use the [PushToHubCallback](https://huggingface.co/docs/transformers/main_classes/keras_callbacks#transformers.PushToHubCallback) to upload checkpoints regularly during a longer training run! Either way, you’ll get a model page and an autogenerated model card, and most importantly of all, anyone else can use your model to get predictions, or as a starting point for further training, using exactly the same API as they use to load any existing model:

```py
model_name = "your-username/my-new-model"
model = TFAutoModelForImageClassification.from_pretrained(model_name)
```

I think the fact that there’s no distinction between big famous foundation models and models fine-tuned by a single user exemplifies the core belief at Hugging Face - the power of users to build great things. Machine learning was never meant to be a trickle of results from closed models held at a rarefied few companies; it should be a collection of open tools, artifacts, practices and knowledge that’s constantly being expanded, tested, critiqued and built upon - a bazaar, not a cathedral. If you hit upon a new idea, a new method, or you train a new model with great results, let everyone know!

And, in a similar vein, are there things you’re missing? Bugs? Annoyances? Things that should be intuitive but aren’t? Let us know! If you’re willing to get a (metaphorical) shovel and start fixing it, that’s even better, but don’t be shy to speak up even if you don’t have the time or skillset to improve the codebase yourself. Often, the core maintainers can miss problems because users don’t bring them up, so don’t assume that we must be aware of something! If it’s bothering you, please [ask on the forums](https://discuss.huggingface.co/), or if you’re pretty sure it’s a bug or a missing important feature, then [file an issue](https://github.com/huggingface/transformers).

A lot of these things are small details, sure, but to coin a (rather clunky) phrase, great software is made from thousands of small commits. It’s through the constant collective effort of users and maintainers that open-source software improves. Machine learning is going to be a major societal issue in the 2020s, and the strength of open-source software and communities will determine whether it becomes an open and democratic force open to critique and re-evaluation, or whether it is dominated by giant black-box models whose owners will not allow outsiders, even those whom the models make decisions about, to see their precious proprietary weights. So don’t be shy - if something’s wrong, if you have an idea for how it could be done better, if you want to contribute but don’t know where, then tell us!

<small>(And if you can make a meme to troll the PyTorch team with after your cool new feature is merged, all the better.)</small>
