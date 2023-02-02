---
title: "How to train your model dynamically using adversarial data"
thumbnail: /blog/assets/88_mnist_adversarial/mnist-adversarial.png
authors:
- user: chrisjay
---

<h1>
    How to train your model dynamically using adversarial data
</h1>

{blog_metadata}

{authors}

##### What you will learn here
- 💡the basic idea of dynamic adversarial data collection and why it is important.
- ⚒ how to collect adversarial data dynamically and train your model on them - using an MNIST handwritten digit recognition task as an example. 


## Dynamic adversarial data collection (DADC)
Static benchmarks, while being a widely-used way to evaluate your model's performance, are fraught with many issues: they saturate, have biases or loopholes, and often lead researchers to chase increment in metrics instead of building trustworthy models that can be used by humans <sup>[1](https://dynabench.org/about)</sup>.

Dynamic adversarial data collection (DADC) holds great promise as an approach to mitigate some of the issues of static benchmarks. In DADC, humans create examples to _fool_ state-of-the-art (SOTA) models. This process offers two benefits: 
1. it allows users to gauge how robust their models really are;
2. it yields data that may be used to further train even stronger models. 
 
This process of fooling and training the model on the adversarially collected data is repeated over multiple rounds leading to a more robust model that is aligned with humans<sup>[1](https://aclanthology.org/2022.findings-acl.18.pdf) </sup>.

## Training your model dynamically using adversarial data
 
Here I will walk you through dynamically collecting adversarial data from users and training your model on them - using the MNIST handwritten digit recognition task.  
 
In the MNIST handwritten digit recognition task, the model is trained to predict the number given a `28x28` grayscale image input of the handwritten digit (see examples in the figure below). The numbers range from 0 to 9. 

![](https://i.imgur.com/1OiMHhE.png)

> Image source: [mnist | Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/mnist) 

This task is widely regarded as the _hello world_ of computer vision and it is very easy to train models that achieve high accuracy on the standard (and static) benchmark test set. Nevertheless, it has been shown that these SOTA models still find it difficult to predict the correct digits when humans write them (and give them as input to the model): researchers opine that this is largely because the static test set does not adequately represent the very diverse ways humans write. Therefore humans are needed in the loop to provide the models with _adversarial_ samples which will help them generalize better.

This walkthrough will be divided into the following sections:
1. Configuring your model
2. Interacting with your model
3. Flagging your model
4. Putting it all together

### Configuring your model
First of all, you need to define your model architecture. My simple model architecture below is made up of two convolutional networks connected to a 50 dimensional fully connected layer and a final layer for the 10 classes. Finally, we use the softmax activation function to turn the model's output into a probability distribution over the classes.
```python
# Adapted from: https://nextjournal.com/gkoehler/pytorch-mnist
class MNIST_Model(nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```

Now that you have defined the structure of your model, you need to train it on the standard MNIST train/dev dataset.

### Interacting with your model

At this point we assume you have your trained model. Although this model is trained, we aim to make it robust using human-in-the-loop adversarial data. For that, you need a way for users to interact with it: specifically you want users to be able to write/draw numbers from 0-9 on a canvas and have the model try to classify it. You can do all that with [🤗 Spaces](https://huggingface.co/spaces) which allows you to quickly and easily build a demo for your ML models. Learn more about Spaces and how to build them [here](https://huggingface.co/spaces/launch). 

Below is a simple Space to interact with the `MNIST_Model` which I trained for 20 epochs (achieved 89% accuracy on the test set). You draw a number on the white canvas and the model predicts the number from your image. The full Space can be accessed [here](https://huggingface.co/spaces/chrisjay/simple-mnist-classification). Try to fool this model😁. Use your funniest handwriting; write on the sides of the canvas; go wild!

<iframe src="https://chrisjay-simple-mnist-classification.hf.space" frameBorder="0" width="100%" height="700px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

### Flagging your model
  
Were you able to fool the model above?😀 If yes, then it's time to _flag_ your adversarial example. Flagging entails:
1. saving the adversarial example to a dataset
2. training the model on the adversarial examples after some threshold samples have been collected.
3. repeating steps 1-2 a number of times. 

I have written a custom `flag` function to do all that. For more details feel free to peruse the full code [here](https://huggingface.co/spaces/chrisjay/mnist-adversarial/blob/main/app.py#L314). 

>Note: Gradio has a built-in flaggiing callback that allows you easily flag adversarial samples of your model. Read more about it [here](https://gradio.app/using_flagging/).

### Putting it all together

The final step is to put all the three components (configuring the model, interacting with it and flagging it) together as one demo Space! To that end, I have created the [MNIST Adversarial](https://huggingface.co/spaces/chrisjay/mnist-adversarial) Space for dynamic adversarial data collection for the MNIST handwritten recognition task. Feel free to test it out below.

<iframe src="https://chrisjay-mnist-adversarial.hf.space" frameBorder="0" width="100%" height="1400px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>


## Conclusion


Dynamic Adversarial Data Collection (DADC) has been gaining traction in the machine learning community as a way to gather diverse non-saturating human-aligned datasets, and improve model evaluation and task performance. By dynamically collecting human-generated adversarial data with models in the loop, we can improve the generalization potential of our models. 

This process of fooling and training the model on the adversarially collected data should be repeated over multiple rounds<sup>[1](https://aclanthology.org/2022.findings-acl.18.pdf)</sup>. [Eric Wallace et al](https://aclanthology.org/2022.findings-acl.18), in their experiments on natural language inference tasks, show that while in the short term standard non-adversarial data collection performs better, in the long term however dynamic adversarial data collection leads to the highest accuracy by a noticeable margin. 

Using the [🤗 Spaces](https://huggingface.co/spaces), it becomes relatively easy to build a platform to dynamically collect adversarial data for your model and train on them. 