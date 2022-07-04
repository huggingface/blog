# How to train a model using community adversarial data
> Still working on the title...feel free to suggest alternatives.

### What you will learn here
- 💡the idea of dynamic adversarial data collection (DADC) and why it is important.
- ⚒how to collect adversarial data dynamically and train your model on them - using an MNIST handwritten digit recognition task as an example. 


### Dynamic adversarial data collection (DADC)
Static benchmarks, while being a good and widely-used way to understand your model's performance, are fraught with many issues: they saturate, have biases or loopholes, and usually leads to researchers chasing increment in metric instead of trustworthy models that can be used by humans <sup>[dynabench](https://dynabench.org/about)</sup>.

Dynamic adversarial data collection (DADC) holds great promise as an approach to mitigate some of the issues of static benchmarks. In DADC, humans create examples to fool state-of-the-art (SOTA) models but are answerable by humans. This offers two benefits: 
1. it allows us to gauge how good our current SOTA methods really are;
2. it yields data that may be used to further train even stronger SOTA models. 
 
The process (of fooling and training the model on the adversarially collected data) is repeated over multiple rounds leading to a more robust model that is algned with humans.

Researchers<sup>[1](https://aclanthology.org/2022.findings-acl.18.pdf) </sup>argue that running DADC over many training rounds (as against only 1-3 rounds) of the model maximizes its training-time benefits, as the different rounds can together cover many of the task-relevant phenomena.


## How to do DADC with your model (using the MNIST as an example)
> Still working on the title...feel free to suggest alternatives.
> 
In this short guide, I will walk you through creating an adversarial space for your model. I will use an MNIST classification model as a simple example. MNIST is a popular image classsification task where the input is a 28x28 black and white image of a number from 0-9 and the model learns to predict the number. The space can be found [here](https://huggingface.co/spaces/chrisjay/mnist-adversarial). 


#### The model
The first part to start is with the model. My simple model architecture is made up of two convolutional networks connected to a `50 dim` fully connected layer and the final layer for the 10 classes.
```python=
# Code adapted from: https://nextjournal.com/gkoehler/pytorch-mnist
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


#### The interface

Now you need a way for users to interact with your model: specifically you want them to be able to draw a number, submit it to the model and see the model's prediction. You can do all that with [Gradio](https://gradio.app/docs/) 😍. 

Below is a simple interactive interface I built to query a randomly initialized `MNIST_Model`. The full space is [here](https://huggingface.co/spaces/chrisjay/simple-mnist-classification). 

<iframe src="https://hf.space/embed/chrisjay/simple-mnist-classification/+" frameBorder="0" width="1000px" height="660px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>
Currently the model is not trained. Therefore we expect inaccurate predictions. We will proceed to train this model with adversarially collected data.

### Flagging the model
> Work on this title
  
Right now, we have an interface by which users can test our model and try to fool it. The last thing needed is to handle what happens when the user is able to fool the model (draw a number which the model __incorrectly__ predicts). This is called __flagging__.

There are a couple of ways we can go about flagging an instance:
1. __using the Gradio default flag__
[This guide](https://gradio.app/using_flagging/) explains all there is to know about using flagging in Gradio. 
2. __writing your own custom flag function or class__
Now this is the interesting part. The default flagging opportunities from Gradio may not have all the functionalities you want. No worries, you can easily create your own flagging.

All you need is a `Flag` button and then you define a `flag` function that will be called once the `Flag` button is clicked. Inside the `flag` function, you can define what you want to happen.

Here is an explanation of what my `flag` function does. For more details see the full code [here](https://huggingface.co/spaces/chrisjay/mnist-adversarial/blob/main/app.py#L316). 

```python=
def flag(input_image,correct_result,adversarial_number):
     """
    It takes in an image, the correct result, and the number
    of adversarial images that have been uploaded so far.
    It saves the image and metadata to a local directory, uploads
    the image and metadata to the hub, and then pulls the data from
    the hub to the local directory. 
    
    If the number of images in the local directory is divisible by
    the TRAIN_CUTOFF, then it trains the model on the adversarial data.
    
    :param input_image: The adversarial image that you want to save
    :param correct_result: The correct number that the image represents
    :param adversarial_number: This is the number of adversarial examples
    that have been uploaded to the dataset
    
    :return: The output is the output of the flag function.
    """

```

Basically for this project you want to be able to:
1. Save the flagged instance (input image, correct target and maybe model's wrong predicton, if needed) to a Hugging Face dataset. 
2. Retrain the model on the dataset periodically (for me I do that after 10 adversaial samples have been added).



### Putting it all together
> Still working on it...

<iframe src="https://hf.space/embed/chrisjay/mnist-adversarial/+" frameBorder="0" width="1000px" height="660px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>