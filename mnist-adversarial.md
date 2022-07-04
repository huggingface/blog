# How to train a model using community adversarial data
> Still working on the title...feel free to suggest alternatives.

##### What you will learn here
- 💡the idea of dynamic adversarial data collection and why it is important.
- ⚒how to collect adversarial data dynamically and train your model on them - using an MNIST handwritten digit recognition task as an example. 


## Dynamic adversarial data collection (DADC)
Static benchmarks, while being a widely-used way to evaluate your model's performance, are fraught with many issues: they saturate, have biases or loopholes, and usually leads to researchers chasing increment in metric instead of trustworthy models that can be used by humans <sup>[dynabench](https://dynabench.org/about)</sup>.

Dynamic adversarial data collection (DADC) holds great promise as an approach to mitigate some of the issues of static benchmarks. In DADC, humans create examples to fool state-of-the-art (SOTA) models but are answerable by humans. This offers two benefits: 
1. it allows us to gauge how good our current SOTA methods really are;
2. it yields data that may be used to further train even stronger SOTA models. 
 
The process (of fooling and training the model on the adversarially collected data) is repeated over multiple rounds leading to a more robust model that is aligned with humans<sup>[1](https://aclanthology.org/2022.findings-acl.18.pdf) </sup>.

## Creating a simple DADC space on Gradio
> Still working on the title...feel free to suggest alternatives.
 
In the MNIST handwritten digit recognition task, the model is trained to predict the number given a `28x28` grayscale image input of the handwirtten digit. The numbers are from 0 to 9. This task is widely regarded as the _hello world_ of computer vision and there are many SOTA models that achieve >95% accuracy on a benchmark test set for the task.

Nevertheless, it has been shown that these SOTA models still find it difficult to correctly predict human handwriting of the digits: largely due to the distrbution shift in the way humans actually write and the data in the test set.

Therefore these models need humans in the loop to provide them with adversarial data in order to make them more robust and generalize better.

In this short guide, I will walk you through creating an adversarial space for your MNIST model. This space will enable users to try to fool your model by providing examples that are hard for the model. Finally, the model will be trained on these adversarial samples after a certain number of them have been collected. The cycle continues for as many rounds as possible. Researchers<sup>[1](https://aclanthology.org/2022.findings-acl.18.pdf) </sup>have shown that to get the best out of DADC, one should run it for many rounds (as against only 1-3 rounds).

This part will be divided into the following sections:
1. Configuring the model
2. Interacting with the model
3. Flagging the model
4. Putting it all together

### Configuring the model
First of all, you need to define your model. My simple model architecture below is made up of two convolutional networks connected to a 50 dimensional fully connected layer and the final layer for the 10 classes.
```python=
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

Now that you have defined your model, you can go ahead with training your model on the standard MNIST train/dev dataset or use a pretrained model. 


### Interacting with the model

Now that you have your model, you need a way for users to interact with it: specifically you want them to be able to draw a number and see the model's prediction. You can do all that with [🤗 Spaces](https://huggingface.co/spaces). 

Below is a simple space to interact with the `MNIST_Model` which I trained up till 98% accuracy on the test set. You draw a number and see the model's prediction. The space is [here](https://huggingface.co/spaces/chrisjay/simple-mnist-classification). Try to fool this model😁. Use your funniest handwriting; write on the sides of the canvas; go wild!

<iframe src="https://hf.space/embed/chrisjay/simple-mnist-classification/+" frameBorder="0" width="100%" height="360px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>
Currently the model is not trained. Therefore we expect inaccurate predictions. We will proceed to train this model with adversarially collected data.

### Flagging the model
  
Were you able to fool the model above?😀 Now it's time to _flag_ your adversarial example. __Flagging__ refers to the processes that happen when a user is able to fool the model. Generally flagging involves saving the adversarial example, and other custom things you want to implement. For our flagging, we will define a function to: 
1. save the adversarial example to a dataset
2. train the model on the adversarial example after some threshold samples have been collected.
3. repeat 1-2 many times. 

>Note: Gradio has a built-in flaggiing system that allows you easily flag adversarial samples of your model. Read more about it [here](https://gradio.app/using_flagging/).

Here is an explanation of what my `flag` function does. For more details feel free to peruse the full code [here](https://huggingface.co/spaces/chrisjay/mnist-adversarial/blob/main/app.py#L314). 

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

### Putting it all together

The final step is to put all the components together into one space! I have created the [MNIST Adversarial](https://huggingface.co/spaces/chrisjay/mnist-adversarial) space here for dynamic adversarial data collection for the MNIST handwritten recognition task. 

<iframe src="https://hf.space/embed/chrisjay/mnist-adversarial/+" frameBorder="0" width="1000px" height="660px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>


## Conclusion