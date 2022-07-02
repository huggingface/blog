# How to create an adversarial space
> Still working on the title...feel free to suggest alternatives.

### What you will learn in this blog
- the idea of DADC and why it is important
- how to do DADC with your model using MNIST as an example
    - the model
    - the space and code for input-output
    - adding the flag button
        - what the flag button does
        - using the gradio hf_writer
        - writing your own custom flag 


> Add intro to DADC; what it is; why it is important. 

## How to do DADC with your model (using MNIST as an example)

In this short guide, I will walk you through creating an adversarial space for your model. I will use an MNIST classification model as a simple example. MNIST is a popular image classsification task where the input is a 28x28 black and white image of a number from 0-9 and the model learns to predict the number. The space can be found [here](https://huggingface.co/spaces/chrisjay/mnist-adversarial). 


#### The model
The first part to start is with the model. My simple model architecture is made up of two convolutional networks connected to a `50 dim` fully connected layer and the final layer for the 10 classes.
```python=
# Source: https://nextjournal.com/gkoehler/pytorch-mnist
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

Now you need a way for users to interact with your model: specifically you want them to be able to draw a number, submit it to the model and see the model's prediction. You can do all that with Gradio. 

Below is a simple interactive interface to query `MNIST_Model`.
> Put interactive interface here

Currently the model is not trained. Therefore we expect many in accurate predictions. You can use a model you have trained.

Right now, we have an interface by which users can test our model and try to fool it. The last thing needed is to handle what happens when the user is able to fool the model (draw a number which the model __incorrectly__ predicts). This is called __flagging__.

There are a couple of ways we can go about flagging an instance:
1. __using the Gradio default flag__
[This guide](https://gradio.app/using_flagging/) explains all there is to know about using flagging in Gradio. 
2. __writing your own custom flag function or class__
Now this is the interesting part. The default flagging opportunities from Gradio may not have all the functionalities you want. No worries, you can easily create your own flagging.

All you need is a `Flag` button and then you define a `flag` function that will be called once the `Flag` button is clicked. Inside the `flag` function, you can define what you want to happen.

Here is what my `flag` function looks like

```python=
def flag(input_image,correct_result,adversarial_number):

    adversarial_number = 0 if None else adversarial_number

    metadata_name = get_unique_name()
    SAVE_FILE_DIR = os.path.join(LOCAL_DIR,metadata_name)
    os.makedirs(SAVE_FILE_DIR,exist_ok=True)
    image_output_filename = os.path.join(SAVE_FILE_DIR,'image.png')
    try:
        input_image.save(image_output_filename)
    except Exception:
        raise Exception(f"Had issues saving PIL image to file")    

    # Write metadata.json to file
    json_file_path = os.path.join(SAVE_FILE_DIR,'metadata.jsonl')
    metadata= {'id':metadata_name,'file_name':'image.png',
                'correct_number':correct_result
                }
    
    dump_json(metadata,json_file_path)  
        
    # Simply upload the image file and metadata using the hub's upload_file
    # Upload the image
    repo_image_path = os.path.join(REPOSITORY_DIR,os.path.join(metadata_name,'image.png'))
    
    _ = upload_file(path_or_fileobj = image_output_filename,
                path_in_repo =repo_image_path,
                repo_id=f'chrisjay/{HF_DATASET}',
                repo_type='dataset',
                token=HF_TOKEN
            ) 

    # Upload the metadata
    repo_json_path = os.path.join(REPOSITORY_DIR,os.path.join(metadata_name,'metadata.jsonl'))
    _ = upload_file(path_or_fileobj = json_file_path,
                path_in_repo =repo_json_path,
                repo_id=f'chrisjay/{HF_DATASET}',
                repo_type='dataset',
                token=HF_TOKEN
            )        
    adversarial_number+=1
    output = f'<div> ✔ ({adversarial_number}) Successfully saved your adversarial data. </div>'
    repo.git_pull()
    length_of_dataset = len([f for f in os.scandir("./data_mnist/data")])
    test_metric = f"<html> {DEFAULT_TEST_METRIC} </html>"
    if length_of_dataset % TRAIN_CUTOFF ==0:
        test_metric_ = train_and_test()
        test_metric = f"<html> {test_metric_} </html>"
        output = f'<div> ✔ ({adversarial_number}) Successfully saved your adversarial data and trained the model on adversarial data! </div>'
    return output,adversarial_number
```
> Consider removing the code block above. It is too long and does not explain much.

Basically for this project you want to be able to:
1. Save the flagged instance (input image, correct target and maybe model's wrong predicton, if needed) to a Hugging Face dataset. 
2. Retrain the model on the dataset periodically (for me I do that after 10 adversaial samples have been added).