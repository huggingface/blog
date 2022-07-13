---
title: Faster TensorFlow models in Hugging Face Transformers
thumbnail: /blog/assets/10_tf-serving/thumbnail.png
---

<h1>Faster TensorFlow models in Hugging Face Transformers</h1>

<div class="blog-metadata">
    <small>Published January 26, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/tf-serving.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/jplu">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1584609257509-5df8987fda6d0311fd3d540d.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>jplu</code>
            <span class="fullname">Julien Plu</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/10_tf_serving.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

In the last few months, the Hugging Face team has been working hard on improving Transformers’ TensorFlow models to make them more robust and faster. The recent improvements are mainly focused on two aspects:

1. Computational performance: BERT, RoBERTa, ELECTRA and MPNet have been improved in order to have a much faster computation time. This gain of computational performance is noticeable for all the computational aspects: graph/eager mode, TF Serving and for CPU/GPU/TPU devices.
2. TensorFlow Serving: each of these TensorFlow model can be deployed with TensorFlow Serving to benefit from this gain of computational performance for inference.

## Computational Performance

To demonstrate the computational performance improvements, we have done a thorough benchmark where we compare BERT's performance with TensorFlow Serving of v4.2.0 to the official implementation from [Google](https://github.com/tensorflow/models/tree/master/official/nlp/bert). The benchmark has been run on a GPU V100 using a sequence length of 128 (times are in millisecond):

| Batch size | Google implementation | v4.2.0 implementation | Relative difference Google/v4.2.0 implem |
|:----------:|:---------------------:|:---------------------:|:----------------------------------------:|
|      1     |          6.7          |         6.26          |                   6.79%                  |
|      2     |          9.4          |         8.68          |                   7.96%                  |
|      4     |         14.4          |         13.1          |                   9.45%                  |
|      8     |           24          |         21.5          |                  10.99%                  |
|     16     |         46.6          |         42.3          |                   9.67%                  |
|     32     |         83.9          |         80.4          |                   4.26%                  |
|     64     |        171.5          |          156          |                   9.47%                  |
|    128     |        338.5          |          309          |                   9.11%                  |

The current implementation of Bert in v4.2.0 is faster than the Google implementation by up to ~10%. Apart from that it is also twice as fast as the implementations in the 4.1.1 release.

## TensorFlow Serving

The previous section demonstrates that the brand new Bert model got a dramatic increase in computational performance in the last version of Transformers. In this section, we will show you step-by-step how to deploy a Bert model with TensorFlow Serving to benefit from the increase in computational performance in a production environment.

### What is TensorFlow Serving?

TensorFlow Serving belongs to the set of tools provided by [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx/guide/serving) that makes the task of deploying a model to a server easier than ever. TensorFlow Serving provides two APIs, one that can be called upon using HTTP requests and another one using gRPC to run inference on the server.

### What is a SavedModel?

A SavedModel contains a standalone TensorFlow model, including its weights and its architecture. It does not require the original source of the model to be run, which makes it useful for sharing or deploying with any backend that supports reading a SavedModel such as Java, Go, C++ or JavaScript among others. The internal structure of a SavedModel is represented as such:
```
savedmodel
    /assets
        -> here the needed assets by the model (if any)
    /variables
        -> here the model checkpoints that contains the weights
   saved_model.pb -> protobuf file representing the model graph
```

### How to install TensorFlow Serving?

There are three ways to install and use TensorFlow Serving:
- through a Docker container, 
- through an apt package,
- or using [pip](https://pypi.org/project/pip/). 

To make things easier and compliant with all the existing OS, we will use Docker in this tutorial.

### How to create a SavedModel?

SavedModel is the format expected by TensorFlow Serving. Since Transformers v4.2.0, creating a SavedModel has three additional features:

1. The sequence length can be modified freely between runs.
2. All model inputs are available for inference.
3. `hidden states` or `attention` are now grouped into a single output when returning them with `output_hidden_states=True` or `output_attentions=True`.

Below, you can find the inputs and outputs representations of a `TFBertForSequenceClassification` saved as a TensorFlow SavedModel:

```
The given SavedModel SignatureDef contains the following input(s):
  inputs['attention_mask'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_attention_mask:0
  inputs['input_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_input_ids:0
  inputs['token_type_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_token_type_ids:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['attentions'] tensor_info:
      dtype: DT_FLOAT
      shape: (12, -1, 12, -1, -1)
      name: StatefulPartitionedCall:0
  outputs['logits'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 2)
      name: StatefulPartitionedCall:1
Method name is: tensorflow/serving/predict
```

To directly pass `inputs_embeds` (the token embeddings) instead of `input_ids` (the token IDs) as input, we need to subclass the model to have a new serving signature. The following snippet of code shows how to do so:

```python
from transformers import TFBertForSequenceClassification
import tensorflow as tf

# Creation of a subclass in order to define a new serving signature
class MyOwnModel(TFBertForSequenceClassification):
    # Decorate the serving method with the new input_signature
    # an input_signature represents the name, the data type and the shape of an expected input
    @tf.function(input_signature=[{
        "inputs_embeds": tf.TensorSpec((None, None, 768), tf.float32, name="inputs_embeds"),
        "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
        "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
    }])
    def serving(self, inputs):
        # call the model to process the inputs
        output = self.call(inputs)

        # return the formated output
        return self.serving_output(output)

# Instantiate the model with the new serving method
model = MyOwnModel.from_pretrained("bert-base-cased")
# save it with saved_model=True in order to have a SavedModel version along with the h5 weights.
model.save_pretrained("my_model", saved_model=True)
```

The serving method has to be overridden by the new `input_signature` argument of the `tf.function` decorator. See the [official documentation](https://www.tensorflow.org/api_docs/python/tf/function#args_1) to know more about the `input_signature` argument. The `serving` method is used to define how will behave a SavedModel when deployed with TensorFlow Serving. Now the SavedModel looks like as expected, see the new `inputs_embeds` input:

```
The given SavedModel SignatureDef contains the following input(s):
  inputs['attention_mask'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_attention_mask:0
  inputs['inputs_embeds'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, -1, 768)
      name: serving_default_inputs_embeds:0
  inputs['token_type_ids'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: serving_default_token_type_ids:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['attentions'] tensor_info:
      dtype: DT_FLOAT
      shape: (12, -1, 12, -1, -1)
      name: StatefulPartitionedCall:0
  outputs['logits'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 2)
      name: StatefulPartitionedCall:1
Method name is: tensorflow/serving/predict
```

## How to deploy and use a SavedModel?

Let’s see step by step how to deploy and use a BERT model for sentiment classification.

### Step 1

Create a SavedModel. To create a SavedModel, the Transformers library lets you load a PyTorch model called `nateraw/bert-base-uncased-imdb` trained on the IMBD dataset and convert it to a TensorFlow Keras model for you:

```python
from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-imdb", from_pt=True)
# the saved_model parameter is a flag to create a SavedModel version of the model in same time than the h5 weights
model.save_pretrained("my_model", saved_model=True)
```

### Step 2

Create a Docker container with the SavedModel and run it. First, pull the TensorFlow Serving Docker image for CPU (for GPU replace serving by serving:latest-gpu):
``` 
docker pull tensorflow/serving
```

Next, run a serving image as a daemon named serving_base:
```
docker run -d --name serving_base tensorflow/serving
```

copy the newly created SavedModel into the serving_base container's models folder:
```
docker cp my_model/saved_model serving_base:/models/bert
```

commit the container that serves the model by changing MODEL_NAME to match the model's name (here `bert`), the name (`my_bert_model`) corresponds to the name we want to give to our SavedModel:
```
docker commit --change "ENV MODEL_NAME bert" serving_base my_bert_model
```

and kill the serving_base image ran as a daemon because we don't need it anymore:
```
docker kill serving_base
```

Finally, Run the image to serve our SavedModel as a daemon and we map the ports 8501 (REST API), and 8500 (gRPC API) in the container to the host and we name the the container `bert`.
```
docker run -d -p 8501:8501 -p 8500:8500 --name bert my_bert_model
```

### Step 3

Query the model through the REST API:

```python
from transformers import BertTokenizerFast, BertConfig
import requests
import json
import numpy as np

sentence = "I love the new TensorFlow update in transformers."

# Load the corresponding tokenizer of our SavedModel
tokenizer = BertTokenizerFast.from_pretrained("nateraw/bert-base-uncased-imdb")

# Load the model config of our SavedModel
config = BertConfig.from_pretrained("nateraw/bert-base-uncased-imdb")

# Tokenize the sentence
batch = tokenizer(sentence)

# Convert the batch into a proper dict
batch = dict(batch)

# Put the example into a list of size 1, that corresponds to the batch size
batch = [batch]

# The REST API needs a JSON that contains the key instances to declare the examples to process
input_data = {"instances": batch}

# Query the REST API, the path corresponds to http://host:port/model_version/models_root_folder/model_name:method
r = requests.post("http://localhost:8501/v1/models/bert:predict", data=json.dumps(input_data))

# Parse the JSON result. The results are contained in a list with a root key called "predictions"
# and as there is only one example, takes the first element of the list
result = json.loads(r.text)["predictions"][0]

# The returned results are probabilities, that can be positive or negative hence we take their absolute value
abs_scores = np.abs(result)

# Take the argmax that correspond to the index of the max probability.
label_id = np.argmax(abs_scores)

# Print the proper LABEL with its index
print(config.id2label[label_id])
```

This should return POSITIVE. It is also possible to pass by the gRPC (google Remote Procedure Call) API to get the same result:

```python
from transformers import BertTokenizerFast, BertConfig
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

sentence = "I love the new TensorFlow update in transformers."
tokenizer = BertTokenizerFast.from_pretrained("nateraw/bert-base-uncased-imdb")
config = BertConfig.from_pretrained("nateraw/bert-base-uncased-imdb")

# Tokenize the sentence but this time with TensorFlow tensors as output already batch sized to 1. Ex:
# {
#    'input_ids': <tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[  101, 19082,   102]])>,
#    'token_type_ids': <tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[0, 0, 0]])>,
#    'attention_mask': <tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[1, 1, 1]])>
# }
batch = tokenizer(sentence, return_tensors="tf")

# Create a channel that will be connected to the gRPC port of the container
channel = grpc.insecure_channel("localhost:8500")

# Create a stub made for prediction. This stub will be used to send the gRPC request to the TF Server.
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create a gRPC request made for prediction
request = predict_pb2.PredictRequest()

# Set the name of the model, for this use case it is bert
request.model_spec.name = "bert"

# Set which signature is used to format the gRPC query, here the default one
request.model_spec.signature_name = "serving_default"

# Set the input_ids input from the input_ids given by the tokenizer
# tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
request.inputs["input_ids"].CopyFrom(tf.make_tensor_proto(batch["input_ids"]))

# Same with attention mask
request.inputs["attention_mask"].CopyFrom(tf.make_tensor_proto(batch["attention_mask"]))

# Same with token type ids
request.inputs["token_type_ids"].CopyFrom(tf.make_tensor_proto(batch["token_type_ids"]))

# Send the gRPC request to the TF Server
result = stub.Predict(request)

# The output is a protobuf where the only one output is a list of probabilities
# assigned to the key logits. As the probabilities as in float, the list is
# converted into a numpy array of floats with .float_val
output = result.outputs["logits"].float_val

# Print the proper LABEL with its index
print(config.id2label[np.argmax(np.abs(output))])
```

## Conclusion
Thanks to the last updates applied on the TensorFlow models in transformers, one can now easily deploy its models in production using TensorFlow Serving. One of the next steps we are thinking about is to directly integrate the preprocessing part inside the SavedModel to make things even easier.
