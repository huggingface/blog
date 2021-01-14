---
title: Faster TensorFlow models in Hugging Face Transformers
thumbnail: https://huggingface.co/blog/assets/09_tf-serving/tf-serving_thumbnail.png
---

<h1 class="no-top-margin">Faster TensorFlow models in Hugging Face Transformers</h1>

<div class="blog-metadata">
    <small>Published X Y, 2021.</small>
    <a target="_blank" class="btn-readme" href="https://github.com/huggingface/blog/blob/master/tf-serving.md">
        <img src="/front/assets/icon-github.svg">
        Update on GitHub
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/09_tf_serving.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

The last few months, the Hugging Face team has been working hard on improving transformer’s TensorFlow models to make them more robust and faster. The recent improvements are mainly focused on two aspects:

1. Computational performance: Bert, Roberta, Electra and MPNet have been improved in order to have a much faster computation time. This gain of computational performance is noticeable for all the computational aspects: graph/eager mode, TF Serving and for CPU/GPU/TPU devices.
2. TensorFlow Serving: each of these TensorFlow model can be deployed with TensorFlow Serving to benefit of this gain of computational performance for inference.

## Computational Performance

To demonstrate the computational performance improvements, we have done a thorough benchmark where we compare BERT's performance with TensorFlow Serving of v4.2.0 to the official implementation from [Google](https://github.com/tensorflow/models/tree/master/official/nlp/bert). The benchmark has been run on a GPU V100 using a sequence length of 128:

| Batch size | Google implementation | Current master implementation   | Relative difference Google/master implem |
|:----------:|:---------------------:|:-------------------------------:|:----------------------------------------:|
|      1     |          6.7          |              6.26               |                   6.79%                  |
|      2     |          9.4          |              8.68               |                   7.96%                  |
|      4     |          14.4         |              13.1               |                   9.45%                  |
|      8     |           24          |              21.5               |                  10.99%                  |
|     16     |          46.6         |              42.3               |                   9.67%                  |
|     32     |          83.9         |              80.4               |                   4.26%                  |
|     64     |         171.5         |              156                |                   9.47%                  |
|     128    |         338.5         |              309                |                   9.11%                  |

The current implementation of Bert in master is faster than the Google implementation by up to ~10%. Apart from that it is also twice faster than the 4.1.1 release.

## TensorFlow Serving

The previous section demonstrates that the brand new Bert model got a dramatic increase of computational performance in the last version of Transformers. Now, this section focuses on a walkthrough step-by-step explanation of how to deploy this improved Bert model with TensorFlow Serving and to benefit from this computational performance in a production environment.

### What is TensorFlow Serving?

TensorFlow Serving belongs to the set of tools provided by [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx/guide/serving) that makes the task of deploying a model to a server easier than ever. TensorFlow Serving provides two APIs, one that can be called upon using HTTP requests and another one using gRPC to run inference on the server.

### What is a SavedModel?

A SavedModel contains a standalone TensorFlow model, including its weights and its architecture. It does not require the original source of the model to be run, which makes it useful for sharing or deploying with any backend that supports reading a SavedModel such as Java, Go, C++ or JavaScript among others.

### How to install TensorFlow Serving?

There are three ways to install and use TensorFlow Serving, one is through a Docker container, another one through an apt package and a last one with pip. To make things easier and compliant with all the existing OS, we will use Docker in this tutorial.

### How to create a saved model?

Saved model is the format expected by TensorFlow serving. Since Transformers` v4.2.0, creating a saved model has three additional features:

1. The sequence length can be modified freely between runs.
2. Any expected input by the model can be used to run an inference.
3. When `output_attentions` or `output_hidden_states` is set to True, the attentions or the hidden states are grouped into a single output.

Here a better idea of how looks like the inputs/outputs of a saved model for `TFBertForSequenceClassification`:

<img src="/blog/assets/09_tf_serving/new_saved_model_attns.svg" alt="New saved model with attentions" style="margin: auto; display: block; width: 260px;">

The following snippet of code shows how to use `inputs_embeds` instead of `input_ids` as input:

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

# Instanciate the model with the new serving method
model = MyOwnModel.from_pretrained("bert-base-cased")
# save it with saved_model=True in order to have a saved model version along with the h5 weights.
model.save_pretrained("my_model", saved_model=True)
```

Basically, in this example the new model directly expects the embeddings of the tokens instead of the token ids as input. The serving method has to be overridden by the new `input_signature`. See the [official documentation](https://www.tensorflow.org/api_docs/python/tf/function#args_1) to know more about the `input_signature` argument. The `serving` method is used to define how will behave a saved model when deployed with TensorFlow Serving. Now the saved model looks like as expected, see the new `inputs_embeds` input:

<img src="/blog/assets/09_tf_serving/embeds_saved_model.svg" alt="Saved model with inputs embeds" style="margin: auto; display: block; width: 260px;">

## How to deploy and use a saved model?

Let’s see step by step how to deploy and use a sentiment classification Bert model.

Step 1: create a saved model. To create a saved model we load a PyTorch model called `nateraw/bert-base-uncased-imdb`
trained on the IMBD dataset:

```python
from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-imdb", from_pt=True)
# the saved_model parameter is a flag to create a saved model version of the model in same time than the h5 weights
model.save_pretrained("my_model", saved_model=True)
```

Step 2: create a Docker container containing the saved model and run it:

```
# pull the TensorFlow serving Docker image for CPU
# for GPU replace serving by serving:latest-gpu
docker pull tensorflow/serving

# run a serving image as a daemon named serving_base
docker run -d --name serving_base tensorflow/serving

# copy the newly created saved model into the serving_base container's models folder
docker cp my_model/saved_model serving_base:/models/bert

# commit the container that serves the model by changing MODEL_NAME to match the model's name (here bert)
# the name (bert) corresponds to the name we want to give to our saved model
docker commit --change "ENV MODEL_NAME bert" serving_base my_bert_model

# kill the serving_base image ran as a daemon because we don't need it anymore
docker kill serving_base

# Run the image to serve our saved model as a daemon and we map the ports 8501 (REST API)
# and 8500 (gRPC API) in the container to the host and we name the the container "bert".
docker run -d -p 8501:8501 -p 8500:8500 --name bert my_bert_model
```

Step 3: Query the model through the REST API:

```python
from transformers import BertTokenizerFast, BertConfig
import requests
import json
import numpy as np

sentence = "I love the new TensorFlow update in transformers."
# Load the corresponding tokenizer of our saved model
tokenizer = BertTokenizerFast.from_pretrained("nateraw/bert-base-uncased-imdb")
# Load the model config of our saved model
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
# The returned results are probabilities, that can be positive/negative hence we take their absolute value
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
Thanks to the last updates applied on the TensorFlow models in transformers, one can now easily deploy its models in production in an efficient way. One of the next steps we are thinking about is to directly integrate the preprocessing part inside the saved model to make things even easier.
