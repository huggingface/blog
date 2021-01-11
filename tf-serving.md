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

For a few months the Hugging Face team has been working hard on improving transformer’s TensorFlow models to make them more robust and faster. The recent improvements are mainly focused on two aspects:

1. Computational performance: Bert, Roberta, Electra and MPNet have been improved in order to have a much faster computation time. This gain of computational performance is noticeable for all the computational aspects: graph/eager mode, TF Serving and for CPU/GPU/TPU devices.
2. Serving: each TensorFlow model now benefits from a proper serving.

## Computational Performance

To demonstrate the better computational performance we have done a thorough benchmark where we compare the Bert’s performance with TF Serving of master to the 4.1.1 release and the official implementation from [Google](https://github.com/tensorflow/models/tree/master/official/nlp/bert). The benchmark has been run on a GPU V100 using a sequence length of 128:

| Batch size | v4.1.1 implementation | Google implementation | Current master implementation   | Relative difference 4.1.1/master implem | Relative difference Google/master implem |
|:----------:|:---------------------:|:---------------------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|
|      1     |          21.3         |          6.7          |              6.26               |                 109.14%                 |                   6.79%                  |
|      2     |          24.2         |          9.4          |              8.68               |                  94.4%                  |                   7.96%                  |
|      4     |          28.1         |          14.4         |              13.1               |                  72.82%                 |                   9.45%                  |
|      8     |          36.9         |           24          |              21.5               |                  52.74%                 |                  10.99%                  |
|     16     |          58.6         |          46.6         |              42.3               |                  32.31%                 |                   9.67%                  |
|     32     |          94.7         |          83.9         |              80.4               |                  16.33%                 |                   4.26%                  |
|     64     |         171.4         |         171.5         |              156                |                  9.41%                  |                   9.47%                  |
|     128    |         324.5         |         338.5         |              309                |                  4.89%                  |                   9.11%                  |

The current implementation of Bert in master is faster than the 4.1.1 release implementation by up to ~100% and the Google implementation by up to ~10%.

## TensorFlow Serving

Now that we've seen the dramatic increase in computational performance, let's walk through a step-by-step explanation of how to deploy TFBert with TF Serving.

### What is TensorFlow Serving?

TensorFlow Serving belongs to the set of tools provided by [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx/guide/serving) that makes the task of deploying a model to a server easier than ever. TensorFlow Serving provides two APIs, one that can be called upon using HTTP requests and another one using gRPC to run inference on the server.

### How to install TensorFlow Serving?

There are three ways to install and use TensorFlow Serving, one is through a Docker container, another one through an apt package and a last one with pip. To make things easier and compliant with all the existing OS, we will use Docker in this tutorial.

### How to create a saved model?

Saved model is the format expected by TensorFlow serving. Up to the 4.1.1 release of Transformers, creating a saved model had three main issues:

1. The length of the sequences was fixed to 5.
2. One was able to only pass the input_ids to the model.
3. When `output_attentions` or `output_hidden_states` was set to True, the output contained as many outputs as the number of attentions or hidden states, instead of being grouped into a single output.

As stated above, here a screenshot of how looked like a saved model of a `TFBertForSequenceClassification` with `output_attentions=True`, with an input length fixed to 5 and 12 attentions outputs:

<img src="/blog/assets/09_tf_serving/old_saved_model_attns.svg" alt="Old saved model with attentions" style="margin: auto; display: block; width: 260px;">

With the master implementation the three main issues are now fixed, namely 1) the variable sequence length, 2) multiple inputs are accepted and 3) the attentions are grouped. As shown in this screenshot:

<img src="/blog/assets/09_tf_serving/new_saved_model_attns.svg" alt="New saved model with attentions" style="margin: auto; display: block; width: 260px;">

It is possible to create a different list of inputs than the default by extending the model. For example by using `input_embeds` instead of `input_ids`. To do so, the serving method has to be overridden by the expected `input_signature`. The `serving` method is used to define how will behave a saved model when deployed with TF Serving. The two calls inside this methods are `output=self.call(inputs)` and `return self.serving_output(output)`. They are both mandatory and must not be modified because they respectively represents the model’s serving behavior.

```python
from transformers import TFBertForSequenceClassification
import tensorflow as tf

class MyOwnModel(TFBertForSequenceClassification):
    @tf.function(input_signature=[{
        "inputs_embeds": tf.TensorSpec((None, None, 768), tf.float32, name="inputs_embeds"),
        "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
        "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
    }])
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)

model = MyOwnModel.from_pretrained("bert-base-cased")
model.save_pretrained("my_model", saved_model=True)
```

And now the saved model looks like as expected, see the `inputs_embeds`:

<img src="/blog/assets/09_tf_serving/embeds_saved_model.svg" alt="Saved model with inputs embeds" style="margin: auto; display: block; width: 260px;">

## How to deploy and use a saved model?

Let’s see step by step how to deploy and use a sentiment classification Bert model.

Step 1: create a saved model

```python
from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-imdb", from_pt=True)
model.save_pretrained("my_model", saved_model=True)
```

Step 2: create a Docker container containing the saved model and run it

```
# pull the TensorFlow serving Docker image for CPU
# for GPU replace serving by serving:latest-gpu
docker pull tensorflow/serving

# run a serving image as a daemon
docker run -d --name serving_base tensorflow/serving

# copy the saved model into the container's model folder
docker cp my_model/saved_model serving_base:/models/bert

# commit the container serving the model by changing MODEL_NAME to match the model's name (here bert)
docker commit --change "ENV MODEL_NAME bert" serving_base my_bert_model

# kill the serving image ran as a daemon
docker kill serving_base

# Run the image to serve the model as a daemon
docker run -d -p 8501:8501 -p 8500:8500 --name bert my_bert_model
```

Step 3: Query the model through the REST API

```python
from transformers import BertTokenizerFast, BertConfig
import requests
import json
import numpy as np

sentence = "I love the new TensorFlow update in transformers."
tokenizer = BertTokenizerFast.from_pretrained("nateraw/bert-base-uncased-imdb")
config = BertConfig.from_pretrained("nateraw/bert-base-uncased-imdb")
batch = tokenizer(sentence)
input_data = {"instances": [dict(batch)]}
r = requests.post("http://localhost:8501/v1/models/bert:predict", data=json.dumps(input_data))

print(config.id2label[np.argmax(np.abs(json.loads(r.text)["predictions"][0]))])
```

And get POSITIVE. It is also possible to pass by the gRPC API to get the same result:

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
batch = tokenizer(sentence, return_tensors="tf")
channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = "bert"
request.model_spec.signature_name = "serving_default"
request.inputs["input_ids"].CopyFrom(tf.make_tensor_proto(batch["input_ids"]))
request.inputs["attention_mask"].CopyFrom(tf.make_tensor_proto(batch["attention_mask"]))
request.inputs["token_type_ids"].CopyFrom(tf.make_tensor_proto(batch["token_type_ids"]))
result = stub.Predict(request)

print(config.id2label[np.argmax(np.abs(result.outputs["logits"].float_val))])
```

## Conclusion
Thanks to the last updates applied on the TensorFlow models in transformers, one can now easily deploy its models in production in an efficient way. One of the next steps we are thinking about is to directly integrate the preprocessing part inside the saved model to make things even easier.