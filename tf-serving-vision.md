---
title: Deploying TensorFlow Vision Models in Hugging Face with TF Serving
---

<h1>
  Deploying TensorFlow Vision Models in Hugging Face with TF Serving
</h1>

<div class="blog-metadata">
    <small>Published July 19, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/tf-serving-vision.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/segments-tobias">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/22957388?v=4" title="Gravatar">
        <div class="bfc">
            <code>sayakpaul</code>
            <span class="fullname">Sayak Paul</span>
            <span class="bg-gray-100 dark:bg-gray-700 rounded px-1 text-gray-600 text-sm font-mono">guest</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/github/sayakpaul/deploy-hf-tf-vision-models/blob/main/hf_vision_model_tfserving.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In the past few months, the Hugging Face team and external contributors
added a variety of vision models in TensorFlow to Transformers. This
list is growing comprehensively and already includes state-of-the-art
pre-trained models like [Vision Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/vit),
[Masked Autoencoders](https://huggingface.co/docs/transformers/model_doc/vit_mae),
[RegNet](https://huggingface.co/docs/transformers/main/en/model_doc/regnet),
[ConvNeXt](https://huggingface.co/docs/transformers/model_doc/convnext),
and so on.

When it comes to deploying TensorFlow models, we've got a variety of
options. Depending on your use case, you may want to expose your model
as an endpoint or package it in an application itself. TensorFlow
provides tools that cater to each of these different scenarios.

In this post, we'll see how to deploy a Vision Transformer (ViT) model
locally using [[TensorFlow
Serving]{.ul}](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
(TF Serving). This will allow developers to expose the model either as a
REST or gRPC endpoint. Moreover, TF Serving supports many
deployment-specific features off-the-shelf such as model warmup,
server-side batching, etc.

To get the complete working code shown throughout this post, refer to
the Colab Notebook shown at the beginning.

# Saving the Model

All TensorFlow models in 🤗 Transformers have a method named
`save_pretrained()`. With it, we can serialize the model weights in
the h5 format as well as in the standalone [SavedModel format](https://www.tensorflow.org/guide/saved_model).
TF Serving needs a model to be present in the SavedModel format. So, let's first
load a Vision Transformer model and save it:

```py
from transformers import TFViTForImageClassification

temp_model_dir = "vit"
ckpt = "google/vit-base-patch16-224"

model = TFViTForImageClassification.from_pretrained(ckpt)
model.save_pretrained(temp_model_dir, saved_model=True)
```

By default, `save_pretrained()` will first create a version directory
inside the path we provide to it. So, the path ultimately becomes:
`{temp_model_dir}/saved_model/{version}`.

We can inspect the serving signature of the SavedModel like so:

```bash
saved_model_cli show \--dir {temp_model_dir}/saved_model/1 \--tag_set serve \--signature_def serving_default
```

This should output:

```bash
The given SavedModel SignatureDef contains the following input(s):\
inputs\[\'pixel_values\'\] tensor_info:\
dtype: DT_FLOAT\
shape: (-1, -1, -1, -1)\
name: serving_default_pixel_values:0\
The given SavedModel SignatureDef contains the following output(s):\
outputs\[\'logits\'\] tensor_info:\
dtype: DT_FLOAT\
shape: (-1, 1000)\
name: StatefulPartitionedCall:0\
Method name is: tensorflow/serving/predict
```

As can be noticed the model accepts single 4-d inputs (namely
`pixel_values`) which has the following axes: `(batch_size,
num_channels, height, width)`. For this model, the acceptable height
and width are set to 224, and the number of channels is 3. We can verify
this by inspecting the config argument of the model (`model.config`).
The model yields a 1000-d vector of `logits`.

# Model Surgery

Usually, every ML model has certain preprocessing and postprocessing
steps. Our ViT model is no exception to this. The major preprocessing
steps include:

-   Scaling the image pixel values to [0, 1] range.

-   Normalizing the scaled pixel values to [-1, 1\.

-   Resizing the image so that it has a spatial resolution of (224, 224).

We can confirm these by investigating the feature extractor associated
with the model:

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
print(feature_extractor)
```

This should print:

```bash
ViTFeatureExtractor {
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

Since this is an image classification model pre-trained on the
[ImageNet-1k dataset](https://huggingface.co/datasets/imagenet-1k), the model
outputs need to be mapped to the ImageNet-1k classes as the
post-processing step.

To reduce the developers' cognitive load and training-serving skew,
it's often a good idea to ship a model that has most of the
preprocessing and postprocessing steps in built. Therefore, we'll
serialize our model as a SavedModel such that the above-mentioned
processing ops get embedded into its computation graph.

## Preprocessing

For preprocessing, image normalization is one of the most essential
components:

```py
def normalize_img(
    img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
):
    # Scale to the value range of [0, 1] first and then normalize.
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std
```

We also need to resize the image, transpose it so that it has leading
channel dimensions since following the standard format of 🤗
Transformers. The below code snippet shows all the preprocessing steps:

```py
CONCRETE_INPUT = "pixel_values" # Which is what we investigated via the SavedModel CLI.
SIZE = feature_extractor.size


def normalize_img(
    img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
):
    # Scale to the value range of [0, 1] first and then normalize.
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std


def preprocess(string_input):
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    resized = tf.image.resize(decoded, size=(SIZE, SIZE))
    normalized = normalize_img(resized)
    normalized = tf.transpose(
        normalized, (2, 0, 1)
    )  # Since HF models are channel-first.
    return normalized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_fn(string_input):
    decoded_images = tf.map_fn(
        preprocess, string_input, dtype=tf.float32, back_prop=False
    )
    return {CONCRETE_INPUT: decoded_images}
```

**Note on making the model accept string inputs**:

When dealing with images via REST or gRPC requests the size of the
request payload can easily spiral up depending on the resolution of the
images being passed. This is why, it is a good practice to compress them
reliably and then prepare the request payload.

## Postprocessing and Model Export

We're now equipped with the preprocessing operations that we can inject
into the model's existing computation graph. In this section, we'll also
inject the post-processing operations into it and finally, export the
model.

```py
def model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, 3, SIZE, SIZE], dtype=tf.float32, name=CONCRETE_INPUT
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        labels = tf.constant(list(model.config.id2label.values()), dtype=tf.string)
        
        images = preprocess_fn(string_input)
        predictions = m_call(**images)
        
        indices = tf.argmax(predictions.logits, axis=1)
        pred_source = tf.gather(params=labels, indices=indices)
        probs = tf.nn.softmax(predictions.logits, axis=1)
        pred_confidence = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "confidence": pred_confidence}

    return serving_fn
```

We first derive the [concrete function](https://www.tensorflow.org/guide/function)
from the model's forward pass method (`call()`) so the model is nicely compiled
into a graph. After that we apply the following steps in order:

-   Pass the inputs through the preprocessing operations.

-   Pass the preprocessing inputs through the derived concrete function.

-   Post-process the outputs and return them in a nicely formatted
    dictionary.

Now we can export our model:

```py
MODEL_DIR = tempfile.gettempdir()
VERSION = 1

tf.saved_model.save(
    model,
    os.path.join(MODEL_DIR, str(VERSION)),
    signatures={"serving_default": model_exporter(model)},
)
os.environ["MODEL_DIR"] = MODEL_DIR
```

After exporting, we can again inspect the model signatures:

```bash
saved_model_cli show --dir {MODEL_DIR}/1 --tag_set serve --signature_def serving_default
```

```bash
The given SavedModel SignatureDef contains the following input(s):
  inputs['string_input'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: serving_default_string_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['confidence'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1)
      name: StatefulPartitionedCall:0
  outputs['label'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: StatefulPartitionedCall:1
Method name is: tensorflow/serving/predict
```

We can notice that the model's signature has now changed. Specifically,
the input type is now a string and the model returns two things: a
confidence score and the string label.

Provided you've already installed TF Serving (covered in the Colab
Notebook), we're now ready to deploy this model!

# Deployment with TensorFlow Serving

It just takes a single command to do this:

```bash
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=vit \
  --model_base_path=$MODEL_DIR >server.log 2>&1
```

From the above command, the important parameters are:

-   `rest_api_port` denotes the port number that TF Serving will use
    deploying the REST endpoint of your model. By default, TF Serving
    uses the 8500 port for the gRPC endpoint.

-   `model_name` specifies the model name (can be anything) that will
    used for calling the APIs.

-   `model_base_path` denotes the base model path that TF Serving will
    use to load the latest version of the model.

(The complete list of supported parameters is
[here](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/main.cc).)

And voila! Within minutes, you should be up and running with a deployed
model having two endpoints - REST and gRPC.

# Querying the REST Endpoint

Recall that we exported our model such that it accepts string inputs
encoded with the [base64 format](https://en.wikipedia.org/wiki/Base64). So, to craft our
request payload we'll do something like this:

```py
# Get image of a cute cat.
image_path = tf.keras.utils.get_file(
    "image.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg"
)

# Read the image from disk as raw bytes and then encode it. 
bytes_inputs = tf.io.read_file(image_path)
b64str = base64.urlsafe_b64encode(bytes_inputs.numpy()).decode("utf-8")


# Create the request payload.
data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
```

TF Serving's request payload format specification for the REST endpoint
is available [here](https://www.tensorflow.org/tfx/serving/api_rest#request_format_2).
Within the `instances` we can pass multiple encoded images. This kind
of endpoints are meant to be consumed for online prediction scenarios.
For inputs having more than a single data point, you would to want to
[enable batching](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md)
to get performance optimization benefits.

Now we can call the API:

```py
headers = {"content-type": "application/json"}
json_response = requests.post(
    "http://localhost:8501/v1/models/vit:predict", data=data, headers=headers
)
print(json.loads(json_response.text))
# {'predictions': [{'label': 'Egyptian cat', 'confidence': 0.896659195}]}

```

Our REST API is -
`http://localhost:8501/v1/models/vit:predict` following the specification from
[here](https://www.tensorflow.org/tfx/serving/api_rest#predict_api). By default,
this always picks up the latest version of the model. But if we wanted a
specific version we can do: `http://localhost:8501/v1/models/vit/versions/1:predict`.

# Querying the gRPC Endpoint

While REST is quite popular in the API world many applications often
benefit from gRPC. [This post](https://blog.dreamfactory.com/grpc-vs-rest-how-does-grpc-compare-with-traditional-rest-apis/)
does a good job comparing the two ways of deployment. gRPC is usually
preferred for low-latency, highly scalable, and distributed systems.

There are a couple of steps are. First, we need to open a communication
channel:

```py
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
```

Then, we create the request payload:

```py
request = predict_pb2.PredictRequest()
request.model_spec.name = "vit"
request.model_spec.signature_name = "serving_default"
request.inputs[serving_input].CopyFrom(tf.make_tensor_proto([b64str]))
```

We can determine the `serving_input` key programmatically like so:

```py
loaded = tf.saved_model.load(f"{MODEL_DIR}/{VERSION}")
serving_input = list(
    loaded.signatures["serving_default"].structured_input_signature[1].keys()
)[0]
print("Serving function input:", serving_input)
# Serving function input: string_input
```

Now, we can get some predictions:

```py
grpc_predictions = stub.Predict(request, 10.0)  # 10 secs timeout
print(grpc_predictions)
```

```bash
outputs {
  key: "confidence"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
    }
    float_val: 0.8966591954231262
  }
}
outputs {
  key: "label"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
    }
    string_val: "Egyptian cat"
  }
}
model_spec {
  name: "resnet"
  version {
    value: 1
  }
  signature_name: "serving_default"
}
```

We can also fetch the key-values of our interest from the above results like so:

```py
grpc_predictions.outputs["label"].string_val, grpc_predictions.outputs[
    "confidence"
].float_val
# ([b'Egyptian cat'], [0.8966591954231262])
```

# Wrapping Up

In this post, we learned how to deploy a TensorFlow vision model from
Transformers with TF Serving. While local deployments are great for
weekend projects, we would want to be able to scale these deployments to
serve many users. In the next series of posts, we'll see how to scale up
these deployments with Kubernetes and Vertex AI.

# Additional References

-   [gRPC](https://grpc.io/)

-   [Practical Machine Learning for Computer Vision](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/)

-   [Faster TensorFlow models in Hugging Face Transformers](https://huggingface.co/blog/tf-serving)
