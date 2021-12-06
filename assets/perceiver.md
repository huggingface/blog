---
title: "Perceiver IO: a scalable, fully-attentional model that works on any modality"
thumbnail: /todo
---

<h1>Perceiver IO: a scalable, fully-attentional model that works on any modality</h1>

<div class="blog-metadata">
    <small>Published December 5, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/perceiver.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/nielsrogge">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/48327001?v=4" title="Gravatar">
        <div class="bfc">
            <code>nielsrogge</code>
            <span class="fullname">Niels Rogge</span>
        </div>
    </a>
</div>

### Introduction

The [Transformer](https://arxiv.org/abs/1706.03762), originally introduced by 
Vaswani et al. in 2017, caused a revolution in the AI community, initially improving
state-of-the-art (SOTA) results in machine translation. In 2018, [BERT](https://arxiv.org/abs/1810.04805)
was released, a Transformer encoder-only model that crushed the benchmarks of natural language
processing (NLP), most famously the [GLUE benchmark](https://gluebenchmark.com/). 

Not long after that, AI researchers started to apply the idea of BERT to other domains. To name a few examples, [Wav2Vec2](https://arxiv.org/abs/2006.11477) by Facebook AI illustrated that the architecture could be extended to audio, the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) by Google AI showed that the architecture also works really well for computer vision, and most recently the [Video Vision transformer (ViViT)](https://arxiv.org/abs/2103.15691), also by Google AI, applied the architecture to video. In all of these domains, state-of-the-art results were improved dramatically, thanks to the combination of this powerful architecture with large-scale pre-training.

However, there's an important limitation to the architecture of the Transformer: it scales very poorly in both compute and memory. This is because the self-attention mechanism has a complexity of O(n^2). The memory and compute requirements scale quadratically in the number of tokens that you provide as input to the model. In every layer, all inputs are used to produce queries and keys, for which a pairwise dot product is computed. Hence, it is not possible to apply self-attention on high-dimensional data, without some form of preprocessing. Wav2Vec2 for example solves this by employing a feature encoder, to turn a raw waveform into a sequence of time-based features. The Vision Transformer (ViT) divides an image into a sequence of non-overlapping patches, which serve as "tokens". The Video Vision Transformer (ViViT) extracts non-overlapping, spatio-temporal
“tubes” from a video, which serve as "tokens". To make the Transformer work on a particular modality, one typically discretizes it to a sequence of tokens to make it work.

The Perceiver aims to solve this limitation by employing the self-attention mechanism on a set of latent variables, rather than on the inputs. The inputs are only used for doing cross-attention with the latents. This has the advantage that the bulk of compute happens in a latent space, where compute is cheap (one typically uses 256 or 512 latents). The resulting architecture has no quadratic dependence on the input size: the Transformer encoder only depends linearly on the input size, while latent attention is independent of it. In a follow-up paper, called Perceiver IO, the authors extend this idea to let the Perceiver also handle arbitrary outputs. The idea is similar: one only uses the outputs for doing cross-attention with the latents. 

In the following section, we look in a bit more detail at how Perceiver IO actually works. This model is implemented in HuggingFace Transformers, and in the sections below we explain in detail  - in terms of shapes of tensors - how the inputs are actually pre and post processed.

To initialize a `PerceiverModel`, one can provide 3 additional instances to the model:
- a preprocessor
- a decoder
- a postprocessor.

Note that each of these are optional. A preprocessor is only required in case one hasn't already embedded the input modality (such as text, images, audio) him/herself. A decoder is only required in case one wants to decode the output of the Perceiver encoder (i.e. the last hidden states of the latents) into something more useful such as classification logits or optical flow. A postprocessor is only required in case one wants to turn the output of the decoder into a specific feature (this is only required when doing auto-encoding, as we will see further). Let's start of by showing how you can use the Perceiver on text.

## Perceiver for text

Suppose that you want to apply the Perceiver to perform text classification. As the memory and time requirements of the self-attention mechanism don't depend on the size of the inputs, one can directly provide raw UTF-8 bytes to the model. For a fair comparison to BERT, the authors used input sequences of 2048 bytes. Let's say we use a batch size of 1, then the `inputs` to the model are of shape (1, 2048). The `inputs` contain the byte IDs (similar to the `input_ids` of BERT) for a single piece of text.

In this case, one provides `PerceiverTextPreprocessor` as preprocessor to the model, which will take care of embedding the `inputs` (i.e. turn each byte ID into a corresponding vector). As decoder, one provides `PerceiverClassificationDecoder` to the model (which will turn the last hidden states of the latents into classification logits). No postprocessor is required. 

In other words, a Perceiver model for text classification (which is called `PerceiverForSequenceClassification` in HuggingFace Transformers) can be initialized as follows:

``` python
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor, PerceiverClassificationDecoder

class PerceiverForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__(config)

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverTextPreprocessor(config),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
                use_query_residual=True,
            ),
        )
```
One can already see here that the decoder is initialized with trainable position encoding arguments. Why is that? Well, let's take a look in detail at how Perceiver IO works. Recall that one provides raw UTF-8 bytes to the model as `inputs`. We can use `PerceiverTokenizer` to turn a text into a sequence of byte IDs, padded up to a length of 2048:

``` python
from transformers import PerceiverTokenizer

tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")

text = "hello world"

inputs = tokenizer(text, padding="max_length", return_tensors="pt").input_ids
```

When providing the `inputs` to the model to perform a forward pass, what does actually happen? At initialization, `PerceiverModel` defines a set of latent variables, as follows:

``` python
from torch import nn

self.latents = nn.Parameter(torch.randn(config.num_latents, config.d_latents))
```

In the Perceiver IO paper, one uses 256 latents, and sets the dimensionality of the latents to 1280. If we also add a batch dimension, the Perceiver has latents of shape (1, 256, 1280). First, the preprocessor (which one provides at initialization) will take care of embedding the UTF-8 byte IDs to embedding vectors. Hence, `PerceiverTextPreprocessor` will turn the `inputs` of shape (batch_size, 2048) to a tensor of shape (batch_size, 2048, 768) - assuming that each byte is turned into a vector of size 768.

The first thing Perceiver IO does is applying cross-attention between the latents (which serve as queries) of shape (batch_size, 256, 1280) and the inputs (which serve as keys and values) of shape (batch_size, 2048, 768). The output of this initial cross-attention operation is a tensor that has the same shape as the queries (which are the latents, in this case). In other words, the output of the cross-attention operation is of shape (1, 256, 1280). 

Next, a deep stack of self-attention layers are applied to update the representations of the latents. Note that these don't depend on the length of the inputs (i.e. the bytes) we provided, as these were only used during the cross-attention operation. In the Perceiver IO paper, 26 self-attention layers were used to update the representation of the latents. Note that the output after these 26 self-attention layers still has the same shape as what one initially provided as input: (1, 256, 1280). These are also called the "last hidden state" of the latents. This is very similar to the "last hidden states" of the tokens one provides to BERT. 

Ok, so now we have final hidden states of shape (1, 256, 1280). Great, but we actually want to turn these into classification logits of shape (batch_size, num_labels). How can we make the Perceiver output these? 

This is handled by `PerceiverClassificationDecoder`. The idea is very similar to what we did when mapping the inputs to the latent space: we are going to use cross-attention. But now, the latent variables will serve as keys and values, and we will provide a tensor of whatever shape we'd like - in this case we'll provide a tensor of shape (batch_size, 1, num_labels) which will act as queries (the authors refer to these as "decoder queries", because they are used in the decoder). This tensor will be randomly initialized at the beginning of training. As you can see, we just provide a dummy sequence length dimension of 1. Note that the output of an attention layer always has the same shape as the shape of the queries - hence the decoder will output a tensor of shape (batch_size, 1, num_labels). We can then simply squeeze this tensor to have shape (batch_size, num_labels) and boom, we have our classification logits. 


## Perceiver for images

Now that we've seen how to apply the Perceiver to perform text classification, it is straightforward to apply the Perceiver to do image classification. The only difference is that we'll provide a different preprocessor to the model, which will embed our image modality. The Perceiver authors actually tried out 3 different preprocessors: 
- one that flattens the pixel values, applies a convolutional layer with kernel size 1 and adds learned absolute 1D position embeddings.
- one that flattens the pixel values and adds fixed 2D Fourier position embeddings.
- one that applies a 2D convolutional + maxpool layer and adds fixed 2D Fourier position embeddings.

Each of these are implemented in the Transformers library, and they are called `PerceiverForImageClassificationLearned`, `PerceiverForImageClassificationFourier` and `PerceiverForImageClassificationConvProcessing` respectively. They only differ in their settings of the `PerceiverImagePreprocessor`. Let's take a closer look at `PerceiverForImageClassificationLearned`. It initializes a `PerceiverModel` as follows: 

``` python
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor, PerceiverClassificationDecoder

class PerceiverForImageClassificationLearned(nn.Module):
    def __init__(self, config):
        super().__init__(config)

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="trainable",
                concat_or_add_pos="concat",
                project_pos_dim=256,
                trainable_position_encoding_kwargs=tdict(num_channels=256, index_dims=config.image_size ** 2),
            ),
            decoder=PerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
                use_query_residual=True,
            ),
        )
```

We can see that `PerceiverImagePreprocessor` is initialized with `prep_type = "conv1x1"` and that we add arguments for the trainable position encodings. So how does this preprocessor work in detail? Suppose that we provide a batch of images to the model. Let's say we center crop and normalize our images first, such that we have `inputs` of shape (batch_size, num_channels, height, width) = (1, 3, 224, 224). 

The preprocessor (with the settings defined above) will first apply a convolutional layer with kernel size (1, 1) to turn the pixel values into a tensor of shape (1, 256, 224, 224) - hence increasing the channel dimension. It will then place the channel dimension last - so now one has a tensor of shape (1, 224, 224, 256). Next, it flattens the spatial (height + width) dimensions such that one has a tensor of shape (1, 50176, 256). Next, it adds trainable 1D position embeddings. As the dimensionality of the position embeddings is defined to be 256, one is left with a tensor of shape (1, 50176, 512). This tensor will be used for the cross-attention operation with the latents.

The authors use 512 latents for all image models, and set the dimensionality of the latents to 1024. Hence, the latents are a tensor of shape (1, 512, 1024) - assuming we add a batch dimension. The cross-attention layer takes the queries of shape (1, 512, 1024) and keys + values of shape (1, 50176, 512) as input, and produces a tensor that has the same shape as the queries, (1, 512, 1024) as output. Next, 8 blocks of 6 self-attention layers are applied, to produce final hidden states of the latents of shape (1, 512, 1024). To turn these into classification logits, `PerceiverClassificationDecoder` is used, which works similarly to the one for text classification: it uses the latents as keys + values, and uses trainable position embeddings of shape (1, 1, num_labels) as queries. The output of the cross-attention operation is a tensor of shape (1, 1, num_labels), which is squeezed to have classification logits of shape (1, num_labels).

## Perceiver for optical flow

The authors show that it's straightforward to make the Perceiver also work on optical flow, which is a decades-old problem in computer vision, with many broader applications. Given two images of the same scene (e.g. two consecutive frames of a video), the task is to estimate the 2D displacement for each pixel in the first image. Existing algorithms are quite hand-engineered and complex, however with the Perceiver, this becomes relatively simple. The model is implemented in the Transformers library, and called `PerceiverForOpticalFlow`. The model is implemented as follows:

``` python
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor, PerceiverOpticalFlowDecoder

class PerceiverForOpticalFlow(nn.Module):
    def __init__(self, config):
        super().__init__(config)

        fourier_position_encoding_kwargs_preprocessor = dict(
            num_bands=64,
            max_resolution=config.train_size,
            sine_only=False,
            concat_pos=True,
        )
        fourier_position_encoding_kwargs_decoder = dict(
            concat_pos=True, max_resolution=config.train_size, num_bands=64, sine_only=False
        )
        
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverImagePreprocessor(
                config,
                prep_type="patches",
                spatial_downsample=1,
                conv_after_patching=True,
                conv_after_patching_in_channels=54,
                temporal_downsample=2,
                position_encoding_type="fourier",
                # position_encoding_kwargs
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=PerceiverOpticalFlowDecoder(
                config,
                num_channels=config.d_model,
                output_image_shape=config.train_size,
                rescale_factor=100.0,
                use_query_residual=False,
                output_num_channels=2,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
            ),
        )
```
As one can see, `PerceiverImagePreprocessor` is used as preprocessor (i.e. to embed the 2 images for the cross-attention operation with the latents) and `PerceiverOpticalFlowDecoder` is used as decoder (i.e. to decode the final hidden states of the latents to an actual predicted flow). For each of the 2 frames, the authors extract a 3 x 3 patch around each pixel, leading to 3 x 3 x 3 = 27 values for each pixel (as each pixel also has 3 color channels). The authors use a training resolution of (368, 496). If one stacks 2 frames of size (368, 496) of each training example on top of each other, the `inputs` to the model are of shape (batch_size, 2, 27, 368, 496). 

The preprocessor (with the settings defined above) will first concatenate the frames along the channel dimension, leading to a tensor of shape (batch_size, 368, 496, 54) - assuming we also move the channel dimension to be last. The authors explain in their paper (page 8) why concatenation along the channel dimension makes sense. Next, the spatial dimensions are flattened, leading to a tensor of shape (batch_size, 368*496, 54) = (batch_size, 182528, 54). Then, position embeddings (each of which have dimensionalityy 258) are concatenated, leading to a final preprocessed input of shape (batch_size, 182528, 322). These will be used to perform cross-attention with the latents.

The authors use 2048 latents for the optical flow model (yes, 2048!), with a dimensionality of 512 for each latent. Hence, the latents have shape (batch_size, 2048, 512). After the cross-attention, we again have a tensor of the same shape (as the latents act as queries). Next, a single block of 24 self-attention layers (each of which has 16 attention heads) are applied to update the embeddings of the latents. 

To decode the final hidden states of the latents to an actual predicted flow, `PerceiverOpticalFlowDecoder` simply uses the preprocessed inputs of shape (batch_size, 182528, 322) as queries for the cross-attention operation. Next, these are projected to a tensor of shape (batch_size, 182528, 2). Finally, one rescales and reshapes this to a predicted flow of shape (batch_size, 368, 496, 2). The authors claim state-of-the-art results on important benchmarks including Sintel and KTTI when training on [AutoFlow](), a dataset of 400,000 annotated image pairs.

## Perceiver for multimodal autoencoding

The authors also use the Perceiver for multimodal autoencoding. The goal of multimodal autoencoding is to learn a model that can accurately reconstruct multimodal inputs in the presence of a bottleneck induced by an architecture. The authors train the model on the [Kinetics-700 dataset](https://deepmind.com/research/open-source/kinetics), in which each example consists of a sequence of images (i.e. frames), audio and a class label. This model is also implemented in HuggingFace Transformers, and available as `PerceiverForMultimodalAutoencoding`. For brevity, I will omit the code of defining this model, but important to note is that it uses `PerceiverMultimodalPreprocessor` to prepare the inputs for the model. This preprocessor will first use the respective preprocessor for each modality (image, audio, label). Suppose we have a video of 16 frames, then the modalities are preprocessed as follows: 

- The images - actually a sequence of frames - of shape (1, 16, 3, 224, 224) are turned into a tensor of shape (1, 50176, 243) using `PerceiverImagePreprocessor`. This is a “space to depth” transformation, after which fixed 2D Fourier position embeddings are concatenated.
- The audio has shape (1, 30720, 1) and is turned into a tensor of shape (1, 1920, 401) using `PerceiverAudioPreprocessor`.
- The class label of shape (1, 700) is turned into a tensor of shape (1, 1, 700) using `PerceiverForMultimodalAutoencoding`. In other words, this preprocessor just adds a dummy time (index) dimension. 

Next, `PerceiverMultimodalPreprocessor` will pad the preprocessed modalities with modality-specific trainable embeddings to make concatenation along the time dimension possible. In this case, the modality with the highest channel dimension is the class label (it has 700 channels). The authors enforce a minimum padding size of 4, hence each modality will be padded to have 704 channels. They can then be concatenated, hence the final input to the Perceiver model is a tensor of shape (batch_size, 50176 + 1920 + 1, 704) = (batch_size, 52097, 704). 

The authors use 784 latents, with a dimensionality of 512 for each latent. Hence, the latents have shape (batch_size, 784, 512). After the cross-attention, we again have a tensor of the same shape (as the latents act as queries). Next, a single block of 8 self-attention layers (each of which has 8 attention heads) are applied to update the embeddings of the latents. 

Next, we have `PerceiverMultimodalDecoder`, which will create queries for each modality separately. However, as it is not possible to decode an entire video in a single forward pass, the authors instead auto-encode in chunks. Each chunk will subsample certain index dimensions for every modality.

- For the image modality, the total size of the decoder query is 16x3x224x224 = 802,816. However, when auto-encoding the first chunk, one subsamples the first 802,816/128 = 6272 values. The shape of the image output query is (1, 6272, 195).
- For the audio modality, the total input has 30,720 values. However, one only subsamples the first 30720/128/16 = 15 values. Hence, the shape of the audio query is (1, 15, 385).
- For the class label modality, one doesn't subsample, as one does not provide the label input to the model. Hence, the subsampled index is set to 1. The shape of the label output query is (1, 1, 1024).

Similarly to the preprocessor, `PerceiverMultimodalDecoder` pads the different modalities to the same number of channels, to make concatenation of the modality-specific queries possible along the time dimension. Here, the class label has again the highest number of channels (1024), and the authors enforce a minimum padding size of 2, hence every modality will be padded to have 1026 channels. After concatenation, the final decoder query will have shape (batch_size, 6727 + 15 + 1, 1026) = (batch_size, 6288, 1026). This tensor acts as queries in the cross-attention operation, while the latents act as keys and values. Hence, the output of the cross-attention operation is a tensor of shape (batch_size, 6288, 1026). Next, `PerceiverMultimodalDecoder` employs a linear layer to reduce the output channels to get a tensor of shape (1, 6288, 512). 

Finally, we have `PerceiverMultimodalPostprocessor`. This class will postprocess the output of the decoder to produce an actual reconstruction of each modality. It first splits up the time dimension of the decoder output according to the different modalities: (1, 6272, 512) for image, (batch_size, 15, 512) for audio and (1, 1, 512) for the class label. Next, the respective postprocessors for each modality are applied:

- `PerceiverImagePostprocessor` (which is actually called `PerceiverProjectionPostprocessor`) simply turns the (1, 6272, 512) tensor into a tensor of shape (1, 6272, 3) - i.e. projects the final dimension to RGB values.
- `PerceiverAudioPostprocessor` turns the (1, 15, 512) tensor into a tensor of shape (1, 240).
- `PerceiverClassificationPostprocessor` simply takes the first (and only index), to get a tensor of shape (1, 700). 

So now one ends up with tensors containing the reconstruction of the image, audio and class label modalities respectively. As one auto-encodes an entire video in chunks, one needs to concatenate the reconstruction of each chunk to have a final reconstruction of an entire video.



