---
title: "Finetune Whisper with LoRA & BNB powered by 🤗 PEFT" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: reach-vb
---

# Fine tune Whisper with LoRA powered by PEFT

<!-- {blog_metadata} -->
<!-- {authors} -->

<a target="_blank" href="https://colab.research.google.com/github/Vaibhavs10/notebooks/blob/main/Whisper_w_PEFT.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A one size fits all walkthrough, to fine-tune Whisper (large) **5x faster** on a consumer GPU with **less than 8GB GPU VRAM**, all with comparable performance to full-finetuning. ⚡️

## Table of Contents

1. [Why Parameter Efficient Fine Tuning?](#introduction)
2. [Fine-tuning Whisper in a Google Colab](#fine-tuning-whisper-in-a-google-colab)
    1. [Prepare Environment](#prepare-environment)
    2. [Load Dataset](#load-dataset)
    3. [Prepare Feature Extractor, Tokenizer and Data](#prepare-feature-extractor-tokenizer-and-data)
    4. [Training and Evaluation](#training-and-evaluation)
3. [Closing Remarks](#closing-remarks)

We present a step-by-step guide on how to fine-tune Whisper with Common Voice 13.0 dataset using 🤗 Transformers and PEFT. In this Colab, we leverage `PEFT` and `bitsandbytes` to train a `whisper-large-v2` checkpoint seamlessly with a free T4 GPU (16 GB VRAM).

For more details on Whisper fine-tuning, datasets and metrics, refer to Sanchit Gandhi's brilliant blogpost: [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)

## Why Parameter Efficient Fine Tuning ([PEFT](https://github.com/huggingface/peft))?

As model sizes continue to increase, fine-tuning a model has become both computationally expensive and storage heavy. For example, a `Whisper-large-v2` model requires ~24GB of GPU VRAM for full fine-tuning and requires ~7 GB of storage for each fine-tuned checkpoint. For low-resource environments this becomes quite a bottleneck and often near impossible to get meaningful results.

Cue: PEFT! With PEFT you can tackle this bottleneck head-on. Like Low Rank Adaptation (LoRA), PEFT only fine-tunes a small number of (extra) model parameters while freezing most parameters of the pretrained model, thereby greatly decreasing the computational and storage costs.

### Aha! So wait, what's this LoRA thing?

PEFT comes out-of-the-box with multiple parameter efficient techniques. One such technique is [Low Rank Adaptation or LoRA](https://github.com/microsoft/LoRA). LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This greatly reduces the number of trainable parameters for downstream tasks. 

LoRA performs on-par or better than fine-tuning in model quality despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency.

### That's all cool, but show me the numbers?

Don't worry, we got ya! We ran multiple experiments to compare a full fine-tuning of the Whisper-large-v2 checkpoint and with PEFT fine-tuning. Here's what we found:

1. We were able to fine-tune a 1.6B parameter model with less than 8GB GPU VRAM. 🤯
2. With significantly less number of traininable parameters, we were able to fit almost **5x** more batch size. 📈
3. The resulting checkpoints were less than 1% the size of the original model, ~60MB. 🚀

To make things even better, all of this comes with minimal changes to the existing 🤗 transformers Whisper inference codebase.

Curious to test this out for yourself? Follow along!

## Fine-tuning Whisper in a Google Colab

### Prepare Environment

We'll employ several popular Python packages to fine-tune the Whisper model.
We'll use `datasets` to download and prepare our training data and 
`transformers` to load and train our Whisper model. We'll also require
the `librosa` package to pre-process audio files, `evaluate` and `jiwer` to
assess the performance of our model. Finally, we'll
use `PEFT`, `bitsandbytes`, `accelerate` to prepare and fine-tune the model with LoRA.

```python
!pip install -q transformers datasets librosa evaluate jiwer gradio bitsandbytes==0.37 accelerate 
!pip install -q git+https://github.com/huggingface/peft.git@main
```

We strongly advise you to upload model checkpoints directly the [Hugging Face Hub](https://huggingface.co/) 
whilst training. The Hub provides:
- Integrated version control: you can be sure that no model checkpoint is lost during training.
- Tensorboard logs: track important metrics over the course of training.
- Model cards: document what a model does and its intended use cases.
- Community: an easy way to share and collaborate with the community!

Linking the notebook to the Hub is straightforward - it simply requires entering your Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

```python
from huggingface_hub import notebook_login

notebook_login()
```

# Load Dataset

Using 🤗 Datasets, downloading and preparing data is extremely simple. 
We can download and prepare the Common Voice splits in just one line of code. 

First, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_13_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.

Since Hindi is very low-resource, we'll combine the `train` and `validation` 
splits to give approximately 12 hours of training data. We'll use the 6 hours 
of `test` data as our held-out test set:

```python
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)

print(common_voice)
```

**Print output:**

```
DatasetDict({
    train: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 6760
    })
    test: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 2947
    })
})
```

Most ASR datasets only provide input audio samples (`audio`) and the 
corresponding transcribed text (`sentence`). Common Voice contains additional 
metadata information, such as `accent` and `locale`, which we can disregard for ASR.
Keeping the notebook as general as possible, we only consider the input audio and
transcribed text for fine-tuning, discarding the additional metadata information:

```python
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
)

print(common_voice)
```

**Print output:**
```
DatasetDict({
    train: Dataset({
        features: ['audio', 'sentence'],
        num_rows: 6760
    })
    test: Dataset({
        features: ['audio', 'sentence'],
        num_rows: 2947
    })
})
```

### Prepare Feature Extractor, Tokenizer and Data

The ASR pipeline can be de-composed into three stages: 
1. A feature extractor which pre-processes the raw audio-inputs
2. The model which performs the sequence-to-sequence mapping 
3. A tokenizer which post-processes the model outputs to text format

In 🤗 Transformers, the Whisper model has an associated feature extractor and tokenizer, 
called [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor)
and [WhisperTokenizer](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer) 
respectively.

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
```

```python
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
```

To simplify using the feature extractor and tokenizer, we can _wrap_ both into a single `WhisperProcessor` class. This processor object can be used on the audio inputs and model predictions as required. 
In doing so, we only need to keep track of two objects during training: 
the `processor` and the `model`:

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
```

### Prepare Data

Let's print the first example of the Common Voice dataset to see 
what form the data is in:

```python
print(common_voice["train"][0])
```

**Print output:**
```
{'audio': {'path': '/root/.cache/huggingface/datasets/downloads/extracted/ff5a2373454f699ff252bdfd5f333826b18a3a91903d16ed05625bbdbabea9c7/common_voice_hi_26008353.mp3', 'array': array([ 5.81611368e-26, -1.48634016e-25, -9.37040538e-26, ...,
        1.06425901e-07,  4.46416450e-08,  2.61450239e-09]), 'sampling_rate': 48000}, 'sentence': 'हमने उसका जन्मदिन मनाया।'}
```

Since 
our input audio is sampled at 48kHz, we need to _downsample_ it to 
16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. 

We'll set the audio inputs to the correct sampling rate using dataset's 
[`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
method. This operation does not change the audio in-place, 
but rather signals to `datasets` to resample audio samples _on the fly_ the 
first time that they are loaded:

```python
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
```

Re-loading the first audio sample in the Common Voice dataset will resample 
it to the desired sampling rate:

```python
print(common_voice["train"][0])
```
Re-loading the first audio sample in the Common Voice dataset will resample 
it to the desired sampling rate:
```
{'audio': {'path': '/root/.cache/huggingface/datasets/downloads/extracted/ff5a2373454f699ff252bdfd5f333826b18a3a91903d16ed05625bbdbabea9c7/common_voice_hi_26008353.mp3', 'array': array([ 5.81611368e-26, -1.48634016e-25, -9.37040538e-26, ...,
        1.06425901e-07,  4.46416450e-08,  2.61450239e-09]), 'sampling_rate': 48000}, 'sentence': 'हमने उसका जन्मदिन मनाया।'}
```

Now we can write a function to prepare our data ready for the model:
1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
3. We encode the transcriptions to label ids through the use of the tokenizer.

```python
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
```

We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially.

Make yourself some tea 🍵, depending on dataset size, this might take 20-30 minutes ⏰

```python
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
```

```python
common_voice["train"]
```
**Print output:**
```
Dataset({
    features: ['input_features', 'labels'],
    num_rows: 6760
})
```

### Training and Evaluation

## Training and Evaluation

Now that we've prepared our data, we're ready to dive into the training pipeline. 
The [🤗 Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer)
will do much of the heavy lifting for us. All we have to do is:

- Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.

- Evaluation metrics: during evaluation, we want to evaluate the model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric.

- Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.

- Define the training configuration: this will be used by the 🤗 Trainer to define the training schedule.

Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it 
to transcribe speech in Hindi.

### Define a Data Collator

The data collator for a sequence-to-sequence speech model is unique in the sense that it 
treats the `input_features` and `labels` independently: the  `input_features` must be 
handled by the feature extractor and the `labels` by the tokenizer.

The `input_features` are already padded to 30s and converted to a log-Mel spectrogram 
of fixed dimension by action of the feature extractor, so all we have to do is convert the `input_features`
to batched PyTorch tensors. We do this using the feature extractor's `.pad` method with `return_tensors=pt`.

The `labels` on the other hand are un-padded. We first pad the sequences
to the maximum length in the batch using the tokenizer's `.pad` method. The padding tokens 
are then replaced by `-100` so that these tokens are **not** taken into account when 
computing the loss. We then cut the BOS token from the start of the label sequence as we 
append it later during training.

We can leverage the `WhisperProcessor` we defined earlier to perform both the 
feature extractor and the tokenizer operations:

```python
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```

Let's initialise the data collator we've just defined:

```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

### Evaluation metrics

We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing 
ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:

```python
import evaluate

metric = evaluate.load("wer")
```
### Load a Pre-Trained Checkpoint

Now let's load the pre-trained Whisper checkpoint. Again, this 
is trivial through use of 🤗 Transformers!

To reduce our models memory footprint, we load the model in 8bit, this means we quantize the model to use 1/4th precision (when comapared to float32) with minimal loss to performance. To read more about how this works, head over [here](https://huggingface.co/blog/hf-bitsandbytes-integration).

```python
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
```
### Post-processing on the model

Finally, we need to apply some post-processing steps on the 8-bit model to enable training. We do so by first freezing all the model layers, and then cast the layer-norm and the output layer in `float32` for training and model stability.

```python
from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
```

Since the Whisper model uses Convolutional layers in the Encoder, checkpointing disables grad computation to avoid this we specifically need to make the inputs trainable.

```python
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
```

### Apply Low-rank adapters (LoRA) to the model

Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

```python
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()
```
**Print output:**
```
trainable params: 15728640 || all params: 1559033600 || trainable%: 1.0088711365810203
```
We are ONLY using **1%** of the total trainable parameters, thereby performing **Parameter-Efficient Fine-Tuning**

### Define the Training Configuration

In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="reach-vb/test",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=1,
    evaluation_strategy="steps",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=100,
    max_steps=100, # only for testing purposes, remove this from your final run :)
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)
```

Fine-tuning a model with PEFT comes with a few caveats.

1. We need to explicitly set `remove_unused_columns=False` and `label_names=["labels"]` as the PeftModel's forward doesn't inherit the signature of the base model's forward.

2. Since INT8 training requires autocasting, we cannot use the native `predict_with_generate` call in Trainer as it doesn't automatically cast.

3. Similarly, since we cannot autocast, we cannot pass the `compute_metrics` to `Seq2SeqTrainer` so we'll comment it out whilst instantiating the Trainer.

```python
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
```

and, we are all set! We can now set our model to train! <3

```python
trainer.train()
```

Depending on your dataset it should take about 6-8 hours to fine-tune the model.

Once the model is fine-tuned, we can push the model on to Hugging Face Hub, this will later help us directly infer the model from the model repo.

```python
peft_model_id = "reach-vb/whisper-large-v2-hindi-100steps"
model.push_to_hub(peft_model_id)
```

### Evaluation and Inference

On to the fun part, we've successfully fine-tuned our model. Now let's put it to test and calculate the WER on the `test` set.

As with training, we do have a few caveats to pay attention to:
1. Since we cannot use `predict_with_generate` function, we will hand roll our own eval loop with `torch.cuda.amp.autocast()` you can check it out below. 
2. Since the base model is frozen, PEFT model sometimes fails to recognise the language while decoding. To fix that, we force the starting tokens to mention the language we are transcribing. This is done via `forced_decoder_ids = processor.get_decoder_prompt_ids(language="Marathi", task="transcribe")` and passing that too the `model.generate` call.

That's it, let's get transcribing! 🔥

```python
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

peft_model_id = "reach-vb/whisper-large-v2-hindi-100steps" # Use the same model ID as before.
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
model.config.use_cache = True
```

Let's define our Evaluation loop

```Python
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
normalizer = BasicTextNormalizer()

predictions = []
references = []
normalized_predictions = []
normalized_references = []

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
            normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
        del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute(predictions=predictions, references=references)
normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

print(f"{wer=} and {normalized_wer=}")
print(eval_metrics)
```

## Fin!

If you made it all the way till the end then pat yourself on the back. Looking back, we learned how to train *any* Whisper checkpoint faster, cheaper and with negligible loss in WER.

With PEFT, you can also go beyond Speech recognition and apply the same set of techniques to other pretrained models as well. Come check it out here: https://github.com/huggingface/peft 🤗

Don't forget to tweet your results and tag us! [@huggingface](https://twitter.com/huggingface) and [@reach_vb](https://twitter.com/reach_vb) ❤️