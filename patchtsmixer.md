---
title: "PatchTSMixer in HuggingFace"
thumbnail: /blog/assets/patchtsmixer/thumbnail.png
authors:
- user: ajati
- user: vijaye12
- user: namctin
- user: wgifford
- user: kashif
---


# PatchTSMixer in HuggingFace - Getting Started


<!-- #region -->
In this blog, we will demonstrate examples of getting started with PatchTSMixer. We will first demonstrate the forecasting capability of `PatchTSMixer` on the Electricity data. We will then demonstrate the transfer learning capability of PatchTSMixer by using the model trained on the Electricity to do zero-shot forecasting on the ETTH2 dataset.

`PatchTSMixer` is a lightweight time-series modeling approach based on the MLP-Mixer architecture. It is proposed in [TSMixer: Lightweight MLP-Mixer Model for Multivariate
 Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) by IBM Research authors `Vijay Ekambaram`, `Arindam Jati`,
 `Nam Nguyen`, `Phanwadee Sinthong` and `Jayant Kalagnanam`.

For effective mindshare and to promote opensourcing - IBM Research join hands with the HuggingFace team to opensource this model in HF.

In this [HuggingFace implementation](https://huggingface.co/docs/transformers/main/en/model_doc/patchtsmixer), we provide PatchTSMixer’s capabilities to effortlessly facilitate lightweight mixing across patches, channels, and hidden features for effective multivariate time-series modeling. It also supports various attention mechanisms starting from simple gated attention to more complex self-attention blocks that can be customized accordingly. The model can be pretrained and subsequently used for various downstream tasks such as forecasting, classification, and regression.

`PatchTSMixer` outperforms state-of-the-art MLP and Transformer models in forecasting by a considerable margin of 8-60%. It also outperforms the latest strong benchmarks of Patch-Transformer models (by 1-2%) with a significant reduction in memory and runtime (2-3X). For more details, refer to the [paper](https://arxiv.org/pdf/2306.09364.pdf)


`Blog authors`: Arindam Jati, Vijay Ekambaram, Nam Ngugen, Wesley Gifford and Kashif Rasul

<!-- #endregion -->

## PatchTSMixer Quick Overview 
#### Skip this section if you are familiar with `PatchTSMixer`!
`PatchTSMixer` patches a given input multivariate time series into a sequence of patches or windows. Subsequently, it passes the series to an embedding layer, which generates a multi-dimensional tensor.

<img src="assets/patchtsmixer/overview/1.gif" width="640" height="360"/>

The multi-dimensional tensor is subsequently passed to the `PatchTSMixer` backbone, which is composed of a sequence of [MLP Mixer](https://arxiv.org/abs/2105.01601) layers. Each MLP Mixer layer learns inter-patch, intra-patch, and inter-channel correlations through a series of permutation and MLP operations.

<img src="assets/patchtsmixer/overview/2.gif" width="640" height="360"/>

`PatchTSMixer` also employs residual connections and gated attentions to prioritize of important features.

<img src="assets/patchtsmixer/overview/3.gif" width="640" height="360"/>

Hence, a sequence of MLP Mixer layers creates the following `PatchTSMixer` backbone. 

<img src="assets/patchtsmixer/overview/4.gif" width="640" height="360"/>

`PatchTSMixer` has a modular design to seamlessly support masked time series pre-training as well as direct time series forecasting.

<img src="assets/patchtsmixer/overview/5.gif" width="640" height="360"/>

## Part 1: Forecasting on Electricity dataset

```python
# Standard
import os
import random

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd
import torch

# First Party
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index
```

### Set seed


```python
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
```

### Load and prepare datasets

In the next cell, please adjust the following parameters to suit your application:
- `PRETRAIN_AGAIN`: Set this to `True` if you want to perform pretraining again. Note that this might take some time depending on the GPU availability. Otherwise, the already pretrained model will be used.
- `dataset_path`: path to local .csv file, or web address to a csv file for the data of interest. Data is loaded with pandas, so anything supported by
`pd.read_csv` is supported: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
- `timestamp_column`: column name containing timestamp information, use None if there is no such column
- `id_columns`: List of column names specifying the IDs of different time series. If no ID column exists, use []
- `forecast_columns`: List of columns to be modeled
- `context_length`: The amount of historical data used as input to the model. Windows of the input time series data with length equal to
`context_length` will be extracted from the input dataframe. In the case of a multi-time series dataset, the context windows will be created
so that they are contained within a single time series (i.e., a single ID).
- `forecast_horizon`: Number of timestamps to forecast in future.
- `train_start_index`, `train_end_index`: the start and end indices in the loaded data which delineate the training data.
- `valid_start_index`, `valid_end_index`: the start and end indices in the loaded data which delineate the validation data.
- `test_start_index`, `test_end_index`: the start and end indices in the loaded data which delineate the test data.
- `num_workers`: Number of dataloder workers in pytorch dataloader.
- `batch_size`: Batch size.
The data is first loaded into a Pandas dataframe and split into training, validation, and test parts. Then the pandas dataframes are converted
to the appropriate torch dataset needed for training.


```python
PRETRAIN_AGAIN = True
# Download ECL data from https://github.com/zhouhaoyi/Informer2020
dataset_path = "~/Downloads/ECL.csv"
timestamp_column = "date"
id_columns = []

context_length = 512
forecast_horizon = 96
num_workers = 16  # Reduce this if you have low number of CPU cores
batch_size = 64  # Adjust according to GPU memory
```


```python
if PRETRAIN_AGAIN:
    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    forecast_columns = list(data.columns[1:])

    # get split
    num_train = int(len(data) * 0.7)
    num_test = int(len(data) * 0.2)
    num_valid = len(data) - num_train - num_test
    border1s = [
        0,
        num_train - context_length,
        len(data) - num_test - context_length,
    ]
    border2s = [num_train, num_train + num_valid, len(data)]

    train_start_index = border1s[0]  # None indicates beginning of dataset
    train_end_index = border2s[0]

    # we shift the start of the evaluation period back by context length so that
    # the first evaluation timestamp is immediately following the training data
    valid_start_index = border1s[1]
    valid_end_index = border2s[1]

    test_start_index = border1s[2]
    test_end_index = border2s[2]

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )
    valid_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=valid_start_index,
        end_index=valid_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        scaling=True,
    )
    tsp.train(train_data)
```


```python
if PRETRAIN_AGAIN:
    train_dataset = ForecastDFDataset(
        tsp.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column="date",
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        tsp.preprocess(valid_data),
        id_columns=id_columns,
        timestamp_column="date",
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        tsp.preprocess(test_data),
        id_columns=id_columns,
        timestamp_column="date",
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
```

 ## Configure the PatchTSMixer model

 The settings below control the different components in the PatchTSMixer model.
  - `num_input_channels`: the number of input channels (or dimensions) in the time series data. This is
    automatically set to the number for forecast columns.
  - `context_length`: As described above, the amount of historical data used as input to the model.
  - `prediction_length`: This is same as the forecast horizon as decribed above.
  - `patch_length`: The patch length for the `PatchTSMixer` model. It is recommended to choose a value that evenly divides `context_length`.
  - `patch_stride`: The stride used when extracting patches from the context window.
  - `d_model`: Hidden feature dimension of the model.
  - `num_layers`: The number of model layers.
  - `dropout`: Dropout probability for all fully connected layers in the encoder.
  - `head_dropout`: Dropout probability used in the head of the model.
  - `mode`: PatchTSMixer operating mode. "common_channel"/"mix_channel". Common-channel works in channel-independent mode. For pretraining, use "common_channel".
  - `scaling`: Per-widow standard scaling. Recommended value: "std".

For full details on the parameters - refer [here](https://huggingface.co/docs/transformers/main/en/model_doc/patchtsmixer)

We recommend that you only adjust the values in the next cell.


```python
patch_length = 8
if PRETRAIN_AGAIN:
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        num_input_channels=len(forecast_columns),
        patch_stride=patch_length,
        d_model=16,
        num_layers=8,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        mode="common_channel",
        scaling="std",
    )
    model = PatchTSMixerForPrediction(config)
```

 ## Train model

 Trains the PatchTSMixer model based on the direct forecasting strategy.


```python
if PRETRAIN_AGAIN:
    training_args = TrainingArguments(
        output_dir="./checkpoint/patchtsmixer/electricity/pretrain/output/",
        overwrite_output_dir=True,
        learning_rate=0.001,
        num_train_epochs=100,  # For a quick test of this notebook, set it to 1
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir="./checkpoint/patchtsmixer/electricity/pretrain/logs/",  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"],
        # max_steps=20,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    )

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
    )

    # pretrain
    trainer.train()
```

<div>
  <progress value='2450' max='7000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [2450/7000 21:35 < 40:08, 1.89 it/s, Epoch 35/100]
</div>

| Epoch | Training Loss | Validation Loss |
|-------|---------------|------------------|
|   1   |    0.247100   |     0.141067     |
|   2   |    0.168600   |     0.127757     |
|   3   |    0.156500   |     0.122327     |
|   4   |    0.150300   |     0.118918     |
|   5   |    0.146000   |     0.116496     |
|   6   |    0.143100   |     0.114968     |
|   7   |    0.140800   |     0.113678     |
|   8   |    0.139200   |     0.113057     |
|   9   |    0.137900   |     0.112405     |
|   10  |    0.136900   |     0.112225     |
|   11  |    0.136100   |     0.112087     |
|   12  |    0.135400   |     0.112330     |
|   13  |    0.134700   |     0.111778     |
|   14  |    0.134100   |     0.111702     |
|   15  |    0.133700   |     0.110964     |
|   16  |    0.133100   |     0.111164     |
|   17  |    0.132800   |     0.111063     |
|   18  |    0.132400   |     0.111088     |
|   19  |    0.132100   |     0.110905     |
|   20  |    0.131800   |     0.110844     |
|   21  |    0.131300   |     0.110831     |
|   22  |    0.131100   |     0.110278     |
|   23  |    0.130700   |     0.110591     |
|   24  |    0.130600   |     0.110319     |
|   25  |    0.130300   |     0.109900     |
|   26  |    0.130000   |     0.109982     |
|   27  |    0.129900   |     0.109975     |
|   28  |    0.129600   |     0.110128     |
|   29  |    0.129300   |     0.109995     |
|   30  |    0.129100   |     0.109868     |
|   31  |    0.129000   |     0.109928     |
|   32  |    0.128700   |     0.109823     |
|   33  |    0.128500   |     0.109863     |
|   34  |    0.128400   |     0.109794     |
|   35  |    0.128100   |     0.109945     |


 ## Evaluate model on the test set.



```python
if PRETRAIN_AGAIN:
    results = trainer.evaluate(test_dataset)
    print("Test result:")
    print(results)
```


<div>

  <progress value='21' max='21' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [21/21 00:03]
</div>



    Test result:
    {'eval_loss': 0.12884521484375, 'eval_runtime': 5.7532, 'eval_samples_per_second': 897.763, 'eval_steps_per_second': 3.65, 'epoch': 35.0}


We get MSE score of 0.128 which is the SOTA result on the Electricity data.

 ## Save model


```python
if PRETRAIN_AGAIN:
    save_dir = "patchtsmixer/electricity/model/pretrain/"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
```

# Part 2: Transfer Learning from Electricity to ETTH2

In this section, we will demonstrate the transfer learning capability of the `PatchTSMixer` model.
We use the model pretrained on Electricity dataset to do zeroshot testing on ETTH2 dataset.


In Transfer Learning,  we pretrain the model for a forecasting task on a `source` dataset (already done on `Electricity` data). Then, we will use the
 pretrained model for zero-shot forecasting on a `target` dataset. The zero-shot forecasting
 performance will denote the `test` performance of the model in the `target` domain, without any
 training on the target domain. Subsequently, we will do linear probing and (then) finetuning of
 the pretrained model on the `train` part of the target data, and will validate the forecasting
 performance on the `test` part of the target data. In this example, the source dataset is the Electricity dataset and the target dataset is ETTH2

## Transfer Learing on `ETTh2` data. All evaluations are on the `test` part of the `ETTh2` data.
Step 1: Directly evaluate the electricity-pretrained model. This is the zero-shot performance.  
Step 2: Evalute after doing linear probing.  
Step 3: Evaluate after doing full finetuning.  

### Load ETTh2 data


```python
dataset = "ETTh2"
```


```python
print(f"Loading target dataset: {dataset}")
dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
train_start_index = None  # None indicates beginning of dataset
train_end_index = 12 * 30 * 24

# we shift the start of the evaluation period back by context length so that
# the first evaluation timestamp is immediately following the training data
valid_start_index = 12 * 30 * 24 - context_length
valid_end_index = 12 * 30 * 24 + 4 * 30 * 24

test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
test_end_index = 12 * 30 * 24 + 8 * 30 * 24
```

    Loading target dataset: ETTh2



```python
data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

train_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=train_start_index,
    end_index=train_end_index,
)
valid_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=valid_start_index,
    end_index=valid_end_index,
)
test_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=test_start_index,
    end_index=test_end_index,
)

tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    scaling=True,
)
tsp.train(train_data)
```




    TimeSeriesPreprocessor {
      "context_length": 64,
      "feature_extractor_type": "TimeSeriesPreprocessor",
      "id_columns": [],
      "input_columns": [
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      "output_columns": [
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      "prediction_length": null,
      "processor_class": "TimeSeriesPreprocessor",
      "scaler_dict": {
        "0": {
          "copy": true,
          "feature_names_in_": [
            "HUFL",
            "HULL",
            "MUFL",
            "MULL",
            "LUFL",
            "LULL",
            "OT"
          ],
          "mean_": [
            41.53683496078959,
            12.273452896210882,
            46.60977329964991,
            10.526153112865156,
            1.1869920139097505,
            -2.373217913729173,
            26.872023494265697
          ],
          "n_features_in_": 7,
          "n_samples_seen_": 8640,
          "scale_": [
            10.448841072588488,
            4.587112566531959,
            16.858190332598408,
            3.018605566682919,
            4.641011217319063,
            8.460910779279644,
            11.584718923414682
          ],
          "var_": [
            109.17827976021215,
            21.04160169803542,
            284.19858129011436,
            9.111979567209104,
            21.538985119281367,
            71.58701121493046,
            134.20571253452223
          ],
          "with_mean": true,
          "with_std": true
        }
      },
      "scaling": true,
      "time_series_task": "forecasting",
      "timestamp_column": "date"
    }




```python
train_dataset = ForecastDFDataset(
    tsp.preprocess(train_data),
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
valid_dataset = ForecastDFDataset(
    tsp.preprocess(valid_data),
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
test_dataset = ForecastDFDataset(
    tsp.preprocess(test_data),
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
```

## Zero-shot forecasting on `ETTh2`


```python
print("Loading pretrained model")
finetune_forecast_model = PatchTSMixerForPrediction.from_pretrained(
    "patchtsmixer/electricity/model/pretrain/"
)
print("Done")
```

    Loading pretrained model
    Done



```python
finetune_forecast_args = TrainingArguments(
    output_dir="./checkpoint/patchtsmixer/transfer/finetune/output/",
    overwrite_output_dir=True,
    learning_rate=0.0001,
    num_train_epochs=100,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=num_workers,
    report_to="tensorboard",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    logging_dir="./checkpoint/patchtsmixer/transfer/finetune/logs/",  # Make sure to specify a logging directory
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
)

# Create a new early stopping callback with faster convergence properties
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.001,  # Minimum improvement required to consider as improvement
)

finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model,
    args=finetune_forecast_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)

print("\n\nDoing zero-shot forecasting on target data")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data zero-shot forecasting result:")
print(result)
```

    
    
    Doing zero-shot forecasting on target data





<div>

  <progress value='22' max='11' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [11/11 02:52]
</div>



    Target data zero-shot forecasting result:
    {'eval_loss': 0.3038313388824463, 'eval_runtime': 1.8364, 'eval_samples_per_second': 1516.562, 'eval_steps_per_second': 5.99}


### By a direct zeroshot, we get MSE of 0.3 which is near to the SOTA result. Lets see, how we can do a simple linear probing to match the SOTA results.

## Target data `ETTh2` linear probing
We can do a quick linear probing on the `train` part of the target data to see any possible `test` performance improvement. 


```python
# Freeze the backbone of the model
for param in finetune_forecast_trainer.model.model.parameters():
    param.requires_grad = False

print("\n\nLinear probing on the target data")
finetune_forecast_trainer.train()
print("Evaluating")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data head/linear probing result:")
print(result)
```

    
    
    Linear probing on the target data

<div>

  <progress value='416' max='3200' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [ 416/3200 01:01 < 06:53, 6.73 it/s, Epoch 13/100]
</div>

| Epoch | Training Loss | Validation Loss |
|-------|---------------|------------------|
|   1   |    0.447000   |     0.216436     |
|   2   |    0.438600   |     0.215667     |
|   3   |    0.429400   |     0.215104     |
|   4   |    0.422500   |     0.213820     |
|   5   |    0.418500   |     0.213585     |
|   6   |    0.415000   |     0.213016     |
|   7   |    0.412000   |     0.213067     |
|   8   |    0.412400   |     0.211993     |
|   9   |    0.405900   |     0.212460     |
|  10   |    0.405300   |     0.211772     |
|  11   |    0.406200   |     0.212154     |
|  12   |    0.400600   |     0.212082     |
|  13   |    0.405300   |     0.211458     |


    Evaluating






<div>

  <progress value='11' max='11' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [11/11 00:00]
</div>



    Target data head/linear probing result:
    {'eval_loss': 0.27119266986846924, 'eval_runtime': 1.7621, 'eval_samples_per_second': 1580.478, 'eval_steps_per_second': 6.242, 'epoch': 13.0}


### By doing a simple linear probing, MSE decreased from 0.3 to 0.271 achiving the SOTA results.



```python
save_dir = f"patchtsmixer/electricity/model/transfer/{dataset}/model/linear_probe/"
os.makedirs(save_dir, exist_ok=True)
finetune_forecast_trainer.save_model(save_dir)

save_dir = f"patchtsmixer/electricity/model/transfer/{dataset}/preprocessor/"
os.makedirs(save_dir, exist_ok=True)
tsp.save_pretrained(save_dir)
```




    ['patchtsmixer/electricity/model/transfer/ETTh2/preprocessor/preprocessor_config.json']



Lets now see, if we get any more improvements by doing a full finetune.

## Target data `ETTh2` full finetune

We can do a full model finetune (instead of probing the last linear layer as shown above) on the `train` part of the target data to see a possible `test` performance improvement.


```python
# Reload the model
finetune_forecast_model = PatchTSMixerForPrediction.from_pretrained(
    "patchtsmixer/electricity/model/pretrain/"
)
finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model,
    args=finetune_forecast_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)
print("\n\nFinetuning on the target data")
finetune_forecast_trainer.train()
print("Evaluating")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data full finetune result:")
print(result)
```

    
    
    Finetuning on the target data




<div>

  <progress value='288' max='3200' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [ 288/3200 00:44 < 07:34, 6.40 it/s, Epoch 9/100]
</div>

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
|   1   |    0.432900   |     0.215200    |
|   2   |    0.416700   |     0.210919    |
|   3   |    0.401400   |     0.209932    |
|   4   |    0.392900   |     0.208808    |
|   5   |    0.388100   |     0.209692    |
|   6   |    0.375900   |     0.209546    |
|   7   |    0.370000   |     0.210207    |
|   8   |    0.367000   |     0.211601    |
|   9   |    0.359400   |     0.211405    |



    Evaluating





<div>

  <progress value='11' max='11' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [11/11 00:00]
</div>



    Target data full finetune result:
    {'eval_loss': 0.2734043300151825, 'eval_runtime': 1.5853, 'eval_samples_per_second': 1756.725, 'eval_steps_per_second': 6.939, 'epoch': 9.0}


There is not much improvement with ETTH2 dataset with full finetuning. Lets save the model anyway.


```python
save_dir = f"patchtsmixer/electricity/model/transfer/{dataset}/model/fine_tuning/"
os.makedirs(save_dir, exist_ok=True)
finetune_forecast_trainer.save_model(save_dir)
```


## Summary 
In this blog, we presented a step-by-step guide on leveraging PatchTSMixer for tasks related to forecasting and transfer learning. We intend to facilitate the seamless integration of the PatchTSMixer HF model for your forecasting use cases. We trust that this content serves as a useful resource to expedite your adoption of PatchTSMixer. Thank you for tuning in to our blog, and we hope you find this information beneficial for your projects.
